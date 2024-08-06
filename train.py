import gc
import logging
import multiprocessing
import os
import warnings
from dataclasses import dataclass
from octo_transformer import OctoConfig


import pytorch_lightning as pl
import torch
import wandb
from dataset import ShardedLeRobotDataset
from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split


from dit import DiT, DiTConfig

multiprocessing.set_start_method("spawn", force=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@dataclass
class TrainerConfig:
    batch_size: int = 32
    lr: float = 1e-4
    obs_horizon: int = 5
    pred_horizon: int = 10
    image_size: int = 96
    patch_size: int = 8
    action_dim: int = 2
    agent_state_dim: int = 2
    n_dit_layer: int = 8
    n_dit_head: int = 8
    n_dit_embd: int = 512
    n_octo_layer: int = 6
    n_octo_head: int = 6
    n_ocot_embd: int = 384


class DiTLightning(pl.LightningModule):
    def __init__(self, trainer_config: TrainerConfig):
        super().__init__()
        self.save_hyperparameters()
        self.trainer_config = trainer_config

        octo_config = OctoConfig(
            obs_horizon=trainer_config.obs_horizon,
            n_embd=trainer_config.n_ocot_embd,
            image_size=trainer_config.image_size,
            patch_size=trainer_config.patch_size,
            n_layer=trainer_config.n_octo_layer,
            n_head=trainer_config.n_octo_head,
        )

        dit_config = DiTConfig(
            action_dim=trainer_config.action_dim,
            n_layer=trainer_config.n_dit_layer,
            n_head=trainer_config.n_dit_head,
            n_embd=trainer_config.n_dit_embd,
            pred_horizon=trainer_config.pred_horizon,
        )
        self.net = DiT(dit_config, octo_config)
        self.num_train_timesteps = 1000
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )
        self.lr = trainer_config.lr
        self.batch_size = trainer_config.batch_size

    def forward(self, x, t, class_labels):
        return self.net(x, t, class_labels)

    def _shared_step(self, batch, batch_idx):
        observation_states = batch["observation_states"] / 512
        observation_images = batch["observation_images"]
        prediction_actions = batch["prediction_actions"] / 512

        noise = torch.randn_like(prediction_actions)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (prediction_actions.shape[0],),
            device=self.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(
            prediction_actions, noise, timesteps
        )

        noise_pred = self.net(
            noisy_latents, observation_images, observation_states, timesteps
        )

        loss = F.mse_loss(noise_pred, noise)
        if not torch.isfinite(loss):
            print(f"Non-finite loss detected: {loss}")
            return None
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if not torch.isfinite(loss):
            print(f"Non-finite loss detected: {loss}")
            return None

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)


if __name__ == "__main__":
    wandb.finish()
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="diffusion_policy", save_code=True, mode="online")
    wandb_logger = WandbLogger(log_model=True)

    prediction_horizon = 16
    observation_horizon = 2

    trainer_config = TrainerConfig(
        batch_size=64,
        lr=3e-4,
        obs_horizon=2,
        pred_horizon=16,
        image_size=96,
        patch_size=8,
        action_dim=2,
        n_dit_layer=8,
        n_dit_head=8,
        n_dit_embd=512,
        n_ocot_embd=384,
    )

    dataset = ShardedLeRobotDataset(
        "lerobot/pusht",
        prediction_horizon=trainer_config.pred_horizon,
        observation_horizon=trainer_config.obs_horizon,
        shard_size=1000,
    )

    # Create a 70/30 split for train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = DiTLightning(trainer_config)

    checkpoint_dir = "checkpoints_dp"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="dp-{epoch:02d}-{step}-{train_loss:.2f}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=10,
        save_on_train_epoch_end=True,
        save_last=True,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        precision="bf16-mixed",
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=25,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=25,
        persistent_workers=True,
    )

    # Check for the latest checkpoint
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
            if not os.path.exists(latest_checkpoint):
                latest_checkpoint = os.path.join(
                    checkpoint_dir, sorted(checkpoints)[-1]
                )

    if latest_checkpoint and os.path.isfile(latest_checkpoint):
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.fit(
            model, train_dataloader, val_dataloader, ckpt_path=latest_checkpoint
        )
    else:
        print("Starting training from scratch")
        trainer.fit(model, train_dataloader, val_dataloader)

    wandb.finish()
