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
from dataset import CustomLeRobotDataset
from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dit import DiT, DiTConfig

multiprocessing.set_start_method("spawn", force=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@dataclass
class TrainerConfig:
    batch_size: int = 32
    lr: float = 1e-4


class DiTLightning(pl.LightningModule):
    def __init__(self, dit_config, octo_config, trainer_config: TrainerConfig):
        super().__init__()
        self.save_hyperparameters()
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

    def training_step(self, batch, batch_idx):
        (
            observation_states,
            observation_actions,
            observation_images,
            prediction_actions,
        ) = batch

        noise = torch.randn_like(observation_actions)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (observation_actions.shape[0],),
            device=self.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(
            observation_actions, noise, timesteps
        )  # TODO: verify is does not modify observation_actions inplace

        noise_pred = self.net(
            noisy_latents, observation_images, timesteps, observation_states, timesteps
        )

        loss = F.mse_loss(noise_pred, prediction_actions)
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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="diffusion_policy", save_code=True, mode="online")
    wandb_logger = WandbLogger(log_model=True)

    dataset = CustomLeRobotDataset(
        "lerobot/pusht", prediction_horizon=16, observation_horizon=2
    )

    trainer_config = TrainerConfig(batch_size=64, lr=1e-4)

    octo_config = OctoConfig(
        obs_horizon=5,
        n_embd=384,
        image_size=96,
        patch_size=8,
    )

    dit_config = DiTConfig(
        action_dim=2,
        n_layer=8,
        n_head=8,
        n_embd=512,
        pred_horizon=10,
    )

    model = DiTLightning(dit_config, octo_config, trainer_config)

    checkpoint_dir = "checkpoints_dp"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="dp-{epoch:02d}-{step}-{train_loss:.2f}",
        monitor="train_loss",
        mode="min",
        save_top_k=10,
        save_on_train_epoch_end=True,
        save_last=True,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=400,
        logger=wandb_logger,
        precision="bf16-mixed",
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=4,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=1,
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
        trainer.fit(model, train_dataloader, ckpt_path=latest_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.fit(model, train_dataloader)

    wandb.finish()
