import pytorch_lightning as pl
import torch
import torchvision
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from unet import UNET, UnetConfig


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


class UNETLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNET(config)
        self.loss_fn = nn.MSELoss()
        self.num_train_timesteps = 1000
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )

    def forward(self, x, t, class_labels):
        return self.net(x, t, class_labels)

    def training_step(self, batch, batch_idx):
        x, class_labels = batch

        # Sample noise to add to the images
        noise = torch.randn_like(x)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.num_train_timesteps, (x.shape[0],), device=self.device
        ).long()

        # Add noise to the input images according to the noise magnitude at each timestep
        noisy_images = self.noise_scheduler.add_noise(x, noise, timesteps)

        noise_pred = self.net(noisy_images, timesteps, class_labels)
        loss = F.mse_loss(noise_pred, noise)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="diffusion-policy", log_model="all")

    dataset = torchvision.datasets.MNIST(
        root="mnist/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    config = UnetConfig()
    model = UNETLightning(config)

    trainer = pl.Trainer(
        max_epochs=6,
        logger=wandb_logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(model, train_dataloader)
    wandb.finish()
