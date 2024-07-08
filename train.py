import pytorch_lightning as pl
import torch
import torchvision
import wandb
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

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

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        noise_amount = torch.rand(x.shape[0], device=self.device)
        noisy_x = corrupt(x, noise_amount)
        pred = self.net(noisy_x)
        loss = self.loss_fn(pred, x)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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
        max_epochs=3,
        logger=wandb_logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(model, train_dataloader)
    wandb.finish()
