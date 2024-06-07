import torch
import torch.nn as nn
import math


class Block(nn.Module):
    """
    A convolutional block optionally conditioned on external data using FiLM layers.
    Includes two convolutional layers with ReLU activation and batch normalization.

    Parameters:
    - inputs (int): Number of input channels.
    - middles (int): Number of middle layer channels.
    - outs (int): Number of output channels.
    - use_film (bool): Whether to apply FiLM layers.
    - film_dim (int): Dimension of the conditioning vector for FiLM layers.
    """

    def __init__(
        self,
        inputs: int = 1,
        middles: int = 64,
        outs: int = 64,
        use_film: bool = False,
        film_dim: int = None,
    ):
        super().__init__()
        self.use_film = use_film
        self.conv1 = nn.Conv1d(inputs, middles, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(middles, outs, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(outs)

        if self.use_film:
            self.film1 = FiLM(film_dim, middles)
            self.film2 = FiLM(film_dim, outs)

    def forward(
        self, x: torch.Tensor, film_params: torch.Tensor = None
    ) -> torch.Tensor:
        if self.use_film:
            x = self.film1(x, film_params)
        x = self.relu(self.conv1(x))
        if self.use_film:
            x = self.film2(x, film_params)
        x = self.relu(self.bn(self.conv2(x)))
        return x

class SinusoidalPositionalEncoding(nn.Module):
    """
    Generates and applies sinusoidal positional encodings to the input feature maps.
    """
    def __init__(self, channels, length):
        super().__init__()
        self.encoding = nn.Parameter(self.create_positional_encoding(channels, length), requires_grad=False)

    @staticmethod
    def create_positional_encoding(channels, length):
        if channels % 2 == 1:
            channels -= 1

        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe = torch.zeros(length, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.transpose(0, 1).unsqueeze(0) 

    def forward(self, x):
        return x + self.encoding

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer to conditionally modulate the feature maps.
    """

    def __init__(self, film_dim: int, num_features: int):
        super().__init__()
        self.scale_transform = nn.Linear(film_dim, num_features)
        self.shift_transform = nn.Linear(film_dim, num_features)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale = self.scale_transform(z).unsqueeze(2)
        shift = self.shift_transform(z).unsqueeze(2)
        return x * scale + shift


class UNet(nn.Module):
    def __init__(self, film_dim: int, length: int):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(64, length)
        self.en1 = Block(64, 64, 64, use_film=True, film_dim=film_dim)
        self.en2 = Block(64, 128, 128, use_film=True, film_dim=film_dim)
        self.en3 = Block(128, 256, 256, use_film=True, film_dim=film_dim)
        self.en4 = Block(256, 512, 512, use_film=True, film_dim=film_dim)
        self.en5 = Block(512, 1024, 512, use_film=True, film_dim=film_dim)

        self.upsample4 = nn.ConvTranspose1d(
            512, 512, kernel_size=4, stride=2, padding=1
        )
        self.de4 = Block(1024, 512, 256, use_film=True, film_dim=film_dim)

        self.upsample3 = nn.ConvTranspose1d(
            256, 256, kernel_size=4, stride=2, padding=1
        )
        self.de3 = Block(512, 256, 128, use_film=True, film_dim=film_dim)

        self.upsample2 = nn.ConvTranspose1d(
            128, 128, kernel_size=4, stride=2, padding=1
        )
        self.de2 = Block(256, 128, 64, use_film=True, film_dim=film_dim)

        self.upsample1 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1)
        self.de1 = Block(128, 64, 64, use_film=True, film_dim=film_dim)

        self.conv_last = nn.Conv1d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        x = self.en1(x, film_params)
        x = self.en2(x, film_params)
        x = self.en3(x, film_params)
        x = self.en4(x, film_params)
        x = self.en5(x, film_params)

        x = self.upsample4(x)
        x = self.de4(x, film_params)

        x = self.upsample3(x)
        x = self.de3(x, film_params)

        x = self.upsample2(x)
        x = self.de2(x, film_params)

        x = self.upsample1(x)
        x = self.de1(x, film_params)

        x = self.conv_last(x)
        return x
