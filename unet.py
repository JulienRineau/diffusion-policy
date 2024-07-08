from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math


@dataclass
class UnetConfig:
    in_channels: int = 1
    out_channels: int = 1
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    time_emb_dim: int = 32
    num_classes: int = 10
    class_emb_dim: int = 16


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.cat(
                [embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1
            )

        return embeddings


class ResNetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_emb_dim, class_emb_dim, dropout_rate=0.5
    ):
        super(ResNetBlock, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim + class_emb_dim, out_channels)

        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout_rate)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        identity = self.downsample(x)

        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.gelu(out)

        # Time and class embedding injection
        time_emb = self.time_mlp(t)
        time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)  # Reshape to [B, C, 1, 1]
        out = out + time_emb

        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.gelu(out)
        return out


class UNET(nn.Module):
    def __init__(self, UnetConfig):
        super(UNET, self).__init__()

        self.class_emb = nn.Embedding(UnetConfig.num_classes, UnetConfig.class_emb_dim)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = UnetConfig.in_channels
        time_emb_dim = UnetConfig.time_emb_dim
        class_emb_dim = UnetConfig.class_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(approximate="tanh"),
        )

        # Down part
        for feature in UnetConfig.features:
            self.downs.append(
                ResNetBlock(in_channels, feature, time_emb_dim, class_emb_dim)
            )
            in_channels = feature

        for feature in reversed(UnetConfig.features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(
                ResNetBlock(feature * 2, feature, time_emb_dim, class_emb_dim)
            )

        self.bottleneck = ResNetBlock(
            UnetConfig.features[-1],
            UnetConfig.features[-1] * 2,
            time_emb_dim,
            class_emb_dim,
        )
        self.final_conv = nn.Conv2d(
            UnetConfig.features[0], UnetConfig.out_channels, kernel_size=1
        )

    def forward(self, x, timestep, class_labels):
        skip_connections = []
        t = self.time_mlp(timestep)
        c = self.class_emb(class_labels)

        # Combine time and class embeddings
        t_c = torch.cat([t, c], dim=-1)

        for down in self.downs:
            x = down(x, t_c)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, t_c)
        skip_connections = skip_connections[::-1]  # take all elements, in reverse order

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # transposed convolution (upsampling) operation.
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip, t_c)  # DoubleConv operation

        logits = self.final_conv(x)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = UnetConfig()
    model = UNET(config).to(device)
    x = torch.randn((8, 1, 28, 28)).to(device)  # Image should be (B, C, W, H)
    timesteps = torch.randint(0, 1000, (8,)).to(device)
    class_labels = torch.randint(0, 10, (8,)).to(device)

    with torch.no_grad():
        preds = model(x, timesteps, class_labels)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
