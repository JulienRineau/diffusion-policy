from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


@dataclass
class UnetConfig:
    in_channels: int = 1
    out_channels: int = 1
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, UnetConfig):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = UnetConfig.in_channels  # init the first channel

        # Down part
        for feature in UnetConfig.features:
            self.downs.append(DoubleConv(in_channels, feature))
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
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(
            UnetConfig.features[-1], UnetConfig.features[-1] * 2
        )
        self.final_conv = nn.Conv2d(
            UnetConfig.features[0], UnetConfig.out_channels, kernel_size=1
        )  # weighted sum across all input channels for each spatial position

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # take all elements, in reverse order

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # transposed convolution (upsampling) operation.
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # DoubleConv operation

        logits = self.final_conv(x)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNET(UnetConfig()).to(device)
    x = torch.randn((8, 1, 28, 28)).to(device)  # Image should be (B, C, W, H)

    with torch.no_grad():
        preds = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
