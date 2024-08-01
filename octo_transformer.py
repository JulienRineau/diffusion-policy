import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class OctoConfig:
    obs_horizon: int = 2
    pred_horizon: int = 64
    act_horizon: int = 12
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    n_img_tokens: int = 49
    d_img: int = 128
    d_pos: int = 64

    @property
    def d_combined(self):
        return self.d_img + self.d_pos


class ImageTokenizer(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.d_img, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.conv(x)
        x = self.flatten(x)
        x = x.view(B, T, config.n_img_tokens, config.d_img)
        return x


class PositionTokenizer(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, config.d_pos),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.mlp(x)
        return x.view(B, T, 1, config.d_pos)


class MaskedSelfAttention(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.obs_tokens = config.obs_horizon * (config.n_img_tokens + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq length, embedding depth

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        mask[:, -1] = True  # Allow readout token to attend to every token
        mask[-1, :-1] = (
            False  # Prevent other tokens from attending to the readout token
        )

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """A single transformer block containing a layer normalization, a masked self-attention layer,
    another layer normalization, and an MLP.
    """

    def __init__(self, config: OctoConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MaskedSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    """
    Attributes:
        fc (nn.Linear): The first fully connected layer.
        gelu (nn.GELU): Gaussian Error Linear Unit activation layer.
        proj (nn.Linear): The second fully connected layer projecting back to embedding dimension.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class OctoTransformer(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.config = config

        self.image_tokenizer = ImageTokenizer(config)
        self.position_tokenizer = PositionTokenizer(config)

        self.readout_token = nn.Parameter(torch.randn(1, 1, config.n_embd))
        self.projection = nn.Linear(config.d_combined, config.n_embd)
        self.pos_embedding = nn.Parameter(
            torch.randn(
                1, config.obs_horizon * (config.n_img_tokens + 1) + 1, config.n_embd
            )
        )

        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, images: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape
        assert (
            T == self.config.obs_horizon
        ), f"Expected obs_horizon {self.config.obs_horizon}, got {T}"

        # Tokenize inputs
        image_tokens = self.image_tokenizer(images)
        pos_tokens = self.position_tokenizer(agent_pos)

        # Combine tokens
        combined_tokens = torch.cat([pos_tokens, image_tokens], dim=2)
        combined_tokens = combined_tokens.view(B, -1, self.config.d_combined)

        # Project to embedding dimension
        projected_tokens = self.projection(combined_tokens)

        # Add position embeddings and readout token (at the end)
        x = torch.cat([projected_tokens, self.readout_token.expand(B, -1, -1)], dim=1)
        x = x + self.pos_embedding

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Return the readout token (now at the end)
        return x[:, -1]


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test the OctoTransformer
    logger.info("Testing OctoTransformer")

    # Create a config
    config = OctoConfig()

    # Initialize the model
    model = OctoTransformer(config)
    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Generate random input data
    batch_size = 4
    images = torch.randn(batch_size, config.obs_horizon, 3, 96, 96)
    agent_pos = torch.randn(batch_size, config.obs_horizon, 2)

    # Move model and inputs to the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    images = images.to(device)
    agent_pos = agent_pos.to(device)

    # Perform a forward pass
    try:
        output = model(images, agent_pos)
        logger.info(f"Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        logger.error(f"Forward pass failed with error: {str(e)}")
        raise

    # Check output shape
    expected_shape = (batch_size, config.n_embd)
    if output.shape == expected_shape:
        logger.info(f"Output shape matches expected shape: {expected_shape}")
    else:
        logger.error(
            f"Output shape {output.shape} does not match expected shape {expected_shape}"
        )

    # Check for NaNs in the output
    if torch.isnan(output).any():
        logger.error("Output contains NaNs")
    else:
        logger.info("Output does not contain NaNs")

    # Additional checks (optional)
    assert (
        output.shape == expected_shape
    ), f"Output shape {output.shape} does not match expected shape {expected_shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"

    logger.info("Testing complete")
