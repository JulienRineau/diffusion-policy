import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class OctoConfig:
    obs_horizon: int = 2
    pred_horizon: int = 64
    act_horizon: int = 12
    agent_state_dim: int = 2
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    image_size: int = 96
    patch_size: int = 8

    @property
    def n_img_tokens(self):
        return (self.image_size // self.patch_size) ** 2


class ImageTokenizer(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.projection = nn.Linear(
            self.patch_size * self.patch_size * 3, config.n_embd
        )
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, config.obs_horizon, 1, config.n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        p = self.patch_size

        assert (
            H % p == 0 and W % p == 0
        ), f"Image dimensions must be divisible by the patch size {p}"

        x = x.view(B * T, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B * T, -1, p * p * C)

        x = self.projection(x)
        x = x.view(B, T, -1, self.config.n_embd)

        # Add temporal embedding
        x = x + self.temporal_embedding

        return x


class AgentStateTokenizer(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.agent_state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config.n_embd),
        )
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, config.obs_horizon, 1, config.n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.mlp(x)
        x = x.view(B, T, 1, -1)

        # Add temporal embedding
        x = x + self.temporal_embedding

        return x


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

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            k.size(-1) + 1e-8
        )
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

        y = torch.matmul(attn_weights, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class Block(nn.Module):
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
        self.agent_state_tokenizer = AgentStateTokenizer(config)

        self.readout_token = nn.Parameter(torch.randn(1, 1, config.n_embd))

        total_tokens = (
            config.obs_horizon * (config.n_img_tokens + 1) + 1
        )  # +1 for readout token

        self.pos_embed = nn.Parameter(
            self._get_sinusoidal_embeddings(total_tokens, config.n_embd)
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

    def _get_sinusoidal_embeddings(self, n_position, d_hid):
        position = torch.arange(n_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2) * -(math.log(10000.0) / d_hid))
        pos_embedding = torch.zeros(1, n_position, d_hid)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embedding

    def forward(self, images: torch.Tensor, agent_state: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape
        assert (
            T == self.config.obs_horizon
        ), f"Expected obs_horizon {self.config.obs_horizon}, got {T}"

        # Tokenize inputs (temporal embeddings are added within the tokenizers)
        image_tokens = self.image_tokenizer(images)
        agent_state_tokens = self.agent_state_tokenizer(agent_state)

        # Combine tokens for each timestep
        combined_tokens = torch.cat([agent_state_tokens, image_tokens], dim=2)

        # Reshape to flatten the temporal and spatial dimensions
        combined_tokens = combined_tokens.view(B, -1, self.config.n_embd)

        # Add readout token
        x = torch.cat([combined_tokens, self.readout_token.expand(B, -1, -1)], dim=1)

        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Return the readout token (now at the end)
        return x[:, -1]


def generate_sample_data(config: OctoConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 4
    images = torch.randn(
        batch_size, config.obs_horizon, 3, config.image_size, config.image_size
    )
    agent_state = torch.randn(batch_size, config.obs_horizon, 2)
    return images, agent_state


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    config = OctoConfig()
    model = OctoTransformer(config)

    images, agent_state = generate_sample_data(config)

    output = model(images, agent_state)
    print(f"Output shape: {output.shape}")

    # Check output shape
    expected_output_shape = (images.shape[0], config.n_embd)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    print("Basic dimension check passed.")

    # Additional checks with detailed dimension prints
    print("\nDetailed Dimension Information:")

    print(f"Input images shape: {images.shape}")
    print(f"Input agent_state shape: {agent_state.shape}")

    image_tokens = model.image_tokenizer(images)
    print(f"Image tokens shape: {image_tokens.shape}")
    expected_image_tokens_shape = (
        images.shape[0],
        config.obs_horizon,
        config.n_img_tokens,
        config.n_embd,
    )
    assert (
        image_tokens.shape == expected_image_tokens_shape
    ), f"Expected image tokens shape {expected_image_tokens_shape}, but got {image_tokens.shape}"

    agent_state_tokens = model.agent_state_tokenizer(agent_state)
    print(f"Agent state tokens shape: {agent_state_tokens.shape}")
    expected_agent_state_tokens_shape = (
        agent_state.shape[0],
        config.obs_horizon,
        1,
        config.n_embd,
    )
    assert (
        agent_state_tokens.shape == expected_agent_state_tokens_shape
    ), f"Expected agent state tokens shape {expected_agent_state_tokens_shape}, but got {agent_state_tokens.shape}"

    print(f"Readout token shape: {model.readout_token.shape}")
    print(f"Positional embedding shape: {model.pos_embed.shape}")

    combined_tokens = torch.cat([agent_state_tokens, image_tokens], dim=2)
    print(f"Combined tokens shape (before reshape): {combined_tokens.shape}")

    combined_tokens_reshaped = combined_tokens.view(images.shape[0], -1, config.n_embd)
    print(f"Combined tokens shape (after reshape): {combined_tokens_reshaped.shape}")

    print("Image and agent state tokenizer checks passed.")

    # Check if weights are different from default initialization
    for name, param in model.named_parameters():
        if "weight" in name:
            assert not torch.allclose(
                param, torch.zeros_like(param)
            ), f"Weights in {name} are close to zero, which suggests they might not have been properly initialized"

    print("Weight initialization check passed.")

    print("All tests passed successfully!")
