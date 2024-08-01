import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


@dataclass
class DiTConfig:
    obs_dim: int = 3  # Dimension of observation (excluding agent position)
    action_dim: int = 4  # Dimension of action
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512  # Hidden dimension size
    obs_horizon: int = 2  # Number of observation steps
    pred_horizon: int = 10  # Number of prediction steps
    agent_pos_dim: int = 2  # Dimension of agent position


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
        return embeddings


class FirstLayer(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        # Action embedding
        self.act_embedding = nn.Linear(config.action_dim, config.n_embd)
        # Action positional embedding
        self.x_pos_embed = nn.Parameter(
            self._get_sinusoidal_embeddings(config.pred_horizon, config.n_embd)
        )

        # Observation embedding
        self.obs_embedding = nn.Linear(
            config.obs_dim + config.agent_pos_dim, config.n_embd
        )
        # Observation positional embedding
        self.obs_pos_embed = nn.Parameter(
            self._get_sinusoidal_embeddings(config.obs_horizon, config.n_embd)
        )
        self.obs_projection = nn.Linear(
            config.obs_horizon * config.n_embd, config.n_embd
        )

        # Timestep embedding
        self.t_embedder = nn.Sequential(
            SinusoidalPositionEmbeddings(config.n_embd),
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.n_embd, config.n_embd),
        )

    def _get_sinusoidal_embeddings(self, n_position, d_hid):
        position = torch.arange(n_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2) * -(math.log(10000.0) / d_hid))
        pos_embedding = torch.zeros(1, n_position, d_hid)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embedding

    def forward(
        self,
        x: torch.Tensor,
        obs: torch.Tensor,
        agent_pos: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = obs.shape

        x = self.act_embedding(x)
        x = x + self.x_pos_embed  # Add positional embedding

        # Concatenate observation and agent positions
        obs_features = torch.cat([obs, agent_pos], dim=-1)

        # Create observation embedding
        obs_emb = self.obs_embedding(obs_features)  # Shape: (B, T, n_embd)
        obs_emb = obs_emb + self.obs_pos_embed

        # Flatten observation embedding
        obs_cond = obs_emb.view(B, -1)  # Shape: (B, T * n_embd)

        # Project flattened observation to match timestep embedding dimension
        obs_cond = self.obs_projection(obs_cond)  # Shape: (B, n_embd)

        # Embed timestep
        t_emb = self.t_embedder(t)  # Shape: (B, n_embd)

        # Combine observation and timestep embeddings
        c = obs_cond + t_emb  # Shape: (B, n_embd)

        return x, c


class DiTBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # Layer norms
        self.norm1 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_head, batch_first=True
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

        # AdaLN-Zero modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        modulation = self.adaLN_modulation(c)  # Shape: (B, 6*C)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(6, dim=1)

        # Self-attention
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(
                self.norm1(x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)),
                self.norm1(x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)),
                self.norm1(x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)),
            )[0]
        )

        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.norm2(x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        )

        return x


class FinalLayer(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            config.n_embd, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(config.n_embd, config.action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        modulation = self.adaLN_modulation(c)  # Shape: (B, 2*C)
        shift, scale = modulation.chunk(2, dim=1)
        x = self.norm_final(x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(self, config: DiTConfig, skip_init=False):
        super().__init__()
        self.config = config

        self.first_layer = FirstLayer(config)
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_layer)])
        self.final_layer = FinalLayer(config)

        if not skip_init:
            self.__init_weights()

    def __init_weights(self):
        # Initialize final layer weights to zero
        init.zeros_(self.final_layer.linear.weight)
        init.zeros_(self.final_layer.linear.bias)

        # Initialize adaLN weights to zero
        init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

        for block in self.blocks:
            init.zeros_(block.adaLN_modulation[-1].weight)
            init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        obs: torch.Tensor,
        agent_pos: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        x, c = self.first_layer(noisy_actions, obs, agent_pos, t)

        for block in self.blocks:
            x = block(x, c)

        denoised_actions = self.final_layer(x, c)
        return denoised_actions


# Test the model
if __name__ == "__main__":
    config = DiTConfig(
        obs_dim=2,
        action_dim=2,
        n_layer=8,
        n_head=8,
        n_embd=512,
        obs_horizon=5,
        pred_horizon=10,
        agent_pos_dim=2,
    )
    model = DiT(config, skip_init=True)

    batch_size = 8
    obs = torch.randn(batch_size, config.obs_horizon, config.obs_dim).float()
    agent_pos = torch.randn(
        batch_size, config.obs_horizon, config.agent_pos_dim
    ).float()
    t = torch.randint(0, 1000, (batch_size,)).float()

    noisy_actions = torch.randn(
        batch_size, config.pred_horizon, config.action_dim
    ).float()

    output = model(noisy_actions, obs, agent_pos, t)
    print("Forward pass successful!")
    print(f"Input shape: {noisy_actions.shape}")
    print(f"Output shape: {output.shape}")

    expected_shape = (batch_size, config.pred_horizon, config.action_dim)
    assert (
        output.shape == expected_shape
    ), f"Output shape {output.shape} doesn't match expected shape {expected_shape}"
    print("Output shape is correct.")

    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
    print("Output values are finite.")

    output2 = model(noisy_actions, obs, agent_pos, t + 1)
    assert not torch.allclose(
        output, output2
    ), "Model output doesn't change with different timesteps"
    print("Model is responsive to different timesteps.")

    print("All tests passed successfully!")
