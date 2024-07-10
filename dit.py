from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class DiTConfig:
    image_size: int = 32  # Size of the input image
    patch_size: int = 4  # Size of each patch
    in_channels: int = 4
    out_channels: int = 4  # Usually same as in_channels
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768  # Hidden dimension size
    num_classes: int = 10

    def __post_init__(self):
        self.block_size = (self.image_size // self.patch_size) ** 2


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

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # Layer norms
        self.norm1 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)

        # Self-attention
        self.attn = SelfAttention(config)

        # MLP
        self.mlp = MLP(config)

        # AdaLN-Zero modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class SelfAttention(nn.Module):
    """Causal multihead self-attention implementation.

    Attributes:
        c_attn (nn.Linear): Linear layer to create queries, keys, and values.
        c_proj (nn.Linear): Linear layer to project the output of attention back to the embedding dimension.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd
        )  # projects embedding to bigger space to extract Q, K, V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq length, embedding depth

        qkv = self.c_attn(x)
        # Split the combined qkv matrix and reshape it to get individual q, k, v matrices
        q, k, v = qkv.split(self.n_embd, dim=2)
        # q, k, v shapes: each (B, T, C)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # q, k, v shapes after reshape and transpose: each (B, n_head, T, C // n_head)

        y = F.scaled_dot_product_attention(q, k, v)
        # y shape: (B, n_head, T, C // n_head)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # y shape after transpose and reshape: (B, T, C)

        y = self.c_proj(y)
        # y shape after projection: (B, T, C)

        return y


class Patchify(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.projection = nn.Linear(
            config.patch_size * config.patch_size * config.in_channels, config.n_embd
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        assert (
            h == w == self.config.image_size
        ), f"Input image size ({h}x{w}) doesn't match the expected size ({self.config.image_size}x{self.config.image_size})"
        assert (
            h % p == 0 and w % p == 0
        ), f"Image dimensions must be divisible by the patch size {p}"

        # Reshape and permute to get patches
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b, -1, p * p * c)

        # Project patches to embedding dimension
        x = self.projection(x)

        return x

    @property
    def num_patches(self):
        """Calculate the number of patches based on the input shape and patch size."""
        h, w = 256, 256  # Assuming 256x256 images as mentioned in the text
        return (h // self.patch_size) * (w // self.patch_size)


class FirstLayer(nn.Module):
    """
    Process the input through the first layer of the model.

    Args:
        x (torch.Tensor): Input image tensor.
        t (torch.Tensor): Timestep tensor.
        y (torch.Tensor): Class labels tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Processed image tensor and conditioning tensor.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        # Patchify
        self.patchify = Patchify(config)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        # Timestep embedding
        self.t_embedder = nn.Sequential(
            nn.Linear(1, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        # Class embedding
        self.y_embedder = nn.Embedding(config.num_classes, config.n_embd)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W)
            t (torch.Tensor): Timestep tensor of shape (N,)
            y (torch.Tensor): Class labels tensor of shape (N,)

        Returns:
            tuple: (x, c) where x is the patched and embedded input, and c is the combined timestep and class embedding
        """
        # Patchify x
        x = self.patchify(x)  # Shape: (N, block_size, n_embd)
        x = x + self.pos_embed  # Add positional embedding

        # Embed timestep
        t_emb = self.t_embedder(t.float().unsqueeze(-1))  # Shape: (N, n_embd)

        # Embed class labels
        y_emb = self.y_embedder(y.long())  # Shape: (N, n_embd)

        # Combine timestep and class embeddings
        c = t_emb + y_emb  # Shape: (N, n_embd)

        return x, c


class FinalLayer(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            config.n_embd, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            config.n_embd,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# Create a simple DiT model with 2 blocks
class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) model.

    This class implements the main architecture of the DiT model, which combines
    elements of transformers and diffusion models for image generation tasks.

    Attributes:
        config (DiTConfig): Configuration object containing model parameters.
        patchify (Patchify): Module to convert input images into patches.
        pos_embedding (nn.Parameter): Learnable positional embeddings.
        blocks (nn.ModuleList): List of DiTBlock modules.
        final_layer (FinalLayer): Final processing layer of the model.

    Args:
        config (DiTConfig): Configuration object for the DiT model.

    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.patchify = Patchify(config)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        self.first_layer = FirstLayer(config)
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_layer)])
        self.final_layer = FinalLayer(config)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the DiT model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, image_size, image_size).
            t (torch.Tensor): Timestep tensor of shape (batch_size,).
            y (torch.Tensor): Class labels tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output image tensor of shape (batch_size, out_channels, image_size, image_size).

        Raises:
            ValueError: If input tensor shapes are inconsistent with the model configuration.

        Note:
            This method processes the input through the following steps:
            1. Patchify and embed the input image.
            2. Add positional embeddings.
            3. Process through DiT blocks.
            4. Apply the final layer.
            5. Reshape the output back to image format.

        """
        x, c = self.first_layer(x, t, y)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        # x shape: (batch_size, block_size, patch_size*patch_size*out_channels)

        # Reshape output to image format
        b, t, _ = x.shape
        p = self.config.patch_size
        h = w = self.config.image_size // p
        x = x.view(b, h, w, p, p, self.config.out_channels)
        # x shape: (batch_size, h//p, w//p, patch_size, patch_size, out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        # x shape: (batch_size, out_channels, h//p, patch_size, w//p, patch_size)
        x = x.view(
            b, self.config.out_channels, self.config.image_size, self.config.image_size
        )
        # x shape: (batch_size, out_channels, image_size, image_size)

        return x


if __name__ == "__main__":
    # Set up the configuration
    config = DiTConfig(
        image_size=28,
        patch_size=4,
        in_channels=1,
        out_channels=1,
        n_layer=3,
        n_head=4,
        n_embd=128,
        num_classes=10,
    )

    # Instantiate the model
    model = DiT(config)

    # Create sample inputs
    batch_size = 8
    x = torch.randn(
        batch_size, config.in_channels, config.image_size, config.image_size
    ).float()
    t = torch.randint(0, 1000, (batch_size,)).float()  # Assuming 1000 timesteps
    y = torch.randint(0, config.num_classes, (batch_size,)).float()

    # Forward pass
    try:
        output = model(x, t, y)
        print("Forward pass successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Number of patches (block size): {config.block_size}")

        # Check output shape
        expected_shape = (
            batch_size,
            config.out_channels,
            config.image_size,
            config.image_size,
        )
        assert (
            output.shape == expected_shape
        ), f"Output shape {output.shape} doesn't match expected shape {expected_shape}"
        print("Output shape is correct.")

        # Check if output values are within a reasonable range
        assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
        print("Output values are finite.")

        # Check if the model is responsive to different inputs
        output2 = model(x, t + 1, y)
        assert not torch.allclose(
            output, output2
        ), "Model output doesn't change with different timesteps"
        print("Model is responsive to different timesteps.")

        print("All tests passed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
