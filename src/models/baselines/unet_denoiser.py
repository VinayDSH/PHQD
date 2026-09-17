"""
Full Classical U-Net Latent Denoiser Baseline for PHQD Module 4.2.

Interface:
    Input continuous latent grid z_t [B, 4, 8, 8] + timestep t [B]
                ↓
    2D Convolutional U-Net with Timestep Conditioning
                ↓
    Predicted clean latent grid z_0 [B, 4, 8, 8]

Architectural Context:
- Operates directly on the continuous 4x8x8 latent grid from Module 1.
- Bypasses patch slicing and boundary stitching completely, serving as the
  unconstrained classical latent diffusion baseline.
- Directly feeds into the frozen ConvAutoencoder decoder [B, 4, 8, 8] -> [B, 1, 28, 28].
"""

from typing import Tuple, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hqcld_denoiser import sinusoidal_timestep_embedding


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm, GELU, and additive timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.act = nn.GELU()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        # Inject timestep conditioning additively across spatial dimensions
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return self.act(h)


class UNetLatentDenoiser(nn.Module):
    """Classical 2D Convolutional U-Net for continuous latent grids [B, 4, 8, 8]."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 16,
        time_emb_dim: int = 32,
    ):
        """Initialize U-Net Latent Denoiser.
        
        Args:
            in_channels: Latent grid channels (default: 4).
            out_channels: Output clean latent channels (default: 4).
            base_channels: Base channel multiplier (default: 16).
            time_emb_dim: Timestep embedding dimension (default: 32).
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.time_emb_dim = int(time_emb_dim)

        ch1 = self.base_channels
        ch2 = self.base_channels * 2

        # Timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        # Encoder level 1: 8x8 -> 8x8
        self.enc1 = ConvBlock(in_channels, ch1, self.time_emb_dim)

        # Downsample: 8x8 -> 4x4
        self.down = nn.Conv2d(ch1, ch1, kernel_size=3, stride=2, padding=1)

        # Bottleneck: 4x4
        self.mid1 = ConvBlock(ch1, ch2, self.time_emb_dim)
        self.mid2 = ConvBlock(ch2, ch2, self.time_emb_dim)

        # Upsample: 4x4 -> 8x8
        self.up = nn.ConvTranspose2d(ch2, ch1, kernel_size=4, stride=2, padding=1)

        # Decoder level 1: concatenated skip connection (ch1 + ch1 = 2*ch1)
        self.dec1 = ConvBlock(ch1 * 2, ch1, self.time_emb_dim)

        # Output projection back to in_channels (4x8x8)
        self.out_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, padding=1)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the U-Net."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate_inputs(
        self,
        z: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate input tensor dimensions and format timesteps."""
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected z to be a torch.Tensor, got {type(z).__name__}")
        if not torch.isfinite(z).all():
            raise ValueError("Input z contains non-finite values (NaN or Inf).")

        if z.dim() != 4 or z.shape[1] != self.in_channels or z.shape[2] != 8 or z.shape[3] != 8:
            raise ValueError(
                f"Expected input shape [B, {self.in_channels}, 8, 8], got {list(z.shape)}"
            )

        b = z.size(0)
        if isinstance(t, int):
            t_tensor = torch.full((b,), t, dtype=torch.long, device=z.device)
        elif isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_tensor = t.expand(b).to(device=z.device, dtype=torch.long)
            elif t.dim() == 1:
                if t.size(0) != b:
                    raise ValueError(f"Batch size mismatch: z has batch {b}, t has length {t.size(0)}")
                t_tensor = t.to(device=z.device, dtype=torch.long)
            else:
                raise ValueError(f"Timestep t must be scalar or 1-D tensor, got rank {t.dim()}")
        else:
            raise TypeError(f"Timestep must be int or torch.Tensor, got {type(t).__name__}")

        return z, t_tensor

    def forward(
        self,
        z: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass: [B, 4, 8, 8] + t -> [B, 4, 8, 8]."""
        z_val, t_val = self._validate_inputs(z, t)

        # Timestep embedding
        t_raw = sinusoidal_timestep_embedding(t_val, dim=self.time_emb_dim)
        t_emb = self.time_mlp(t_raw)

        # Encoder
        x1 = self.enc1(z_val, t_emb)  # [B, ch1, 8, 8]
        x_down = self.down(x1)        # [B, ch1, 4, 4]

        # Bottleneck
        m1 = self.mid1(x_down, t_emb) # [B, ch2, 4, 4]
        m2 = self.mid2(m1, t_emb)     # [B, ch2, 4, 4]

        # Decoder with skip connection
        x_up = self.up(m2)            # [B, ch1, 8, 8]
        concat = torch.cat([x_up, x1], dim=1)  # [B, ch1 * 2, 8, 8]
        d1 = self.dec1(concat, t_emb) # [B, ch1, 8, 8]

        # Residual connection
        out = z_val + self.out_conv(d1) # [B, 4, 8, 8]
        return out
