"""
Fourier-MLP Patch Denoiser Baseline for PHQD Module 4.2.

================================================================================
IMPLEMENTATION ASSUMPTION NOTICE (§8.1 Specification Context):
Because the exact mathematical specification of §8.1 from the research blueprint
is not publicly formalized in the repository codebase, this implementation uses a
scientifically rigorous, standard Random Fourier Feature (RFF) projection:
    gamma(x) = [cos(2 * pi * x @ W_fourier.T), sin(2 * pi * x @ W_fourier.T)]
where:
    - W_fourier is a fixed, non-trainable Gaussian random projection matrix
      sampled from N(0, sigma^2) with a dedicated deterministic seed.
    - Number of frequency pairs (num_frequencies) and frequency scale (sigma)
      are user-configurable.
    - Timestep conditioning is injected via sinusoidal time embeddings.
    - A trainable MLP maps the resulting Fourier representation back to 64-D.
This implementation is explicitly designated as a defensible classical baseline
and does NOT claim to reproduce an unreleased proprietary §8.1 blueprint formula.

Exact Parameter Matching Tiers:
- 'denoiser_matched': num_frequencies=32 -> 64-D RFF features.
                      Linear(64, 6, bias=False) [384] + CirculantLinear(6)*3 [18] + Linear(6, 64, bias=False) [384]
                      = strictly 786 trainable parameters (matching HQCLD denoiser).
- 'quantum_matched':  num_frequencies=3 -> 6-D RFF features.
                      CirculantLinear(6)*3 [18] + fixed orthogonal expansion (0)
                      = strictly 18 trainable parameters (matching 6-qubit depth-3 PQC).
- 'standard':         Configurable hidden dimension and frequency count.
================================================================================
"""

from typing import Tuple, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hqcld_denoiser import sinusoidal_timestep_embedding
from src.models.pqc import CirculantLinear


class FourierMLPDenoiser(nn.Module):
    """Classical Fourier-MLP Denoiser using fixed Random Fourier Features (RFF)."""

    def __init__(
        self,
        patch_dim: int = 64,
        num_patches: int = 9,
        num_frequencies: Optional[int] = None,
        sigma: float = 1.0,
        seed: int = 42,
        matching_tier: str = "denoiser_matched",
        hidden_dim: Optional[int] = None,
        bias: bool = False,
    ):
        """Initialize Fourier-MLP Denoiser.
        
        Args:
            patch_dim: Input patch dimension (default: 64).
            num_patches: Number of spatial patch locations (default: 9).
            num_frequencies: Number of random Fourier frequency pairs. If None,
                             defaults to 32 for 'denoiser_matched' (to achieve 786 params)
                             or 3 for 'quantum_matched' (to achieve 18 params).
            sigma: Standard deviation of Gaussian frequency projection weights.
            seed: Deterministic random seed for frozen projection matrix.
            matching_tier: One of 'denoiser_matched' (~786 params),
                           'quantum_matched' (~18 params), or 'standard'.
            hidden_dim: Hidden dimension for 'standard' tier.
            bias: Whether trainable linear layers include bias.
        """
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.num_patches = int(num_patches)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.matching_tier = matching_tier.lower()

        if num_frequencies is None:
            if self.matching_tier == "denoiser_matched":
                self.num_frequencies = 32  # 2 * 32 = 64 RFF features
            elif self.matching_tier == "quantum_matched":
                self.num_frequencies = 3   # 2 * 3 = 6 RFF features
            else:
                self.num_frequencies = 16
        else:
            self.num_frequencies = int(num_frequencies)

        # Fixed, non-trainable Random Fourier Feature projection matrix: [num_frequencies, patch_dim]
        gen = torch.Generator().manual_seed(self.seed)
        w_fourier = torch.randn(self.num_frequencies, self.patch_dim, generator=gen) * self.sigma
        self.register_buffer("w_fourier", w_fourier, persistent=True)

        fourier_dim = 2 * self.num_frequencies  # cos and sin channels

        if self.matching_tier == "quantum_matched":
            # Strictly 18 trainable parameters: 3 circulant layers of dim 6
            circ_dim = 6
            self.circ1 = CirculantLinear(dim=circ_dim)
            self.circ2 = CirculantLinear(dim=circ_dim)
            self.circ3 = CirculantLinear(dim=circ_dim)
            # Fixed orthogonal expansion to 64-D (0 trainable params)
            gen_exp = torch.Generator().manual_seed(self.seed + 100)
            proj_up = torch.randn(self.patch_dim, circ_dim, generator=gen_exp)
            q, _ = torch.linalg.qr(proj_up)
            self.register_buffer("fixed_up", q.contiguous(), persistent=True)  # [64, circ_dim]
            self.net = None
        elif self.matching_tier == "denoiser_matched":
            # Strictly 786 parameters:
            # Linear(64, 6, bias=False) [384] + Circulant(6)*3 [18] + Linear(6, 64, bias=False) [384] = 786
            self.fc1 = nn.Linear(fourier_dim, 6, bias=bias)     # 64 * 6 = 384 (when fourier_dim=64)
            self.core = nn.Sequential(
                CirculantLinear(dim=6),                         # 6
                nn.ReLU(),
                CirculantLinear(dim=6),                         # 6
                nn.ReLU(),
                CirculantLinear(dim=6),                         # 6
            )                                                   # 18
            self.fc2 = nn.Linear(6, self.patch_dim, bias=bias)  # 6 * 64 = 384
            # Total = 384 + 18 + 384 = 786 params!
        else:
            h_dim = hidden_dim or 64
            self.net = nn.Sequential(
                nn.Linear(fourier_dim, h_dim, bias=bias),
                nn.GELU(),
                nn.Linear(h_dim, h_dim, bias=bias),
                nn.GELU(),
                nn.Linear(h_dim, self.patch_dim, bias=bias),
            )

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate input tensor shapes and format timesteps."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}")
        if not torch.isfinite(x).all():
            raise ValueError("Input x contains non-finite values (NaN or Inf).")

        if x.dim() != 3 or x.shape[1] != self.num_patches or x.shape[2] != self.patch_dim:
            raise ValueError(
                f"Expected input shape [B, {self.num_patches}, {self.patch_dim}], got {list(x.shape)}"
            )

        b = x.size(0)
        if isinstance(t, int):
            t_tensor = torch.full((b,), t, dtype=torch.long, device=x.device)
        elif isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_tensor = t.expand(b).to(device=x.device, dtype=torch.long)
            elif t.dim() == 1:
                if t.size(0) != b:
                    raise ValueError(f"Batch size mismatch: x has batch {b}, t has length {t.size(0)}")
                t_tensor = t.to(device=x.device, dtype=torch.long)
            else:
                raise ValueError(f"Timestep t must be scalar or 1-D tensor, got rank {t.dim()}")
        else:
            raise TypeError(f"Timestep must be int or torch.Tensor, got {type(t).__name__}")

        return x, t_tensor

    def compute_fourier_features(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Compute non-trainable Random Fourier Features: [N, 2 * num_frequencies]."""
        proj = 2.0 * math.pi * F.linear(x_flat, self.w_fourier)  # [N, num_frequencies]
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)
        return torch.cat([cos_feat, sin_feat], dim=-1)  # [N, 2 * num_frequencies]

    def forward(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass: [B, 9, 64] + t -> [B, 9, 64]."""
        x_val, t_val = self._validate_inputs(x, t)
        b = x_val.size(0)

        # Sinusoidal timestep embedding injected additively into patch space
        t_emb = sinusoidal_timestep_embedding(t_val, dim=self.patch_dim)
        x_cond = x_val + t_emb.unsqueeze(1)

        x_flat = x_cond.reshape(b * self.num_patches, self.patch_dim)
        fourier_feat = self.compute_fourier_features(x_flat)  # [B*9, 2*num_frequencies]

        if self.matching_tier == "quantum_matched":
            z = F.relu(self.circ1(fourier_feat))
            z = F.relu(self.circ2(z))
            z = self.circ3(z)
            out_flat = F.linear(z, self.fixed_up)
        elif self.matching_tier == "denoiser_matched":
            z = self.fc1(fourier_feat)
            z = self.core(z)
            out_flat = self.fc2(z)
        else:
            out_flat = self.net(fourier_feat)

        return out_flat.reshape(b, self.num_patches, self.patch_dim)
