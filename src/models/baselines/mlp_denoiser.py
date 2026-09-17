"""
MLP Patch Denoiser Baseline for PHQD Module 4.2.

Interface:
    Input patches x_t [B, 9, 64] + timestep t [B]
                ↓
    MLP Patch Denoiser (shared across all 9 patch positions)
                ↓
    Predicted clean patches x_0 [B, 9, 64]

Parameter Matching Tiers:
- 'denoiser_matched': Linear(64, 6, bias=False) [384] + Linear(6, 64, bias=False) [384]
                      = 768 trainable parameters (matching HQCLD adapters).
                      With optional hidden bottleneck Linear(6, 6) [36] or activation,
                      strictly targets the ~786 parameter budget.
- 'quantum_matched':  Parameter-matched circulant or minimal bottleneck targeting strictly
                      ~18 trainable parameters (matching the 6-qubit depth-3 PQC circuit).
- 'standard':         Multi-layer MLP with configurable hidden dimension (e.g. 64 or 128).
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hqcld_denoiser import sinusoidal_timestep_embedding
from src.models.pqc import CirculantLinear


class MLPDenoiser(nn.Module):
    """Classical MLP Denoiser operating directly on spatial latent patches [B, 9, 64]."""

    def __init__(
        self,
        patch_dim: int = 64,
        num_patches: int = 9,
        matching_tier: str = "denoiser_matched",
        hidden_dim: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize MLP Denoiser.
        
        Args:
            patch_dim: Dimension of flattened patch (default: 64).
            num_patches: Number of spatial patch locations (default: 9).
            matching_tier: One of 'denoiser_matched' (~786 params),
                           'quantum_matched' (~18 params), or 'standard'.
            hidden_dim: Hidden dimension for 'standard' tier.
            bias: Whether linear layers include bias (default: False).
            dropout: Dropout probability.
        """
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.num_patches = int(num_patches)
        self.matching_tier = matching_tier.lower()

        if self.matching_tier == "quantum_matched":
            # Target: approximately 18 trainable parameters matching 6-qubit depth-3 PQC
            # Uses 3 CirculantLinear layers of dimension 6 (3 * 6 = 18 params) operating
            # on a fixed 64->6 QR subspace, or a fixed compression.
            # Here: Linear(64, 3, bias=False) is 192 (too many).
            # Instead: 18-parameter circulant block with fixed linear down/up projection:
            gen = torch.Generator().manual_seed(42)
            proj_down = torch.randn(6, self.patch_dim, generator=gen)
            q, _ = torch.linalg.qr(proj_down.t())
            self.register_buffer("fixed_down", q.t().contiguous(), persistent=True)  # [6, 64]
            # 3 circulant layers with dim=6 -> exactly 3 * 6 = 18 trainable parameters
            self.circ1 = CirculantLinear(dim=6)
            self.circ2 = CirculantLinear(dim=6)
            self.circ3 = CirculantLinear(dim=6)
            self.net = None
        elif self.matching_tier == "denoiser_matched":
            # Target: approximately 786 parameters matching the full HQCLD denoiser
            # Linear(64, 6, bias=False) = 384 params
            # Linear(6, 6, bias=False)  = 36 params
            # Linear(6, 64, bias=False) = 384 params
            # Total = 384 + 36 + 384 = 804 params (or 64->6->64 = 768 params without middle layer)
            # Exactly 786: 64->6 (384) + Circulant(6)*3 (18) + 6->64 (384) = 786 parameters!
            self.encoder = nn.Linear(self.patch_dim, 6, bias=bias)
            self.core = nn.Sequential(
                CirculantLinear(dim=6),
                nn.ReLU(),
                CirculantLinear(dim=6),
                nn.ReLU(),
                CirculantLinear(dim=6),
            )
            self.decoder = nn.Linear(6, self.patch_dim, bias=bias)
        elif self.matching_tier == "standard":
            h_dim = hidden_dim or 64
            self.net = nn.Sequential(
                nn.Linear(self.patch_dim, h_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(h_dim, h_dim, bias=bias),
                nn.GELU(),
                nn.Linear(h_dim, self.patch_dim, bias=bias),
            )
        else:
            raise ValueError(
                f"Unsupported matching_tier: '{matching_tier}'. "
                "Must be 'quantum_matched', 'denoiser_matched', or 'standard'."
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
        """Validate input shapes and format timesteps."""
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

    def forward(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass: [B, 9, 64] + t -> [B, 9, 64]."""
        x_val, t_val = self._validate_inputs(x, t)
        b = x_val.size(0)

        # Sinusoidal timestep embedding injected additively into patches
        t_emb = sinusoidal_timestep_embedding(t_val, dim=self.patch_dim)
        x_cond = x_val + t_emb.unsqueeze(1)

        # Reshape to [B * 9, 64] for shared evaluation
        x_flat = x_cond.reshape(b * self.num_patches, self.patch_dim)

        if self.matching_tier == "quantum_matched":
            # Fixed projection down to 6D -> 18 trainable params -> fixed unprojection
            z = F.linear(x_flat, self.fixed_down)
            z = F.relu(self.circ1(z))
            z = F.relu(self.circ2(z))
            z = self.circ3(z)
            out_flat = F.linear(z, self.fixed_down.t())
        elif self.matching_tier == "denoiser_matched":
            z = self.encoder(x_flat)
            z = self.core(z)
            out_flat = self.decoder(z)
        else:
            out_flat = self.net(x_flat)

        return out_flat.reshape(b, self.num_patches, self.patch_dim)
