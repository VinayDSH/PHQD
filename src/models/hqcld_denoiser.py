"""
Reference HQCLD Denoiser for PHQD Module 4.2.

Architecture:
    Input patches x_t [B, 9, 64] + timestep t [B]
                ↓
    shared classical 64→6 adapter
                ↓
    shared 6-qubit depth-3 PQC (src.models.pqc.PQCDenoiser)
                ↓
    shared classical 6→64 adapter
                ↓
    Predicted clean patches x_0 [B, 9, 64]

Downstream Assembly (Module 3 & Module 1 integration):
    [B, 9, 64] -> reshape [B, 9, 4, 4, 4] -> ClassicalStitchingCNN -> [B, 4, 8, 8]
               -> frozen ConvAutoencoder decoder -> [B, 1, 28, 28]

Parameter Count Configurations:
- 'denoiser_matched': Linear(64, 6, bias=False) [384] + PQC(6, 3) [18] + Linear(6, 64, bias=False) [384]
                      = strictly 786 trainable parameters.
- 'quantum_matched':  FixedPatchProjector (frozen, 0) + PQC(6, 3) [18] + pseudo-inverse unproject (0)
                      = strictly 18 trainable parameters.
- 'custom':           Configurable hidden layer or bias settings.
"""

from typing import Tuple, Dict, Any, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pqc import PQCDenoiser
from src.models.projection import FixedPatchProjector


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Generate sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1-D Tensor of timesteps [B].
        dim: Embedding dimension.
        max_period: Maximum frequency period.
        
    Returns:
        Tensor of shape [B, dim].
    """
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        / half_dim
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class HQCLDDenoiser(nn.Module):
    """Reference Hybrid Quantum-Classical Latent Diffusion (HQCLD) Denoiser.
    
    Processes spatial latent patches [B, 9, 64] at diffusion timestep t.
    The 6-qubit depth-3 PQC denoiser is strictly shared across all 9 spatial patch positions.
    """

    def __init__(
        self,
        patch_dim: int = 64,
        n_qubits: int = 6,
        pqc_depth: int = 3,
        num_patches: int = 9,
        adapter_mode: str = "denoiser_matched",
        bias: bool = False,
        pqc_dev: Optional[Any] = None,
        projector: Optional[FixedPatchProjector] = None,
    ):
        """Initialize Reference HQCLD Denoiser.
        
        Args:
            patch_dim: Dimension of flattened patch (default: 64).
            n_qubits: Number of qubits in PQC (default: 6).
            pqc_depth: Variational layers in PQC (default: 3 -> 18 parameters).
            num_patches: Number of spatial patch locations (default: 9).
            adapter_mode: One of 'denoiser_matched' (linear adapters, ~786 params) or
                          'quantum_matched' (fixed projection adapters, 18 params).
            bias: Whether classical linear adapters include bias (default: False).
            pqc_dev: Optional PennyLane quantum device.
            projector: Optional pre-fitted FixedPatchProjector for 'quantum_matched' mode.
        """
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.n_qubits = int(n_qubits)
        self.pqc_depth = int(pqc_depth)
        self.num_patches = int(num_patches)
        self.adapter_mode = adapter_mode.lower()
        self.bias = bias

        # Shared 6-qubit depth-3 PQC denoiser (strictly 6 * pqc_depth parameters)
        self.pqc = PQCDenoiser(
            n_qubits=self.n_qubits,
            n_layers=self.pqc_depth,
            dev=pqc_dev,
        )

        if self.adapter_mode == "denoiser_matched":
            # Shared classical 64 -> 6 adapter
            self.encoder = nn.Linear(self.patch_dim, self.n_qubits, bias=bias)
            # Shared classical 6 -> 64 adapter
            self.decoder = nn.Linear(self.n_qubits, self.patch_dim, bias=bias)
            self.projector = None
        elif self.adapter_mode == "quantum_matched":
            # Uses parameter-free FixedPatchProjector with pseudo-inverse back-projection
            if projector is not None:
                self.projector = projector
            else:
                self.projector = FixedPatchProjector(
                    in_dim=self.patch_dim,
                    out_dim=self.n_qubits,
                    norm_type="standard",
                    mean_64=torch.zeros(self.patch_dim),
                    std_6=torch.ones(self.n_qubits),
                )
            self.encoder = None
            self.decoder = None
        else:
            raise ValueError(f"Unsupported adapter_mode: '{adapter_mode}'. Use 'denoiser_matched' or 'quantum_matched'.")

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_pqc_parameters(self) -> int:
        """Total trainable parameters in the PQC module strictly."""
        return self.pqc.num_parameters

    @property
    def num_adapter_parameters(self) -> int:
        """Total trainable parameters in classical adapters."""
        return self.num_parameters - self.num_pqc_parameters

    def _validate_inputs(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate input tensor dimensions and format timesteps."""
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
        """Forward denoiser pass: [B, 9, 64] + t -> [B, 9, 64].
        
        Args:
            x: Input noisy patches of shape [B, 9, 64].
            t: Diffusion timestep(s), integer or tensor [B].
            
        Returns:
            Predicted clean patches of shape [B, 9, 64].
        """
        x_val, t_val = self._validate_inputs(x, t)
        b = x_val.size(0)

        # 1. Timestep embedding [B, 64]
        t_emb = sinusoidal_timestep_embedding(t_val, dim=self.patch_dim)

        # 2. Time conditioning injected additively into patch representations
        x_cond = x_val + t_emb.unsqueeze(1)  # [B, 9, 64]

        # 3. Reshape to [B * 9, 64] for shared evaluation across all 9 spatial patches
        x_flat = x_cond.reshape(b * self.num_patches, self.patch_dim)

        # 4. Shared 64 -> 6 adapter
        if self.adapter_mode == "denoiser_matched":
            z_6 = self.encoder(x_flat)
        else:
            z_6 = self.projector(x_flat)

        # 5. Shared 6-qubit PQC denoiser
        z_denoised = self.pqc(z_6)  # Output in [-1, 1]

        # 6. Shared 6 -> 64 adapter
        if self.adapter_mode == "denoiser_matched":
            x_pred_flat = self.decoder(z_denoised)
        else:
            x_pred_flat = self.projector.pseudo_inverse_back_project(z_denoised)

        # 7. Reshape back to [B, 9, 64] preserving distinct patch locations
        x_pred = x_pred_flat.reshape(b, self.num_patches, self.patch_dim)
        return x_pred
