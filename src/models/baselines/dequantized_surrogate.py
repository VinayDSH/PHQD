"""
Dequantized Surrogate Patch Denoiser Baseline for PHQD Module 4.2.

================================================================================
IMPLEMENTATION CONTEXT & SCIENTIFIC INTEGRITY NOTICE:
This module provides a conservative classical dequantized surrogate intended to
reproduce the functional structure and parameter efficiency of the 6-qubit,
depth-3 Parameterized Quantum Circuit (PQC) without executing quantum statevector
simulation.

Mathematical & Architectural Design:
1. Classical & Deterministic: Pure PyTorch implementation without quantum gates or simulator.
2. Low-Dimensional Bottleneck: Processes 6-dimensional vectors, mirroring the 6-qubit
   angle-encoding state space.
3. Bounded Output: Output is constrained to [-1.0, 1.0] using Tanh activation, strictly
   matching the Pauli-Z expectation measurement range of the quantum circuit.
4. Harmonic/Fourier Series Approximation: Single-qubit Pauli-Z expectations from an
   L-layer circuit evaluate functions within a bounded multi-frequency trigonometric
   subspace: f(x) = sum_k c_k * cos(k * x + phi_k) with harmonic order bounded by L.
   Here, the order K is configurable (default K=3 matching depth L=3).
5. Exact Parameter Matching:
   - 'quantum_matched': Core surrogate contains strictly 18 trainable parameters
     (matching 6 qubits x depth 3), with frozen QR projection/unprojection adapters.
   - 'denoiser_matched': Core 18-parameter surrogate sandwiched between classical
     Linear(64, 6, bias=False) [384] and Linear(6, 64, bias=False) [384], achieving
     strictly 384 + 18 + 384 = 786 trainable parameters.
6. Important Disclaimer: This model is explicitly designated as a classical surrogate
   baseline and does NOT claim to reproduce an exact analytical PQC spectrum.
================================================================================
"""

from typing import Tuple, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hqcld_denoiser import sinusoidal_timestep_embedding
from src.models.pqc import CirculantLinear
from src.models.projection import FixedPatchProjector


class DequantizedSurrogateCore(nn.Module):
    """Classical surrogate core replacing the 6-qubit depth-3 PQC.
    
    Contains strictly dim * order trainable parameters (default: 6 * 3 = 18 parameters),
    evaluating successive harmonic transformations bounded to [-1, 1] via Tanh.
    """

    def __init__(self, dim: int = 6, order: int = 3):
        super().__init__()
        self.dim = int(dim)
        self.order = int(order)
        # Successive circulant linear layers providing dim parameters each
        self.layers = nn.ModuleList([CirculantLinear(self.dim) for _ in range(self.order)])

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the surrogate core (strictly dim * order)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through harmonic circulant layers with final Tanh bounding to [-1, 1]."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.order - 1:
                # Sine activation introduces trigonometric harmonic non-linearity
                h = torch.sin(h)
        # Final Tanh bounds outputs strictly to [-1.0, 1.0], matching Pauli-Z expectation range
        return torch.tanh(h)


class DequantizedSurrogateDenoiser(nn.Module):
    """Patch-based Denoiser utilizing the classical Dequantized Surrogate core."""

    def __init__(
        self,
        patch_dim: int = 64,
        n_qubits: int = 6,
        order: int = 3,
        num_patches: int = 9,
        matching_tier: str = "denoiser_matched",
        bias: bool = False,
        projector: Optional[FixedPatchProjector] = None,
    ):
        """Initialize Dequantized Surrogate Denoiser.
        
        Args:
            patch_dim: Dimension of flattened patch (default: 64).
            n_qubits: Bottleneck dimension matching quantum qubit count (default: 6).
            order: Harmonic expansion order matching PQC depth (default: 3).
            num_patches: Number of spatial patch locations (default: 9).
            matching_tier: One of 'denoiser_matched' (~786 params),
                           'quantum_matched' (~18 params), or 'standard'.
            bias: Whether classical adapter layers include bias.
            projector: Optional pre-fitted FixedPatchProjector for 'quantum_matched' mode.
        """
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.n_qubits = int(n_qubits)
        self.order = int(order)
        self.num_patches = int(num_patches)
        self.matching_tier = matching_tier.lower()

        # Surrogate core: strictly n_qubits * order parameters (6 * 3 = 18 params)
        self.surrogate_core = DequantizedSurrogateCore(dim=self.n_qubits, order=self.order)

        if self.matching_tier == "denoiser_matched":
            # Adapter 64 -> 6 (384 params) + Core (18 params) + Adapter 6 -> 64 (384 params) = 786 params
            self.encoder = nn.Linear(self.patch_dim, self.n_qubits, bias=bias)
            self.decoder = nn.Linear(self.n_qubits, self.patch_dim, bias=bias)
            self.projector = None
        elif self.matching_tier == "quantum_matched":
            # Zero-parameter projection and pseudo-inverse adapters -> strictly 18 params
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
            self.encoder = nn.Linear(self.patch_dim, self.n_qubits, bias=bias)
            self.decoder = nn.Linear(self.n_qubits, self.patch_dim, bias=bias)
            self.projector = None

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_surrogate_parameters(self) -> int:
        """Trainable parameters in the dequantized surrogate core strictly (18)."""
        return self.surrogate_core.num_parameters

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

    def forward(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass: [B, 9, 64] + t -> [B, 9, 64]."""
        x_val, t_val = self._validate_inputs(x, t)
        b = x_val.size(0)

        # Sinusoidal timestep embedding injected additively into patch representations
        t_emb = sinusoidal_timestep_embedding(t_val, dim=self.patch_dim)
        x_cond = x_val + t_emb.unsqueeze(1)

        # Reshape to [B * 9, 64] for shared evaluation
        x_flat = x_cond.reshape(b * self.num_patches, self.patch_dim)

        # Classical adapter 64 -> 6
        if self.matching_tier == "quantum_matched":
            z_6 = self.projector(x_flat)
        else:
            z_6 = self.encoder(x_flat)

        # Dequantized surrogate core: outputs strictly in [-1, 1] via Tanh
        z_surrogate = self.surrogate_core(z_6)

        # Classical adapter 6 -> 64
        if self.matching_tier == "quantum_matched":
            out_flat = self.projector.pseudo_inverse_back_project(z_surrogate)
        else:
            out_flat = self.decoder(z_surrogate)

        return out_flat.reshape(b, self.num_patches, self.patch_dim)
