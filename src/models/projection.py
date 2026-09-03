"""
Fixed 64-D to 6-D Projection and Normalization Module for PHQD Module 2 Stage 2.

Provides:
- FixedPatchProjector: Deterministic, parameter-free PyTorch module mapping 64-D latent
  patches to 6-D quantum circuit inputs, with training-data-only normalization statistics.

Methodological Context:
- Intentionally Lossy Dimension Reduction: The 64-D -> 6-D projection maps a 64-dimensional
  latent patch space to a 6-dimensional subspace (compression ratio 10.67:1). This reduction
  is intentionally lossy and discards the 58-dimensional orthogonal complement (null space of P).
  Arbitrary 64-D patches cannot be exactly reconstructed.
- Pseudo-Inverse Back-Projection: Mapping 6-D vectors back to 64-D via the pseudo-inverse P.T
  reconstructs the orthogonal projection onto the fixed 6-D projection subspace:
      P^T P (x - mean_64) + mean_64
  Vectors residing strictly within the 6-D subspace are reconstructed exactly, while arbitrary
  64-D vectors incur expected, non-zero reconstruction error.
- Metric Priority for Stage 2 Experiment: In Stage 2 denoising benchmarks, 6-D projected-space
  denoising MSE serves as the primary evaluation metric. 64-D pseudo-inverse reconstruction MSE
  serves as a secondary/diagnostic metric.
"""

from typing import Tuple, Dict, Any, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPatchProjector(nn.Module):
    """Deterministic, parameter-free 64-D to 6-D projection and normalization module.

    Transforms 64-dimensional latent patches (e.g., 4x4x4 spatial patches from Module 1)
    into 6-dimensional vectors suitable for angle encoding on a 6-qubit PQC in Module 2 Stage 2.

    Key Properties:
    1. Deterministic: Projection matrix generated via QR decomposition with a dedicated,
       independent seed (does not alter global PyTorch RNG).
    2. Orthonormal Rows: P @ P.T = I_6, preserving Euclidean distances and inner products
       within the projection subspace without collapsing coordinate axes.
    3. Zero Trainable Parameters: Projection matrix and normalization statistics are registered
       as non-trainable persistent buffers (requires_grad=False).
    4. Training-Data-Only Statistics: Normalization (mean, std, min, max) is computed strictly
       on training data and frozen, preventing any evaluation data leakage.
    5. Pseudo-Inverse Back-Projection: Maps 6-D vectors back to 64-D via linear pseudo-inverse P.T.
       Arbitrary 64-D patches cannot be exactly reconstructed due to intentional lossy compression,
       but vectors within the 6-D projection subspace are recovered faithfully.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 6,
        seed: int = 42,
        norm_type: str = "standard",
        scale_factor: float = 1.0,
        eps: float = 1e-8,
        mean_64: Optional[torch.Tensor] = None,
        mean_6: Optional[torch.Tensor] = None,
        std_6: Optional[torch.Tensor] = None,
        min_6: Optional[torch.Tensor] = None,
        max_6: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seed = seed
        self.norm_type = norm_type.lower()
        self.scale_factor = scale_factor
        self.eps = eps

        valid_norm_types = {"standard", "minmax", "angle", "none"}
        if self.norm_type not in valid_norm_types:
            raise ValueError(
                f"Unsupported norm_type: '{norm_type}'. Must be one of {valid_norm_types}"
            )

        # Generate deterministic semi-orthogonal projection matrix via QR decomposition
        proj_matrix = self._generate_orthogonal_matrix(in_dim, out_dim, seed)
        self.register_buffer("projection_matrix", proj_matrix, persistent=True)

        # Register non-trainable buffers for training-data statistics
        init_mean_64 = mean_64 if mean_64 is not None else torch.zeros(in_dim)
        init_mean_6 = mean_6 if mean_6 is not None else torch.zeros(out_dim)
        init_std_6 = std_6 if std_6 is not None else torch.ones(out_dim)
        init_min_6 = min_6 if min_6 is not None else -torch.ones(out_dim)
        init_max_6 = max_6 if max_6 is not None else torch.ones(out_dim)

        self.register_buffer("mean_64", init_mean_64.clone().detach(), persistent=True)
        self.register_buffer("mean_6", init_mean_6.clone().detach(), persistent=True)
        self.register_buffer("std_6", init_std_6.clone().detach(), persistent=True)
        self.register_buffer("min_6", init_min_6.clone().detach(), persistent=True)
        self.register_buffer("max_6", init_max_6.clone().detach(), persistent=True)

        is_pre_fitted = (mean_64 is not None and std_6 is not None) or (self.norm_type == "none")
        self.register_buffer(
            "is_fitted",
            torch.tensor(is_pre_fitted, dtype=torch.bool),
            persistent=True,
        )

    @staticmethod
    def _generate_orthogonal_matrix(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
        """Construct deterministic semi-orthogonal projection matrix P of shape [out_dim, in_dim]."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        gaussian_matrix = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
        q, r = torch.linalg.qr(gaussian_matrix)
        # Canonicalize signs to ensure cross-platform reproducibility
        diag_signs = torch.sign(torch.diagonal(r))
        diag_signs[diag_signs == 0.0] = 1.0
        q = q * diag_signs.unsqueeze(0)
        # P has orthonormal rows: P @ P.T = I_out_dim
        p = q.t().contiguous()
        return p

    @property
    def num_parameters(self) -> int:
        """Return total trainable parameters (strictly 0)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_trainable_parameters(self) -> int:
        """Return total trainable parameters (strictly 0)."""
        return self.num_parameters

    def fit(self, data: Union[torch.Tensor, Any]) -> "FixedPatchProjector":
        """Compute normalization statistics strictly from training data and freeze them in buffers."""
        if not isinstance(data, torch.Tensor):
            batches = [b[0] if isinstance(b, (list, tuple)) else b for b in data]
            data = torch.cat(batches, dim=0)

        # Flatten any trailing spatial/channel dimensions to [N, in_dim]
        data_flat = data.reshape(-1, self.in_dim)
        data_flat = data_flat.to(dtype=torch.float32, device=self.projection_matrix.device)

        # 1. 64-D statistics
        mean_64 = data_flat.mean(dim=0)

        # 2. Centered raw projection to 6-D
        centered = data_flat - mean_64
        projected = F.linear(centered, self.projection_matrix)  # [N, 6]

        # 3. 6-D statistics
        mean_6 = projected.mean(dim=0)
        std_6 = projected.std(dim=0, unbiased=False)
        std_6 = torch.clamp(std_6, min=self.eps)
        min_6 = projected.amin(dim=0)
        max_6 = projected.amax(dim=0)

        # Update persistent buffers without adding trainable parameters
        self.mean_64.copy_(mean_64)
        self.mean_6.copy_(mean_6)
        self.std_6.copy_(std_6)
        self.min_6.copy_(min_6)
        self.max_6.copy_(max_6)
        self.is_fitted.fill_(True)

        return self

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit on training data and return projected and normalized output."""
        self.fit(data)
        return self.forward(data)

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """Flatten leading dimensions and validate dimension."""
        trailing_numel = 1
        for s in x.shape[1:]:
            trailing_numel *= s

        if x.dim() == 1:
            if x.numel() != self.in_dim:
                raise ValueError(f"Expected 1D input of size {self.in_dim}, got {x.numel()}")
            return x.unsqueeze(0), ()
        elif x.shape[-1] == self.in_dim:
            batch_shape = x.shape[:-1]
            return x.reshape(-1, self.in_dim), batch_shape
        elif trailing_numel == self.in_dim:
            batch_shape = (x.shape[0],)
            return x.reshape(-1, self.in_dim), batch_shape
        else:
            # Multi-dim patch grid: e.g. [B, 9, 4, 4, 4]
            total_elements = x.numel()
            if total_elements % self.in_dim != 0:
                raise ValueError(
                    f"Input total elements {total_elements} not divisible by in_dim {self.in_dim}"
                )
            lead_shape = x.shape[: -3] if x.shape[-3:] == (4, 4, 4) else (x.shape[0], -1)
            return x.reshape(-1, self.in_dim), lead_shape

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear semi-orthogonal projection: (x - mean_64) @ P.T."""
        x_flat, batch_shape = self._prepare_input(x)
        x_centered = x_flat - self.mean_64
        z_proj = F.linear(x_centered, self.projection_matrix)
        if not batch_shape:
            return z_proj.squeeze(0)
        return z_proj.reshape(*batch_shape, self.out_dim)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Apply linear pseudo-inverse back-projection: z @ P + mean_64."""
        batch_shape = z.shape[:-1] if z.dim() > 1 else ()
        z_flat = z.reshape(-1, self.out_dim)
        x_centered = F.linear(z_flat, self.projection_matrix.t())
        x_recon = x_centered + self.mean_64
        if not batch_shape:
            return x_recon.squeeze(0)
        return x_recon.reshape(*batch_shape, self.in_dim)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize projected 6-D vector using frozen training statistics."""
        if self.norm_type == "none":
            return z

        if not bool(self.is_fitted):
            raise RuntimeError(
                "FixedPatchProjector must be fitted on training data before normalization. "
                "Call projector.fit(train_data) first."
            )

        if self.norm_type == "standard":
            z_norm = (z - self.mean_6) / self.std_6
            return z_norm * self.scale_factor
        elif self.norm_type == "minmax":
            range_6 = torch.clamp(self.max_6 - self.min_6, min=self.eps)
            z_norm = 2.0 * (z - self.min_6) / range_6 - 1.0
            return z_norm * self.scale_factor
        elif self.norm_type == "angle":
            range_6 = torch.clamp(self.max_6 - self.min_6, min=self.eps)
            z_norm = (2.0 * (z - self.min_6) / range_6 - 1.0) * math.pi
            return z_norm * self.scale_factor
        return z

    def denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Invert normalization using frozen training statistics."""
        if self.norm_type == "none":
            return z_norm

        if not bool(self.is_fitted):
            raise RuntimeError(
                "FixedPatchProjector must be fitted on training data before denormalization."
            )

        if self.norm_type == "standard":
            z_scaled = z_norm / self.scale_factor
            return z_scaled * self.std_6 + self.mean_6
        elif self.norm_type == "minmax":
            z_scaled = z_norm / self.scale_factor
            range_6 = torch.clamp(self.max_6 - self.min_6, min=self.eps)
            return (z_scaled + 1.0) * 0.5 * range_6 + self.min_6
        elif self.norm_type == "angle":
            z_scaled = (z_norm / self.scale_factor) / math.pi
            range_6 = torch.clamp(self.max_6 - self.min_6, min=self.eps)
            return (z_scaled + 1.0) * 0.5 * range_6 + self.min_6
        return z_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project 64-D inputs to 6-D and normalize using frozen training statistics."""
        z_proj = self.project(x)
        return self.normalize(z_proj)

    def pseudo_inverse_back_project(
        self,
        z: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Denormalize 6-D vector and back-project onto the fixed 6-D projection subspace in 64-D.

        Note:
            This operation is the pseudo-inverse of the lossy 64-D -> 6-D projection. Arbitrary
            64-D patches cannot be exactly reconstructed, but vectors within the 6-D projection
            subspace are reconstructed faithfully.
        """
        z_raw = self.denormalize(z)
        x_recon = self.reconstruct(z_raw)
        if target_shape is not None:
            batch_shape = x_recon.shape[:-1]
            return x_recon.reshape(*batch_shape, *target_shape)
        return x_recon

    def unproject(
        self,
        z: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Alias for pseudo_inverse_back_project."""
        return self.pseudo_inverse_back_project(z, target_shape=target_shape)

    def inverse(
        self,
        z: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Alias for pseudo_inverse_back_project."""
        return self.pseudo_inverse_back_project(z, target_shape=target_shape)
