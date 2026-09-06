"""
Module 3 Milestone 3: Classical Stitching CNN.

Implements:
1. OverlapAverageStitcher:
   - Deterministic 3x3 overlap-averaging operator mapping [B, 9, 4, 4, 4] -> [B, 4, 8, 8].
   - Has strictly zero trainable parameters.
   - Precomputes and registers exact spatial overlap weight counts:
     - 4 corners (2x2 each, 16 cells): count = 1
     - 4 edge overlaps (2x4 each, 32 cells): count = 2
     - 1 central overlap (4x4, 16 cells): count = 4
     - Total count sum across grid = 144 = 9 patches x 4 x 4 elements.
   - Invertibility identity: For clean patches extracted from a clean latent grid z,
     stitch(extract(z)) is bit-identically equal to z (max abs error = 0.0).

2. ResidualRefinementCNN:
   - 4 -> 32 -> 32 -> 4 residual convolutional network refining the overlap-averaged grid:
     Conv2d(4, 32, 3, padding=1) -> ReLU -> Conv2d(32, 32, 3, padding=1) -> ReLU -> Conv2d(32, 4, 3, padding=1).
   - Residual skip connection: output = x + net(x).
   - Contains exactly 11,588 trainable parameters.

3. ClassicalStitchingCNN:
   - Composes OverlapAverageStitcher and ResidualRefinementCNN:
     [B, 9, 4, 4, 4] -> deterministic overlap stitch -> [B, 4, 8, 8] -> residual CNN -> [B, 4, 8, 8].
   - Total model trainable parameters: strictly 11,588.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


def compute_overlap_weight_counts(
    grid_size: int = 8,
    patch_size: int = 4,
    stride: int = 2,
) -> torch.Tensor:
    """Compute the deterministic 2D coverage count matrix for sliding patches.

    For grid_size=8, patch_size=4, stride=2:
    - 3 patches along height, 3 patches along width (total 9 patches).
    - Returns a [1, 1, 8, 8] float32 tensor of coverage counts with values in {1.0, 2.0, 4.0}.
    """
    counts = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    n_h = (grid_size - patch_size) // stride + 1
    n_w = (grid_size - patch_size) // stride + 1

    for r in range(n_h):
        for c in range(n_w):
            h_start = r * stride
            w_start = c * stride
            counts[h_start : h_start + patch_size, w_start : w_start + patch_size] += 1.0

    return counts.view(1, 1, grid_size, grid_size)


class OverlapAverageStitcher(nn.Module):
    """Deterministic 3x3 overlap-average stitcher mapping [B, 9, 4, 4, 4] to [B, 4, 8, 8].

    This layer contains strictly zero trainable parameters. It accumulates the 9 spatial
    patches at their corresponding stride-2 grid offsets and divides by the precomputed
    coverage count tensor.
    """

    def __init__(
        self,
        grid_size: int = 8,
        patch_size: int = 4,
        stride: int = 2,
        channels: int = 4,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.channels = int(channels)

        self.n_h = (self.grid_size - self.patch_size) // self.stride + 1
        self.n_w = (self.grid_size - self.patch_size) // self.stride + 1
        self.num_patches = self.n_h * self.n_w

        if self.num_patches != 9:
            raise ValueError(
                f"Expected 9 patches for grid {grid_size}x{grid_size}, patch {patch_size}x{patch_size}, "
                f"stride {stride}, but got {self.num_patches}"
            )

        # Precompute and register frozen persistent weight counts buffer
        weight_counts = compute_overlap_weight_counts(
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        self.register_buffer("weight_counts", weight_counts, persistent=True)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters (strictly 0)."""
        return 0

    def _validate_inputs(self, patches: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Validate input tensor rank, dimensions, types, and finiteness."""
        if not isinstance(patches, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(patches).__name__}")

        if not torch.isfinite(patches).all():
            raise ValueError("Input patches contain non-finite values (NaN or Inf)")

        is_unbatched = False
        if patches.dim() == 4:
            # Unbatched [9, C, P, P]
            patches = patches.unsqueeze(0)
            is_unbatched = True
        elif patches.dim() != 5:
            raise ValueError(
                f"Expected 5D tensor [B, N, C, H, W] or 4D tensor [N, C, H, W], got rank {patches.dim()}"
            )

        b, n, c, ph, pw = patches.shape
        if n != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {n}")
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        if ph != self.patch_size or pw != self.patch_size:
            raise ValueError(f"Expected patch spatial dims ({self.patch_size}, {self.patch_size}), got ({ph}, {pw})")

        return patches, is_unbatched

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Stitch 9 spatial patches into a single continuous latent grid.

        Args:
            patches: Tensor of shape [B, 9, 4, 4, 4] or [9, 4, 4, 4].

        Returns:
            Stitched continuous latent grid of shape [B, 4, 8, 8] or [4, 8, 8].
        """
        patches_5d, is_unbatched = self._validate_inputs(patches)
        b, _, c, _, _ = patches_5d.shape

        accum = torch.zeros(
            (b, c, self.grid_size, self.grid_size),
            dtype=patches_5d.dtype,
            device=patches_5d.device,
        )

        # Accumulate patches into grid positions
        for p in range(self.num_patches):
            r = p // self.n_w
            col = p % self.n_w
            h_start = r * self.stride
            w_start = col * self.stride
            accum[:, :, h_start : h_start + self.patch_size, w_start : w_start + self.patch_size] += patches_5d[:, p]

        # Normalize by precomputed overlap counts
        stitched = accum / self.weight_counts

        if is_unbatched:
            stitched = stitched.squeeze(0)

        return stitched


class ResidualRefinementCNN(nn.Module):
    """Classical 4 -> 32 -> 32 -> 4 residual convolutional refinement network.

    Architecture:
        Conv2d(4, 32, kernel_size=3, padding=1)
        ReLU()
        Conv2d(32, 32, kernel_size=3, padding=1)
        ReLU()
        Conv2d(32, 4, kernel_size=3, padding=1)

    Residual connection:
        output = x + net(x)

    Trainable parameters: strictly 11,588.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        out_channels: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=kernel_size, padding=padding),
        )

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the refinement network (strictly 11,588)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Validate input tensor rank, channels, spatial size, and finiteness."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")

        if not torch.isfinite(x).all():
            raise ValueError("Input grid contains non-finite values (NaN or Inf)")

        is_unbatched = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            is_unbatched = True
        elif x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W] or 3D tensor [C, H, W], got rank {x.dim()}")

        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

        return x, is_unbatched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine latent grid using residual convolutional path.

        Args:
            x: Stitched latent grid of shape [B, 4, 8, 8] or [4, 8, 8].

        Returns:
            Refined latent grid of shape [B, 4, 8, 8] or [4, 8, 8].
        """
        x_4d, is_unbatched = self._validate_inputs(x)
        residual = self.net(x_4d)
        refined = x_4d + residual

        if is_unbatched:
            refined = refined.squeeze(0)

        return refined


class ClassicalStitchingCNN(nn.Module):
    """End-to-end Classical Stitching CNN pipeline.

    Combines:
    1. Deterministic OverlapAverageStitcher: [B, 9, 4, 4, 4] -> [B, 4, 8, 8] (0 params)
    2. ResidualRefinementCNN: [B, 4, 8, 8] -> [B, 4, 8, 8] (11,588 params)

    Total trainable parameters: strictly 11,588.
    """

    def __init__(
        self,
        grid_size: int = 8,
        patch_size: int = 4,
        stride: int = 2,
        channels: int = 4,
        hidden_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.stitcher = OverlapAverageStitcher(
            grid_size=grid_size,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
        )
        self.refinement_cnn = ResidualRefinementCNN(
            in_channels=channels,
            hidden_channels=hidden_channels,
            out_channels=channels,
            kernel_size=kernel_size,
        )

    @property
    def num_trainable_parameters(self) -> int:
        """Total trainable parameters in the entire model (strictly 11,588)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_stitcher_parameters(self) -> int:
        """Total trainable parameters in the stitcher operator (strictly 0)."""
        return sum(p.numel() for p in self.stitcher.parameters() if p.requires_grad)

    def stitch(self, patches: torch.Tensor) -> torch.Tensor:
        """Perform deterministic overlap-average stitching only without CNN refinement."""
        return self.stitcher(patches)

    def refine(self, z_overlap: torch.Tensor) -> torch.Tensor:
        """Perform residual CNN refinement on a pre-stitched continuous latent grid."""
        return self.refinement_cnn(z_overlap)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Full pipeline: deterministic overlap stitching followed by residual CNN refinement.

        Args:
            patches: Tensor of shape [B, 9, 4, 4, 4] or [9, 4, 4, 4].

        Returns:
            Refined continuous latent grid of shape [B, 4, 8, 8] or [4, 8, 8].
        """
        z_overlap = self.stitcher(patches)
        return self.refinement_cnn(z_overlap)

    def forward_with_intermediates(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute full forward pass returning intermediate overlap-averaged and refined tensors.

        Args:
            patches: Tensor of shape [B, 9, 4, 4, 4] or [9, 4, 4, 4].

        Returns:
            Dict containing:
            - "overlap_average": Stitched grid [B, 4, 8, 8] before CNN refinement
            - "refined": Refined grid [B, 4, 8, 8] after residual CNN
        """
        z_overlap = self.stitcher(patches)
        z_refined = self.refinement_cnn(z_overlap)
        return {
            "overlap_average": z_overlap,
            "refined": z_refined,
        }
