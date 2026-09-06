"""
Module 3 Milestone 4: Seam-Consistency Loss.

Computes the spatial boundary mismatch across overlapping regions of adjacent patches
in a 3x3 patch grid.

Patch Arrangement (stride = 2, patch_size = 4):
    P0  P1  P2
    P3  P4  P5
    P6  P7  P8

Adjacent patches overlap by 2 spatial positions.

Horizontal Seam Comparisons (6 adjacent pairs):
    P0 right overlap (W=2:4) vs P1 left overlap (W=0:2)
    P1 right overlap (W=2:4) vs P2 left overlap (W=0:2)
    P3 right overlap (W=2:4) vs P4 left overlap (W=0:2)
    P4 right overlap (W=2:4) vs P5 left overlap (W=0:2)
    P6 right overlap (W=2:4) vs P7 left overlap (W=0:2)
    P7 right overlap (W=2:4) vs P8 left overlap (W=0:2)

Vertical Seam Comparisons (6 adjacent pairs):
    P0 bottom overlap (H=2:4) vs P3 top overlap (H=0:2)
    P3 bottom overlap (H=2:4) vs P6 top overlap (H=0:2)
    P1 bottom overlap (H=2:4) vs P4 top overlap (H=0:2)
    P4 bottom overlap (H=2:4) vs P7 top overlap (H=0:2)
    P2 bottom overlap (H=2:4) vs P5 top overlap (H=0:2)
    P5 bottom overlap (H=2:4) vs P8 top overlap (H=0:2)

Loss Formula:
    L_horizontal = (1/6) * sum(MSE(P_left[..., 2:4], P_right[..., 0:2]))
    L_vertical   = (1/6) * sum(MSE(P_top[..., 2:4, :], P_bottom[..., 0:2, :]))
    L_seam       = (L_horizontal + L_vertical) / 2
"""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# Exact adjacent neighbor pairs in 3x3 grid
HORIZONTAL_PAIRS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
]

VERTICAL_PAIRS: List[Tuple[int, int]] = [
    (0, 3),
    (3, 6),
    (1, 4),
    (4, 7),
    (2, 5),
    (5, 8),
]


class SeamConsistencyLoss(nn.Module):
    """Computes mean-squared error across overlapping boundaries of adjacent spatial patches.

    This loss contains strictly zero trainable parameters. It operates on patch tensors
    of shape [B, 9, 4, 4, 4] or unbatched [9, 4, 4, 4].
    """

    def __init__(
        self,
        num_patches: int = 9,
        channels: int = 4,
        patch_size: int = 4,
        overlap: int = 2,
    ):
        super().__init__()
        self.num_patches = int(num_patches)
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)

        if self.num_patches != 9:
            raise ValueError(f"Expected 9 patches for 3x3 grid, got {num_patches}")
        if self.patch_size != 4:
            raise ValueError(f"Expected patch size 4, got {patch_size}")
        if self.overlap != 2:
            raise ValueError(f"Expected overlap 2, got {overlap}")

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the loss operator (strictly 0)."""
        return 0

    def _validate_inputs(self, patches: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Validate input tensor rank, dimensions, types, and finiteness."""
        if not isinstance(patches, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(patches).__name__}")

        if not torch.isfinite(patches).all():
            raise ValueError("Input patches contain non-finite values (NaN or Inf)")

        is_unbatched = False
        if patches.dim() == 4:
            # Unbatched [9, C, H, W] -> [1, 9, C, H, W]
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
            raise ValueError(
                f"Expected patch spatial dimensions ({self.patch_size}, {self.patch_size}), got ({ph}, {pw})"
            )

        return patches, is_unbatched

    def compute_horizontal_seam_loss(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss across all 6 horizontal adjacent patch boundaries."""
        patches_5d, _ = self._validate_inputs(patches)
        h_diffs = []
        for left_idx, right_idx in HORIZONTAL_PAIRS:
            left_overlap = patches_5d[:, left_idx, :, :, self.patch_size - self.overlap : self.patch_size]
            right_overlap = patches_5d[:, right_idx, :, :, 0 : self.overlap]
            h_diffs.append(F.mse_loss(left_overlap, right_overlap))

        return torch.stack(h_diffs).mean()

    def compute_vertical_seam_loss(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss across all 6 vertical adjacent patch boundaries."""
        patches_5d, _ = self._validate_inputs(patches)
        v_diffs = []
        for top_idx, bottom_idx in VERTICAL_PAIRS:
            top_overlap = patches_5d[:, top_idx, :, self.patch_size - self.overlap : self.patch_size, :]
            bottom_overlap = patches_5d[:, bottom_idx, :, 0 : self.overlap, :]
            v_diffs.append(F.mse_loss(top_overlap, bottom_overlap))

        return torch.stack(v_diffs).mean()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute total seam consistency loss: L_seam = (L_horizontal + L_vertical) / 2."""
        l_h = self.compute_horizontal_seam_loss(patches)
        l_v = self.compute_vertical_seam_loss(patches)
        return (l_h + l_v) / 2.0

    def forward_with_components(self, patches: torch.Tensor) -> Dict[str, Any]:
        """Compute total seam loss and expose all individual horizontal and vertical components."""
        patches_5d, _ = self._validate_inputs(patches)

        h_losses = []
        h_details = []
        for left_idx, right_idx in HORIZONTAL_PAIRS:
            left_overlap = patches_5d[:, left_idx, :, :, self.patch_size - self.overlap : self.patch_size]
            right_overlap = patches_5d[:, right_idx, :, :, 0 : self.overlap]
            pair_loss = F.mse_loss(left_overlap, right_overlap)
            h_losses.append(pair_loss)
            h_details.append({
                "pair": (left_idx, right_idx),
                "loss": pair_loss.item(),
            })

        v_losses = []
        v_details = []
        for top_idx, bottom_idx in VERTICAL_PAIRS:
            top_overlap = patches_5d[:, top_idx, :, self.patch_size - self.overlap : self.patch_size, :]
            bottom_overlap = patches_5d[:, bottom_idx, :, 0 : self.overlap, :]
            pair_loss = F.mse_loss(top_overlap, bottom_overlap)
            v_losses.append(pair_loss)
            v_details.append({
                "pair": (top_idx, bottom_idx),
                "loss": pair_loss.item(),
            })

        l_h = torch.stack(h_losses).mean()
        l_v = torch.stack(v_losses).mean()
        l_total = (l_h + l_v) / 2.0

        return {
            "total": l_total,
            "horizontal": l_h,
            "vertical": l_v,
            "h_pair_losses": h_details,
            "v_pair_losses": v_details,
        }
