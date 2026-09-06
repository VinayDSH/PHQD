"""
Module 3 Milestone 5: Combined Loss for Latent Patch Diffusion.

Combines three objectives operating directly in spatial latent-patch space [B, 9, 4, 4, 4]:
1. L_diffusion: Mean-squared error between predicted clean patches and clean target patches:
       L_diffusion = MSE(pred_patches, target_patches)
   (Note: For this toy milestone, L_diffusion represents the patch denoising MSE objective).
2. L_seam: Spatial seam-consistency loss across overlapping boundaries of adjacent patches:
       L_seam = SeamConsistencyLoss(pred_patches)
3. L_recon: Full latent grid reconstruction MSE between overlap-stitched patches and target latent:
       L_recon = MSE(stitcher(pred_patches), target_latent)

Combined Total Loss:
    L_total = L_diffusion + lambda_seam * L_seam + lambda_recon * L_recon
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.seam_consistency import SeamConsistencyLoss
from src.models.stitching_cnn import OverlapAverageStitcher


class CombinedLoss(nn.Module):
    """Multi-objective combined loss for spatial patch diffusion.

    Combines patch-level denoising MSE, cross-patch boundary seam-consistency loss,
    and stitched continuous latent grid reconstruction MSE.

    Contains strictly zero trainable parameters.
    """

    def __init__(
        self,
        lambda_seam: float = 0.0,
        lambda_recon: float = 1.0,
        seam_loss_fn: Optional[SeamConsistencyLoss] = None,
        stitcher: Optional[OverlapAverageStitcher] = None,
    ):
        super().__init__()
        self.lambda_seam = float(lambda_seam)
        self.lambda_recon = float(lambda_recon)

        if self.lambda_seam < 0.0:
            raise ValueError(f"lambda_seam must be non-negative, got {lambda_seam}")
        if self.lambda_recon < 0.0:
            raise ValueError(f"lambda_recon must be non-negative, got {lambda_recon}")

        self.seam_loss_fn = seam_loss_fn if seam_loss_fn is not None else SeamConsistencyLoss()
        self.stitcher = stitcher if stitcher is not None else OverlapAverageStitcher()

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the loss operator (strictly 0)."""
        return 0

    def _validate_inputs(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Validate input tensors, shapes, ranks, and finite values."""
        for name, tensor in [
            ("pred_patches", pred_patches),
            ("target_patches", target_patches),
            ("target_latent", target_latent),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for {name}, got {type(tensor).__name__}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

        is_unbatched = False
        if pred_patches.dim() == 4 and target_patches.dim() == 4 and target_latent.dim() == 3:
            pred_patches = pred_patches.unsqueeze(0)
            target_patches = target_patches.unsqueeze(0)
            target_latent = target_latent.unsqueeze(0)
            is_unbatched = True

        if pred_patches.dim() != 5 or target_patches.dim() != 5:
            raise ValueError("Expected 5D tensors [B, 9, 4, 4, 4] for pred_patches and target_patches")
        if target_latent.dim() != 4:
            raise ValueError("Expected 4D tensor [B, 4, 8, 8] for target_latent")

        if pred_patches.shape != target_patches.shape:
            raise ValueError(
                f"Shape mismatch: pred_patches {list(pred_patches.shape)} vs target_patches {list(target_patches.shape)}"
            )

        if pred_patches.shape[0] != target_latent.shape[0]:
            raise ValueError(
                f"Batch size mismatch: patches {pred_patches.shape[0]} vs latent {target_latent.shape[0]}"
            )

        if pred_patches.shape[1:] != (9, 4, 4, 4):
            raise ValueError(f"Expected patches shape [B, 9, 4, 4, 4], got {list(pred_patches.shape)}")
        if target_latent.shape[1:] != (4, 8, 8):
            raise ValueError(f"Expected latent shape [B, 4, 8, 8], got {list(target_latent.shape)}")

        return pred_patches, target_patches, target_latent, is_unbatched

    def forward(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar total loss: L_total = L_diff + lambda_seam * L_seam + lambda_recon * L_recon."""
        pred_p, target_p, target_z, _ = self._validate_inputs(pred_patches, target_patches, target_latent)

        # 1. Patch Denoising / Diffusion MSE
        l_diff = F.mse_loss(pred_p, target_p)

        # 2. Seam-Consistency Loss
        if self.lambda_seam > 0.0:
            l_seam = self.seam_loss_fn(pred_p)
        else:
            l_seam = torch.tensor(0.0, device=pred_p.device, dtype=pred_p.dtype)

        # 3. Overlap-Stitched Reconstruction MSE
        if self.lambda_recon > 0.0:
            z_rec = self.stitcher(pred_p)
            l_recon = F.mse_loss(z_rec, target_z)
        else:
            l_recon = torch.tensor(0.0, device=pred_p.device, dtype=pred_p.dtype)

        l_total = l_diff + (self.lambda_seam * l_seam) + (self.lambda_recon * l_recon)
        return l_total

    def forward_with_components(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute total loss and expose all individual loss components."""
        pred_p, target_p, target_z, _ = self._validate_inputs(pred_patches, target_patches, target_latent)

        l_diff = F.mse_loss(pred_p, target_p)

        seam_components = self.seam_loss_fn.forward_with_components(pred_p)
        l_seam = seam_components["total"]
        l_h = seam_components["horizontal"]
        l_v = seam_components["vertical"]

        z_rec = self.stitcher(pred_p)
        l_recon = F.mse_loss(z_rec, target_z)

        l_total = l_diff + (self.lambda_seam * l_seam) + (self.lambda_recon * l_recon)

        return {
            "total": l_total,
            "diffusion": l_diff,
            "seam": l_seam,
            "horizontal_seam": l_h,
            "vertical_seam": l_v,
            "recon": l_recon,
            "lambda_seam": self.lambda_seam,
            "lambda_recon": self.lambda_recon,
        }
