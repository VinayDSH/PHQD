"""
Reusable Latent-Patch Extraction Module for PHQD Module 1.

Features:
- Configurable sliding-window extraction of spatial patches from 4x8x8 latent grids.
- Preservation of the channel dimension: outputs [B, N_patches, C, P_h, P_w].
- Loading and freezing of trained ConvAutoencoder checkpoints in eval() mode.
- Batch and DataLoader extraction routines without gradient computation.
"""

from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.autoencoder import ConvAutoencoder


def extract_latent_patches(
    z: torch.Tensor,
    patch_size: int = 4,
    stride: int = 2,
) -> torch.Tensor:
    """Extract overlapping 2D spatial patches from a latent grid tensor.

    Args:
        z: Latent tensor of shape [B, C, H, W] or [C, H, W].
        patch_size: Spatial height and width of each extracted patch. Defaults to 4.
        stride: Stride step size along spatial dimensions. Defaults to 2.

    Returns:
        Tensor of extracted patches with shape [B, N_patches, C, patch_size, patch_size].
        For input [B, 4, 8, 8] with patch_size=4 and stride=2:
        Outputs [B, 9, 4, 4, 4].
    """
    is_3d = z.dim() == 3
    if is_3d:
        z = z.unsqueeze(0)  # [1, C, H, W]

    b, c, h, w = z.shape

    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Latent spatial dimensions ({h}, {w}) must be >= patch_size ({patch_size})"
        )

    # Compute number of patches along height and width
    n_h = (h - patch_size) // stride + 1
    n_w = (w - patch_size) // stride + 1

    # Use 2D sliding unfold: unfolds dim 2 (H) then dim 3 (W)
    # Shape after unfolds: [B, C, n_h, n_w, patch_size, patch_size]
    patches = z.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    # Permute to [B, n_h, n_w, C, patch_size, patch_size] and flatten patch grid indices
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(b, n_h * n_w, c, patch_size, patch_size)

    if is_3d:
        patches = patches.squeeze(0)

    return patches


def load_frozen_autoencoder(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[ConvAutoencoder, Dict[str, Any]]:
    """Load trained ConvAutoencoder checkpoint, set to eval(), and freeze all parameters.

    Args:
        checkpoint_path: Path to best_autoencoder.pt checkpoint file.
        device: Device to load model onto. Defaults to CUDA if available else CPU.

    Returns:
        Tuple of (frozen_model, checkpoint_dict).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found at: {checkpoint_path}. "
            f"Train the autoencoder first via 'python src/training/train_autoencoder.py'."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine architecture hyperparameters from checkpoint metadata or defaults
    config = checkpoint.get("config", {})
    in_channels = config.get("dataset", {}).get("channels", 1)
    latent_channels = config.get("autoencoder", {}).get("latent_channels", 4)

    model = ConvAutoencoder(in_channels=in_channels, latent_channels=latent_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze model and switch to evaluation mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint


class LatentPatchExtractor:
    """End-to-end patch extraction engine operating on a frozen autoencoder."""

    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 4,
        stride: int = 2,
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()

        # Enforce parameter freezing
        for param in self.model.parameters():
            param.requires_grad = False

        self.patch_size = patch_size
        self.stride = stride

    @torch.no_grad()
    def extract_from_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space and extract patches without gradients.

        Args:
            images: Tensor of shape [B, 1, 28, 28].

        Returns:
            Tensor of shape [B, 9, 4, 4, 4].
        """
        images = images.to(self.device)
        latents = self.model.encode(images)
        return extract_latent_patches(latents, patch_size=self.patch_size, stride=self.stride)

    @torch.no_grad()
    def extract_from_loader(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract all latent patches and labels from a DataLoader.

        Args:
            dataloader: PyTorch DataLoader yielding (images, labels).
            max_samples: Optional cap on the number of samples extracted (useful for testing).

        Returns:
            Tuple of (all_patches, all_labels) where:
            - all_patches has shape [N, 9, 4, 4, 4]
            - all_labels has shape [N]
        """
        patches_list = []
        labels_list = []
        collected = 0

        for images, labels in dataloader:
            if max_samples is not None and collected >= max_samples:
                break

            if max_samples is not None and collected + images.size(0) > max_samples:
                remaining = max_samples - collected
                images = images[:remaining]
                labels = labels[:remaining]

            batch_patches = self.extract_from_images(images)
            patches_list.append(batch_patches.cpu())
            labels_list.append(labels.cpu())

            collected += images.size(0)

        all_patches = torch.cat(patches_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        return all_patches, all_labels
