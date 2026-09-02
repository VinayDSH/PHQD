"""
Evaluation metrics and reconstruction utilities for PHQD Module 1.

Provides:
- compute_psnr: Peak Signal-to-Noise Ratio for images in [0, 1].
- compute_ssim: Structural Similarity Index Measure for 28x28 grayscale images.
- evaluate_reconstruction: Comprehensive validation loop reporting MSE, PSNR, and SSIM.
- save_reconstruction_grid: Utility saving original vs. reconstructed image comparison grids.
"""

from typing import Tuple, Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between reconstructed and ground-truth images.

    Args:
        pred: Predicted image tensor with values in [0, 1].
        target: Target ground-truth image tensor with values in [0, 1].
        max_val: Dynamic range maximum value. Defaults to 1.0.
        eps: Small constant to avoid log10(0) when MSE is near zero.

    Returns:
        Scalar PSNR value in decibels (dB). For identical images, returns 100.0 dB.
    """
    mse = F.mse_loss(pred, target)
    if mse.item() < eps:
        # Perfectly identical images or near-zero error handled gracefully as finite high value
        return torch.tensor(100.0, device=pred.device, dtype=torch.float32)

    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr


def create_gaussian_window(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a normalized 2D Gaussian kernel window for SSIM calculation."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    gauss = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Compute Structural Similarity Index Measure (SSIM) for grayscale / single-channel images.

    Appropriate for 28x28 Fashion-MNIST images using standard local Gaussian window.

    Args:
        pred: Predicted image tensor with shape [B, C, H, W] or [C, H, W] or [H, W].
        target: Target ground-truth tensor with matching shape.
        window_size: Spatial size of Gaussian sliding window. Defaults to 11.
        sigma: Standard deviation of Gaussian window. Defaults to 1.5.
        max_val: Dynamic range of input images. Defaults to 1.0.

    Returns:
        Mean SSIM scalar tensor in [-1.0, 1.0].
    """
    # Ensure 4D tensors: [B, C, H, W]
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = pred.size(1)
    window = create_gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        device=pred.device,
        dtype=pred.dtype,
    )
    padding = window_size // 2

    mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate autoencoder reconstruction performance across a dataset.

    Args:
        model: ConvAutoencoder model.
        dataloader: Validation or test DataLoader.
        device: PyTorch compute device.

    Returns:
        Tuple of (mean_mse, mean_psnr, mean_ssim).
    """
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    for images, _ in dataloader:
        images = images.to(device)
        reconstructions = model(images)

        batch_mse = F.mse_loss(reconstructions, images).item()
        batch_psnr = compute_psnr(reconstructions, images).item()
        batch_ssim = compute_ssim(reconstructions, images).item()

        total_mse += batch_mse
        total_psnr += batch_psnr
        total_ssim += batch_ssim
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_mse / num_batches,
        total_psnr / num_batches,
        total_ssim / num_batches,
    )


@torch.no_grad()
def save_reconstruction_grid(
    model: nn.Module,
    dataloader: DataLoader,
    output_path: Union[str, Path],
    num_images: int = 8,
    device: Optional[torch.device] = None,
) -> Path:
    """Save a side-by-side grid of original vs. reconstructed images.

    Top row: Original Fashion-MNIST images.
    Bottom row: Autoencoder reconstructed images.

    Args:
        model: ConvAutoencoder model.
        dataloader: DataLoader to sample images from.
        output_path: Destination image path (e.g., results/figures/reconstructions.png).
        num_images: Number of sample images to display in the grid row. Defaults to 8.
        device: Device to run inference on. If None, inferred from model parameters.

    Returns:
        Path to the saved grid image.
    """
    if device is None:
        device = next(model.parameters()).device

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)
    reconstructions = model(images)

    # Combine originals and reconstructions: row 1 originals, row 2 reconstructions
    grid_tensor = torch.cat([images.cpu(), reconstructions.cpu()], dim=0)
    save_image(grid_tensor, output_path, nrow=num_images, normalize=False, pad_value=1.0)

    return output_path
