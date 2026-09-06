"""
Image-space evaluation metrics for PHQD Module 4.

Provides:
- validate_image_tensor: Strict validation for shape, finite values, and value range.
- convert_grayscale_to_rgb: Documented channel replication (C=1 -> C=3) for pretrained backbones.
- compute_psnr: Peak Signal-to-Noise Ratio with explicit validation.
- compute_ssim: Structural Similarity Index Measure for 1-channel or multi-channel images.
- LPIPSMetric / compute_lpips: Reference LPIPS (version 0.1, SqueezeNet backbone) with learned linear calibration.
- InceptionFeatureExtractor / compute_frechet_distance / compute_fid: Standard Inception-v3 pool3 FID.

Grayscale Fashion-MNIST Preprocessing & Conventions:
Fashion-MNIST images are single-channel grayscale with shape [B, 1, 28, 28] and intensities in [0, 1].
- PSNR and SSIM operate natively on single-channel grayscale images without modification.
- LPIPS (reference version 0.1 with SqueezeNet trunk):
  1. Grayscale converted to 3-channel RGB via channel replication R = G = B ([B, 3, H, W]).
  2. Input pixel intensities in [0, 1] are converted to [-1, 1] via x_scaled = 2.0 * x - 1.0
     as required by the reference LPIPS implementation.
- FID (Inception-v3 pool3):
  1. Grayscale converted to 3-channel RGB via channel replication R = G = B ([B, 3, H, W]).
  2. Spatially resized to 299x299 using bilinear interpolation (align_corners=False).
  3. Normalized using standard ImageNet statistics.
"""

from typing import Tuple, Union, Optional, List, Dict, Any
import warnings
import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def validate_image_tensor(
    tensor: torch.Tensor,
    name: str = "image",
    expected_dim: int = 4,
    expected_channels: Tuple[int, ...] = (1, 3),
    val_range: Tuple[float, float] = (0.0, 1.0),
) -> None:
    """Validate input image tensor type, dimensionality, finiteness, and value range.

    Args:
        tensor: PyTorch tensor to validate.
        name: Parameter name for informative error messages.
        expected_dim: Expected number of dimensions (default 4: [B, C, H, W]).
        expected_channels: Allowed channel counts (default (1, 3)).
        val_range: Allowed (min_val, max_val) inclusive range.

    Raises:
        TypeError: If tensor is not a torch.Tensor.
        ValueError: If dimensionality, channel count, range, or finiteness check fails.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(tensor).__name__}")

    if tensor.dim() != expected_dim:
        raise ValueError(
            f"Expected {name} to have {expected_dim} dimensions [B, C, H, W], "
            f"got shape {list(tensor.shape)} ({tensor.dim()} dimensions)"
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

    num_channels = tensor.size(1)
    if num_channels not in expected_channels:
        raise ValueError(
            f"Expected {name} channel dimension to be one of {expected_channels}, "
            f"got C={num_channels} in shape {list(tensor.shape)}"
        )

    min_val, max_val = val_range
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()
    eps = 1e-4  # numerical tolerance
    if actual_min < (min_val - eps) or actual_max > (max_val + eps):
        raise ValueError(
            f"{name} values outside expected range [{min_val}, {max_val}]: "
            f"actual min={actual_min:.6f}, max={actual_max:.6f}"
        )


def convert_grayscale_to_rgb(images: torch.Tensor) -> torch.Tensor:
    """Convert 1-channel grayscale images to 3-channel RGB via channel replication.

    Rationale:
    Pretrained visual backbones (e.g. SqueezeNet, Inception-v3) expect 3-channel RGB inputs
    matching ImageNet statistics. For 1-channel grayscale datasets such as Fashion-MNIST
    ([B, 1, H, W]), channel replication R = G = B:
        images_rgb = images.repeat(1, 3, 1, 1)
    preserves luminance gradients, edges, and spatial frequencies identically across all three
    color channels without introducing synthetic chrominance bias.

    Args:
        images: 4D tensor with shape [B, 1, H, W] or [B, 3, H, W].

    Returns:
        4D tensor with shape [B, 3, H, W]. If input is already 3-channel, returns input as-is.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(images).__name__}")
    if images.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {list(images.shape)}")

    c = images.size(1)
    if c == 1:
        return images.repeat(1, 3, 1, 1)
    elif c == 3:
        return images
    else:
        raise ValueError(f"Unsupported channel count C={c}. Expected 1 (grayscale) or 3 (RGB).")


# ==============================================================================
# PSNR Implementation
# ==============================================================================

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.

    Formula:
        MSE = mean((pred - target) ** 2)
        PSNR = 10 * log10(max_val ** 2 / MSE)

    Args:
        pred: Predicted image tensor with shape [B, C, H, W], values in [0, 1].
        target: Target image tensor with shape matching pred.
        max_val: Maximum dynamic range value (default 1.0).
        eps: Small epsilon to prevent log10(0) for near-zero MSE.
        reduction: 'mean' for batch average scalar, or 'none' for per-sample tensor [B].

    Returns:
        Scalar tensor (if reduction='mean') or [B] tensor (if reduction='none') in decibels (dB).
        For identical images, returns 100.0 dB.
    """
    validate_image_tensor(pred, "pred", expected_dim=4, val_range=(0.0, max_val))
    validate_image_tensor(target, "target", expected_dim=4, val_range=(0.0, max_val))

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {list(pred.shape)} != target shape {list(target.shape)}")

    if reduction not in ("mean", "none"):
        raise ValueError(f"Unsupported reduction: {reduction}. Expected 'mean' or 'none'.")

    # Per-sample MSE: average over channels and spatial dimensions [C, H, W]
    sample_mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])

    # For near-zero MSE, clamp to 100.0 dB
    psnr_samples = torch.where(
        sample_mse < eps,
        torch.tensor(100.0, device=pred.device, dtype=pred.dtype),
        10.0 * torch.log10((max_val ** 2) / torch.clamp(sample_mse, min=eps)),
    )

    if reduction == "mean":
        return psnr_samples.mean()
    return psnr_samples


# ==============================================================================
# SSIM Implementation
# ==============================================================================

def _create_gaussian_window_2d(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate normalized 2D Gaussian sliding window for SSIM calculation."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    gauss = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Structural Similarity Index Measure (SSIM) between images.

    Natively supports grayscale [B, 1, H, W] and RGB [B, 3, H, W] images.

    Args:
        pred: Predicted image tensor [B, C, H, W], values in [0, max_val].
        target: Target image tensor [B, C, H, W], values in [0, max_val].
        window_size: Spatial size of Gaussian sliding window (default 11).
        sigma: Standard deviation of Gaussian window (default 1.5).
        max_val: Dynamic range of input images (default 1.0).
        reduction: 'mean' for batch-averaged scalar, 'none' for per-sample tensor [B].

    Returns:
        Scalar tensor (if reduction='mean') or [B] tensor (if reduction='none') in [-1.0, 1.0].
    """
    validate_image_tensor(pred, "pred", expected_dim=4, val_range=(0.0, max_val))
    validate_image_tensor(target, "target", expected_dim=4, val_range=(0.0, max_val))

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {list(pred.shape)} != target shape {list(target.shape)}")

    if reduction not in ("mean", "none"):
        raise ValueError(f"Unsupported reduction: {reduction}. Expected 'mean' or 'none'.")

    channels = pred.size(1)
    window = _create_gaussian_window_2d(
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

    # Average over channels and spatial dimensions per sample
    sample_ssim = ssim_map.mean(dim=[1, 2, 3])

    if reduction == "mean":
        return sample_ssim.mean()
    return sample_ssim


# ==============================================================================
# Reference LPIPS Implementation (lpips.LPIPS, net='squeeze', version='0.1')
# ==============================================================================

class LPIPSMetric(nn.Module):
    """Reference Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Exact Specification & Rationale:
    - Reference Model: Wraps official `lpips.LPIPS(net='squeeze', version='0.1')`.
    - Backbone & Calibration: Uses SqueezeNet 1.1 feature extractor with learned linear
      calibration layers on top of intermediate unit-normalized activations (Zhang et al., 2018).
    - Grayscale Fashion-MNIST Preprocessing: Fashion-MNIST images [B, 1, H, W] are converted
      to 3-channel RGB [B, 3, H, W] via channel replication (convert_grayscale_to_rgb).
    - Input Dynamic Range Scaling: Input tensors with pixel values in [0, 1] are linearly
      scaled to the [-1, 1] convention expected by the reference LPIPS implementation via:
          x_scaled = 2.0 * x - 1.0
    - Model State: Parameters are frozen (requires_grad=False) and held in evaluation mode (eval()).
    """

    def __init__(self, net: str = "squeeze", version: str = "0.1", verbose: bool = False):
        super().__init__()
        import lpips
        self.net = net
        self.version = version
        # Instantiate reference model
        self.model = lpips.LPIPS(net=net, version=version, verbose=verbose)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute reference LPIPS perceptual distance between two image batches.

        Args:
            x1: Tensor [B, C, H, W] with values in [0, 1].
            x2: Tensor [B, C, H, W] with values in [0, 1].

        Returns:
            Per-sample distance tensor of shape [B].
        """
        validate_image_tensor(x1, "x1", expected_dim=4, val_range=(0.0, 1.0))
        validate_image_tensor(x2, "x2", expected_dim=4, val_range=(0.0, 1.0))

        if x1.shape != x2.shape:
            raise ValueError(f"Shape mismatch: x1 shape {list(x1.shape)} != x2 shape {list(x2.shape)}")

        # 1. Grayscale to 3-channel RGB replication
        if x1.size(1) == 1:
            x1 = convert_grayscale_to_rgb(x1)
            x2 = convert_grayscale_to_rgb(x2)

        # 2. Rescale from [0, 1] to [-1, 1] expected by reference LPIPS
        x1_scaled = 2.0 * x1 - 1.0
        x2_scaled = 2.0 * x2 - 1.0

        # 3. Forward pass through reference model (returns [B, 1, 1, 1])
        dist = self.model(x1_scaled, x2_scaled)
        return dist.view(-1)


# Backward-compatible alias for the SqueezeNet LPIPS metric
SqueezeNetLPIPS = LPIPSMetric


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model: Optional[LPIPSMetric] = None,
    device: Optional[torch.device] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute reference LPIPS distance (version 0.1, SqueezeNet backbone).

    Args:
        pred: Predicted image tensor [B, C, H, W] in [0, 1].
        target: Target image tensor [B, C, H, W] in [0, 1].
        lpips_model: Optional instantiated LPIPSMetric model. If None, instantiates one.
        device: Device to run evaluation on.
        reduction: 'mean' for batch-average scalar, 'none' for per-sample tensor [B].

    Returns:
        Scalar tensor (if reduction='mean') or [B] tensor (if reduction='none').
    """
    if device is None:
        device = pred.device

    if lpips_model is None:
        lpips_model = LPIPSMetric(net="squeeze", version="0.1").to(device)
    else:
        lpips_model = lpips_model.to(device)

    pred = pred.to(device)
    target = target.to(device)

    with torch.no_grad():
        dist = lpips_model(pred, target)

    if reduction == "mean":
        return dist.mean()
    elif reduction == "none":
        return dist
    else:
        raise ValueError(f"Unsupported reduction: {reduction}. Expected 'mean' or 'none'.")


# ==============================================================================
# Inception-v3 pool3 FID Implementation
# ==============================================================================

class InceptionFeatureExtractor(nn.Module):
    """Inception-v3 pool3 feature extractor for 2048-dimensional FID representations.

    Features:
    - Pretrained Inception-v3 (torchvision.models.inception_v3) with weights Inception_V3_Weights.DEFAULT.
    - Replaces final classification linear layer (`fc`) with Identity, outputting 2048-dim pool3 activations.
    - Grayscale Handling: Automatically converts [B, 1, H, W] to [B, 3, H, W] via channel replication.
    - Bilinear Resizing: Resizes input images to [299, 299] (Inception native spatial dimension).
    - Normalization: Applies ImageNet mean/std normalization.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        inc = models.inception_v3(weights=weights)
        inc.fc = nn.Identity()  # pool3 output: 2048 features
        self.inception = inc

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dimensional pool3 features from a batch of images.

        Args:
            x: Image batch [B, C, H, W] with values in [0, 1].

        Returns:
            Feature tensor of shape [B, 2048].
        """
        validate_image_tensor(x, "x", expected_dim=4, val_range=(0.0, 1.0))

        # Convert grayscale to RGB if needed
        if x.size(1) == 1:
            x = convert_grayscale_to_rgb(x)

        # Bilinear resize to Inception input size 299x299 if not already 299x299
        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # ImageNet normalization
        x_norm = (x - self.mean) / self.std

        features = self.inception(x_norm)
        return features


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute mathematical Fréchet distance between two multivariate Gaussians.

    Formula:
        d^2 = ||mu1 - mu2||_2^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))

    Numerical Stabilization:
    - Regularizes singular or near-singular covariance matrices with eps * I.
    - Handles small residual imaginary components from scipy.linalg.sqrtm by extracting real part.
    - Clamps negative roundoff distance to 0.0.

    Args:
        mu1: Mean vector for distribution 1 of shape [D].
        sigma1: Covariance matrix for distribution 1 of shape [D, D].
        mu2: Mean vector for distribution 2 of shape [D].
        sigma2: Covariance matrix for distribution 2 of shape [D, D].
        eps: Diagonal regularization constant for matrix square root.

    Returns:
        Scalar Fréchet distance (float).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean dimension mismatch: mu1 shape {mu1.shape} != mu2 shape {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance dimension mismatch: sigma1 shape {sigma1.shape} != sigma2 shape {sigma2.shape}")
    if sigma1.shape[0] != mu1.shape[0]:
        raise ValueError(f"Dimension mismatch between mean ({mu1.shape[0]}) and covariance ({sigma1.shape[0]})")

    diff = mu1 - mu2

    # Product of covariance matrices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=la.LinAlgWarning)
        covmean = la.sqrtm(sigma1.dot(sigma2))

    # Numerical stabilization if square root produces non-finite values
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=la.LinAlgWarning)
            covmean = la.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Discard negligible imaginary component due to floating point roundoff
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean

    return max(0.0, float(fid))


@torch.no_grad()
def extract_features(
    images: torch.Tensor,
    feature_extractor: nn.Module,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract deep feature representations for a batch of images in minibatches.

    Args:
        images: 4D tensor [N, C, H, W] in [0, 1].
        feature_extractor: nn.Module extracting features [B, D].
        batch_size: Sub-batch size for memory efficiency.
        device: Device to run extraction on.

    Returns:
        Numpy array of shape [N, D] containing extracted features.
    """
    if device is None:
        device = next(feature_extractor.parameters()).device

    feature_extractor.eval()
    feature_extractor.to(device)

    num_samples = images.size(0)
    features_list = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = images[start_idx:end_idx].to(device)
        feat = feature_extractor(batch)
        features_list.append(feat.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def compute_fid(
    gen_images: torch.Tensor,
    ref_images: torch.Tensor,
    feature_extractor: Optional[nn.Module] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> float:
    """Compute Fréchet Inception Distance (FID) between generated and reference images.

    Implementation Details:
    - Extracts activations using Inception-v3 pool3 (2048-dim) or a custom pluggable feature extractor.
    - Computes empirical mean and sample covariance with Bessel's correction (N - 1).
    - Evaluates exact Fréchet distance using scipy.linalg.sqrtm with numerical stabilization.
    - Validates inputs: both must be 4D tensors with matching channel and spatial shapes.
    - Minimum sample size: N >= 2 required to compute sample covariance.

    Args:
        gen_images: Generated image batch [N_gen, C, H, W] in [0, 1].
        ref_images: Reference/ground truth image batch [N_ref, C, H, W] in [0, 1].
        feature_extractor: Optional feature extractor. Defaults to InceptionFeatureExtractor.
        batch_size: Minibatch size for feature extraction (default 32).
        device: Device for feature extraction (default: infer from tensor or CUDA if available).

    Returns:
        Scalar FID value (float). Lower is better; 0.0 indicates identical feature distributions.
    """
    validate_image_tensor(gen_images, "gen_images", expected_dim=4, val_range=(0.0, 1.0))
    validate_image_tensor(ref_images, "ref_images", expected_dim=4, val_range=(0.0, 1.0))

    if gen_images.size(1) != ref_images.size(1):
        raise ValueError(
            f"Channel mismatch: gen_images C={gen_images.size(1)} != ref_images C={ref_images.size(1)}"
        )

    if gen_images.size(0) < 2:
        raise ValueError(f"gen_images requires at least 2 samples for covariance, got N={gen_images.size(0)}")
    if ref_images.size(0) < 2:
        raise ValueError(f"ref_images requires at least 2 samples for covariance, got N={ref_images.size(0)}")

    if device is None:
        device = gen_images.device if gen_images.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extractor is None:
        feature_extractor = InceptionFeatureExtractor(pretrained=True).to(device)

    # Extract features
    feat_gen = extract_features(gen_images, feature_extractor, batch_size=batch_size, device=device)
    feat_ref = extract_features(ref_images, feature_extractor, batch_size=batch_size, device=device)

    # Compute empirical mean and sample covariance with Bessel's correction (ddof=1)
    mu_gen = np.mean(feat_gen, axis=0)
    mu_ref = np.mean(feat_ref, axis=0)

    sigma_gen = np.cov(feat_gen, rowvar=False)
    sigma_ref = np.cov(feat_ref, rowvar=False)

    return compute_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
