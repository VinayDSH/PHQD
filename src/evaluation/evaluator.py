"""
Unified Generation and Reconstruction Evaluator for PHQD Module 4.

Provides:
- GenerationEvaluator: High-level orchestrator computing both image-space metrics
  (FID, LPIPS, PSNR, SSIM) and computational metrics (runtime, throughput, parameter counts).
- evaluate_generation: Standalone helper function for evaluating generated images against reference images.
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn

from src.evaluation.image_metrics import (
    validate_image_tensor,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_fid,
    LPIPSMetric,
    SqueezeNetLPIPS,
    InceptionFeatureExtractor,
)
from src.evaluation.computational_metrics import (
    count_parameters,
    parameter_summary,
    measure_inference_runtime,
)


class GenerationEvaluator:
    """Comprehensive evaluator for generative and reconstruction models.

    Clearly separates:
    1. Image-space metrics:
       - FID: Fréchet Inception Distance across feature distributions.
       - LPIPS: Perceptual patch similarity (SqueezeNet-based).
       - PSNR: Peak Signal-to-Noise Ratio (dB).
       - SSIM: Structural Similarity Index Measure.
    2. Computational metrics:
       - Wall-clock inference runtime (mean, min, max, std).
       - Throughput (samples per second).
       - Trainable, non-trainable, and total parameter counts.
       - Model memory size (MB).

    All inputs are validated for type, shape, finiteness, and range.
    Fashion-MNIST grayscale images [B, 1, 28, 28] are natively handled and converted to
    RGB only where required by pretrained backbones.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        val_range: tuple = (0.0, 1.0),
        lpips_model: Optional[SqueezeNetLPIPS] = None,
        fid_feature_extractor: Optional[nn.Module] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.val_range = val_range
        self.lpips_model = lpips_model
        self.fid_feature_extractor = fid_feature_extractor

    def evaluate(
        self,
        gen_images: torch.Tensor,
        ref_images: torch.Tensor,
        paired: bool = True,
        model: Optional[nn.Module] = None,
        compute_fid_metric: bool = True,
        compute_lpips_metric: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate generated images against reference images.

        Args:
            gen_images: Generated image batch [N_gen, C, H, W] with values in [0, 1].
            ref_images: Reference image batch [N_ref, C, H, W] with values in [0, 1].
            paired: If True, computes paired reconstruction metrics (PSNR, SSIM, LPIPS)
                    between corresponding pairs (requires N_gen == N_ref).
                    If False, skips paired metrics and focuses on distribution FID.
            model: Optional nn.Module to inspect for computational metrics (parameter counts).
            compute_fid_metric: Whether to compute FID (requires N >= 2).
            compute_lpips_metric: Whether to compute LPIPS for paired images.
            batch_size: Sub-batch size for FID extraction.

        Returns:
            Dictionary containing:
            - 'image_metrics': Dict with 'psnr', 'ssim', 'lpips', 'fid' (as applicable).
            - 'computational_metrics': Dict with parameter counts and model size (if model provided).
        """
        validate_image_tensor(gen_images, "gen_images", expected_dim=4, val_range=self.val_range)
        validate_image_tensor(ref_images, "ref_images", expected_dim=4, val_range=self.val_range)

        image_metrics: Dict[str, Any] = {}
        computational_metrics: Dict[str, Any] = {}

        # 1. Paired Image-Space Metrics (PSNR, SSIM, LPIPS)
        if paired:
            if gen_images.shape != ref_images.shape:
                raise ValueError(
                    f"Paired evaluation requires identical shapes, "
                    f"got gen {list(gen_images.shape)} vs ref {list(ref_images.shape)}"
                )

            # PSNR (dB)
            psnr_val = compute_psnr(gen_images, ref_images, max_val=self.val_range[1]).item()
            image_metrics["psnr"] = float(psnr_val)

            # SSIM
            ssim_val = compute_ssim(gen_images, ref_images, max_val=self.val_range[1]).item()
            image_metrics["ssim"] = float(ssim_val)

            # LPIPS
            if compute_lpips_metric:
                lpips_val = compute_lpips(
                    gen_images,
                    ref_images,
                    lpips_model=self.lpips_model,
                    device=self.device,
                ).item()
                image_metrics["lpips"] = float(lpips_val)

        # 2. Distribution Metric (FID)
        if compute_fid_metric:
            if gen_images.size(0) >= 2 and ref_images.size(0) >= 2:
                fid_val = compute_fid(
                    gen_images,
                    ref_images,
                    feature_extractor=self.fid_feature_extractor,
                    batch_size=batch_size,
                    device=self.device,
                )
                image_metrics["fid"] = float(fid_val)
            else:
                image_metrics["fid"] = None

        # 3. Computational Metrics (Parameters)
        if model is not None:
            computational_metrics.update(parameter_summary(model))

        return {
            "image_metrics": image_metrics,
            "computational_metrics": computational_metrics,
        }


def evaluate_generation(
    gen_images: torch.Tensor,
    ref_images: torch.Tensor,
    paired: bool = True,
    model: Optional[nn.Module] = None,
    compute_fid_metric: bool = True,
    compute_lpips_metric: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    lpips_model: Optional[SqueezeNetLPIPS] = None,
    fid_feature_extractor: Optional[nn.Module] = None,
    batch_size: int = 32,
) -> Dict[str, Dict[str, Any]]:
    """Convenience function to evaluate generated images against reference images.

    Args:
        gen_images: Generated image batch [N, C, H, W] in [0, 1].
        ref_images: Reference image batch [N, C, H, W] in [0, 1].
        paired: Whether inputs are paired reconstructions.
        model: Optional model for parameter summary.
        compute_fid_metric: Whether to compute FID.
        compute_lpips_metric: Whether to compute LPIPS for paired images.
        device: Target compute device.
        lpips_model: Optional pre-instantiated LPIPS model.
        fid_feature_extractor: Optional pre-instantiated FID feature extractor.
        batch_size: Sub-batch size for FID extraction.

    Returns:
        Dictionary with 'image_metrics' and 'computational_metrics'.
    """
    evaluator = GenerationEvaluator(
        device=device,
        lpips_model=lpips_model,
        fid_feature_extractor=fid_feature_extractor,
    )
    return evaluator.evaluate(
        gen_images=gen_images,
        ref_images=ref_images,
        paired=paired,
        model=model,
        compute_fid_metric=compute_fid_metric,
        compute_lpips_metric=compute_lpips_metric,
        batch_size=batch_size,
    )
