"""
PHQD Evaluation Package.

Exports:
- Image-space metrics: compute_psnr, compute_ssim, compute_lpips, compute_fid, SqueezeNetLPIPS, InceptionFeatureExtractor
- Validation utilities: validate_image_tensor, convert_grayscale_to_rgb, compute_frechet_distance
- Computational metrics: count_parameters, parameter_summary, measure_inference_runtime
- High-level orchestrators: GenerationEvaluator, evaluate_generation
- Backward-compatible Module 1 utilities: evaluate_reconstruction, save_reconstruction_grid
"""

from src.evaluation.metrics import (
    evaluate_reconstruction,
    save_reconstruction_grid,
)
from src.evaluation.image_metrics import (
    validate_image_tensor,
    convert_grayscale_to_rgb,
    compute_psnr,
    compute_ssim,
    LPIPSMetric,
    SqueezeNetLPIPS,
    compute_lpips,
    InceptionFeatureExtractor,
    compute_frechet_distance,
    compute_fid,
)
from src.evaluation.computational_metrics import (
    count_parameters,
    parameter_summary,
    measure_inference_runtime,
)
from src.evaluation.evaluator import (
    GenerationEvaluator,
    evaluate_generation,
)

__all__ = [
    # Module 1 backwards compatibility
    "evaluate_reconstruction",
    "save_reconstruction_grid",
    # Validation & preprocessing
    "validate_image_tensor",
    "convert_grayscale_to_rgb",
    "compute_frechet_distance",
    # Image metrics
    "compute_psnr",
    "compute_ssim",
    "LPIPSMetric",
    "SqueezeNetLPIPS",
    "compute_lpips",
    "InceptionFeatureExtractor",
    "compute_fid",
    # Computational metrics
    "count_parameters",
    "parameter_summary",
    "measure_inference_runtime",
    # Evaluator
    "GenerationEvaluator",
    "evaluate_generation",
]
