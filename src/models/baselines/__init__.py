"""
Module 4.2 Classical Baseline Models.

Provides:
- MLPDenoiser: Multi-layer perceptron patch denoiser.
- FourierMLPDenoiser: Random Fourier Feature patch denoiser.
- DequantizedSurrogateDenoiser: Bounded harmonic surrogate patch denoiser.
- UNetLatentDenoiser: Full classical 2D convolutional U-Net for continuous latents.
"""

from src.models.baselines.mlp_denoiser import MLPDenoiser
from src.models.baselines.fourier_mlp_denoiser import FourierMLPDenoiser
from src.models.baselines.dequantized_surrogate import (
    DequantizedSurrogateDenoiser,
    DequantizedSurrogateCore,
)
from src.models.baselines.unet_denoiser import UNetLatentDenoiser

__all__ = [
    "MLPDenoiser",
    "FourierMLPDenoiser",
    "DequantizedSurrogateDenoiser",
    "DequantizedSurrogateCore",
    "UNetLatentDenoiser",
]
