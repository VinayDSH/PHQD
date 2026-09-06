"""
Unit tests for PHQD Module 4 Milestone 1: Evaluation Tooling.

Verifies:
1. Strict input validation (types, shapes, finiteness, ranges).
2. Documented grayscale-to-RGB conversion for Fashion-MNIST.
3. PSNR accuracy, identity condition, and CPU/CUDA consistency.
4. SSIM accuracy, identity condition, and CPU/CUDA consistency.
5. FID analytical mathematical correctness (Fréchet distance) and Inception-v3 pipeline.
6. SqueezeNet LPIPS perceptual distance, identity condition, monotonicity, and grayscale handling.
7. Computational metrics: trainable/frozen parameter counts, runtime latency, and throughput.
8. GenerationEvaluator end-to-end orchestration separating image and computational metrics.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.image_metrics import (
    validate_image_tensor,
    convert_grayscale_to_rgb,
    compute_psnr,
    compute_ssim,
    compute_frechet_distance,
    compute_fid,
    LPIPSMetric,
    SqueezeNetLPIPS,
    compute_lpips,
    InceptionFeatureExtractor,
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


# ==============================================================================
# 1. Validation Tests
# ==============================================================================

def test_validate_image_tensor_success():
    """Valid 4D tensors within [0, 1] should pass without error."""
    valid_gray = torch.rand(4, 1, 28, 28)
    validate_image_tensor(valid_gray, "valid_gray")

    valid_rgb = torch.rand(2, 3, 32, 32)
    validate_image_tensor(valid_rgb, "valid_rgb")


def test_validate_image_tensor_type_error():
    """Non-tensor inputs must raise TypeError."""
    with pytest.raises(TypeError, match="Expected image to be a torch.Tensor"):
        validate_image_tensor(np.zeros((4, 1, 28, 28)), "image")


def test_validate_image_tensor_dimension_error():
    """Tensors that are not 4D [B, C, H, W] must raise ValueError."""
    with pytest.raises(ValueError, match="Expected image to have 4 dimensions"):
        validate_image_tensor(torch.rand(28, 28), "image")

    with pytest.raises(ValueError, match="Expected image to have 4 dimensions"):
        validate_image_tensor(torch.rand(1, 28, 28), "image")


def test_validate_image_tensor_non_finite():
    """Tensors containing NaN or Inf must raise ValueError."""
    tensor_nan = torch.rand(2, 1, 28, 28)
    tensor_nan[0, 0, 5, 5] = float("nan")
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_image_tensor(tensor_nan, "tensor_nan")

    tensor_inf = torch.rand(2, 1, 28, 28)
    tensor_inf[0, 0, 5, 5] = float("inf")
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_image_tensor(tensor_inf, "tensor_inf")


def test_validate_image_tensor_channel_error():
    """Tensors with unsupported channel counts must raise ValueError."""
    tensor_4ch = torch.rand(2, 4, 28, 28)
    with pytest.raises(ValueError, match="channel dimension to be one of"):
        validate_image_tensor(tensor_4ch, "tensor_4ch")


def test_validate_image_tensor_range_error():
    """Tensors with values outside [0, 1] must raise ValueError."""
    tensor_neg = torch.tensor([[-0.1]])
    tensor_neg_4d = torch.zeros(1, 1, 4, 4)
    tensor_neg_4d[0, 0, 0, 0] = -0.05
    with pytest.raises(ValueError, match="values outside expected range"):
        validate_image_tensor(tensor_neg_4d, "tensor_neg_4d")

    tensor_high_4d = torch.ones(1, 1, 4, 4)
    tensor_high_4d[0, 0, 0, 0] = 1.2
    with pytest.raises(ValueError, match="values outside expected range"):
        validate_image_tensor(tensor_high_4d, "tensor_high_4d")


# ==============================================================================
# 2. Grayscale Fashion-MNIST Preprocessing Tests
# ==============================================================================

def test_convert_grayscale_to_rgb_1channel():
    """1-channel grayscale [B, 1, H, W] must replicate across 3 channels [B, 3, H, W]."""
    gray = torch.rand(4, 1, 28, 28)
    rgb = convert_grayscale_to_rgb(gray)

    assert rgb.shape == (4, 3, 28, 28)
    # Each channel must be identical to the original grayscale channel
    assert torch.equal(rgb[:, 0:1, :, :], gray)
    assert torch.equal(rgb[:, 1:2, :, :], gray)
    assert torch.equal(rgb[:, 2:3, :, :], gray)


def test_convert_grayscale_to_rgb_3channel():
    """3-channel images must pass through unchanged."""
    rgb_in = torch.rand(2, 3, 28, 28)
    rgb_out = convert_grayscale_to_rgb(rgb_in)
    assert torch.equal(rgb_in, rgb_out)


def test_convert_grayscale_to_rgb_invalid():
    """Invalid channel counts or dimensions must raise errors."""
    with pytest.raises(ValueError, match="Unsupported channel count"):
        convert_grayscale_to_rgb(torch.rand(2, 2, 28, 28))


# ==============================================================================
# 3. PSNR Tests
# ==============================================================================

def test_psnr_identical_images():
    """Identical images must produce 100.0 dB (perfect reconstruction)."""
    img = torch.rand(4, 1, 28, 28)
    psnr = compute_psnr(img, img)
    assert isinstance(psnr, torch.Tensor)
    assert torch.isclose(psnr, torch.tensor(100.0))


def test_psnr_noisy_images():
    """Adding noise must produce a valid finite positive PSNR."""
    torch.manual_seed(42)
    img = torch.rand(4, 1, 28, 28)
    noisy = torch.clamp(img + 0.05 * torch.randn_like(img), 0.0, 1.0)

    psnr = compute_psnr(img, noisy)
    assert psnr.item() > 0.0
    assert psnr.item() < 100.0
    # Expected PSNR for std=0.05 noise is approx 10 * log10(1 / 0.0025) ~ 26 dB
    assert 20.0 < psnr.item() < 35.0


def test_psnr_shape_mismatch():
    """Shape mismatch must raise ValueError."""
    img1 = torch.rand(4, 1, 28, 28)
    img2 = torch.rand(2, 1, 28, 28)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_psnr(img1, img2)


def test_psnr_reduction_none():
    """reduction='none' must return a [B] tensor of per-sample PSNRs."""
    img1 = torch.rand(4, 1, 28, 28)
    img2 = torch.clamp(img1 + 0.1 * torch.randn_like(img1), 0.0, 1.0)
    psnr_per_sample = compute_psnr(img1, img2, reduction="none")
    assert psnr_per_sample.shape == (4,)
    assert (psnr_per_sample > 0.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_psnr_cpu_cuda_consistency():
    """PSNR must yield numerically identical values on CPU and CUDA."""
    torch.manual_seed(42)
    x = torch.rand(4, 1, 28, 28)
    y = torch.clamp(x + 0.1 * torch.randn_like(x), 0.0, 1.0)

    psnr_cpu = compute_psnr(x, y).item()
    psnr_cuda = compute_psnr(x.cuda(), y.cuda()).item()
    assert np.isclose(psnr_cpu, psnr_cuda, atol=1e-5)


# ==============================================================================
# 4. SSIM Tests
# ==============================================================================

def test_ssim_identical_images():
    """Identical images must produce SSIM = 1.0."""
    img = torch.rand(4, 1, 28, 28)
    ssim_val = compute_ssim(img, img)
    assert torch.isclose(ssim_val, torch.tensor(1.0), atol=1e-4)


def test_ssim_noisy_images():
    """Noisy images must produce SSIM < 1.0 and strictly positive."""
    torch.manual_seed(42)
    img = torch.rand(4, 1, 28, 28)
    noisy = torch.clamp(img + 0.1 * torch.randn_like(img), 0.0, 1.0)

    ssim_val = compute_ssim(img, noisy).item()
    assert 0.0 < ssim_val < 1.0


def test_ssim_rgb_native():
    """SSIM must support 3-channel RGB images natively."""
    img = torch.rand(2, 3, 32, 32)
    ssim_val = compute_ssim(img, img).item()
    assert np.isclose(ssim_val, 1.0, atol=1e-4)


def test_ssim_reduction_none():
    """reduction='none' must return a [B] tensor of per-sample SSIMs."""
    img1 = torch.rand(4, 1, 28, 28)
    img2 = torch.clamp(img1 + 0.1 * torch.randn_like(img1), 0.0, 1.0)
    ssim_per_sample = compute_ssim(img1, img2, reduction="none")
    assert ssim_per_sample.shape == (4,)
    assert (ssim_per_sample > 0.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ssim_cpu_cuda_consistency():
    """SSIM must yield numerically identical values on CPU and CUDA."""
    torch.manual_seed(42)
    x = torch.rand(4, 1, 28, 28)
    y = torch.clamp(x + 0.1 * torch.randn_like(x), 0.0, 1.0)

    ssim_cpu = compute_ssim(x, y).item()
    ssim_cuda = compute_ssim(x.cuda(), y.cuda()).item()
    assert np.isclose(ssim_cpu, ssim_cuda, atol=1e-4)


# ==============================================================================
# 5. FID & Fréchet Distance Tests
# ==============================================================================

def test_compute_frechet_distance_identical():
    """Identical distributions must yield Fréchet distance 0.0."""
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.2], [0.0, 0.2, 1.5]])
    fid = compute_frechet_distance(mu, sigma, mu, sigma)
    assert np.isclose(fid, 0.0, atol=1e-5)


def test_compute_frechet_distance_analytical_1d():
    """Known 1D Gaussians must match the analytical Fréchet distance."""
    # N(mu1=1.0, sigma1=4.0) vs N(mu2=4.0, sigma2=9.0)
    # d^2 = (1 - 4)^2 + (sqrt(4) - sqrt(9))^2 = (-3)^2 + (2 - 3)^2 = 9 + 1 = 10.0
    mu1 = np.array([1.0])
    sigma1 = np.array([[4.0]])
    mu2 = np.array([4.0])
    sigma2 = np.array([[9.0]])

    fid = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert np.isclose(fid, 10.0, atol=1e-5)


def test_compute_frechet_distance_analytical_2d():
    """Known 2D diagonal Gaussians must match the analytical Fréchet distance."""
    # mu1 = [0, 0], sigma1 = diag(1, 4)
    # mu2 = [1, 2], sigma2 = diag(4, 9)
    # d^2 = ||[1, 2]||^2 + (1 - 2)^2 + (2 - 3)^2 = (1 + 4) + 1 + 1 = 7.0
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.diag([1.0, 4.0])
    mu2 = np.array([1.0, 2.0])
    sigma2 = np.diag([4.0, 9.0])

    fid = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert np.isclose(fid, 7.0, atol=1e-5)


def test_compute_frechet_distance_dimension_mismatch():
    """Dimension mismatch between mean and covariance must raise ValueError."""
    mu1 = np.array([1.0, 2.0])
    sigma1 = np.eye(3)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_frechet_distance(mu1, sigma1, mu1, sigma1)


def test_compute_fid_with_custom_feature_extractor():
    """FID tooling validation using a deterministic lightweight feature extractor."""
    class ToyFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            # Project [B, 1, 28, 28] to [B, 8]
            self.net = nn.Sequential(
                nn.Conv2d(1, 4, 4, stride=4),  # [B, 4, 7, 7]
                nn.Flatten(),                 # [B, 196]
                nn.Linear(196, 8, bias=False), # [B, 8]
            )
            # Fixed deterministic weights
            with torch.no_grad():
                for p in self.parameters():
                    p.fill_(0.1)

        def forward(self, x):
            return self.net(x)

    extractor = ToyFeatureExtractor()

    torch.manual_seed(42)
    # Two identical batches of 16 synthetic samples
    images1 = torch.rand(16, 1, 28, 28)
    images2 = images1.clone()

    fid_identical = compute_fid(images1, images2, feature_extractor=extractor)
    assert np.isclose(fid_identical, 0.0, atol=1e-4)

    # Shifted images batch
    images3 = torch.clamp(images1 + 0.3, 0.0, 1.0)
    fid_shifted = compute_fid(images1, images3, feature_extractor=extractor)
    assert fid_shifted > 0.0


def test_inception_feature_extractor_structure():
    """InceptionFeatureExtractor should output 2048-dim features from 28x28 grayscale images."""
    extractor = InceptionFeatureExtractor(pretrained=True).eval()

    # Fashion-MNIST format: [B, 1, 28, 28]
    x_gray = torch.rand(2, 1, 28, 28)
    with torch.no_grad():
        feat = extractor(x_gray)

    assert feat.shape == (2, 2048)
    assert torch.isfinite(feat).all()


# ==============================================================================
# 6. Reference LPIPS (SqueezeNet v0.1) Tests
# ==============================================================================

def test_reference_lpips_smoke_test():
    """Smoke test verifying reference lpips.LPIPS(net='squeeze', version='0.1') wrapper."""
    metric = LPIPSMetric(net="squeeze", version="0.1").eval()
    assert metric.net == "squeeze"
    assert metric.version == "0.1"
    # Ensure all parameters are frozen
    for p in metric.parameters():
        assert not p.requires_grad


def test_lpips_identical_images():
    """Identical images must produce LPIPS approximately 0.0."""
    lpips_model = LPIPSMetric(net="squeeze", version="0.1").eval()
    img = torch.rand(2, 1, 28, 28)

    with torch.no_grad():
        dist = lpips_model(img, img)

    assert dist.shape == (2,)
    assert torch.allclose(dist, torch.zeros(2), atol=1e-4)


def test_lpips_noisy_images_monotonicity():
    """Perturbed images must yield positive distance and larger noise must produce higher LPIPS."""
    torch.manual_seed(42)
    lpips_model = LPIPSMetric(net="squeeze", version="0.1").eval()
    clean = torch.rand(4, 1, 28, 28)
    noise_small = torch.clamp(clean + 0.05 * torch.randn_like(clean), 0.0, 1.0)
    noise_large = torch.clamp(clean + 0.30 * torch.randn_like(clean), 0.0, 1.0)

    with torch.no_grad():
        d_small = lpips_model(clean, noise_small).mean().item()
        d_large = lpips_model(clean, noise_large).mean().item()

    assert d_small > 0.0
    assert d_large > d_small


def test_lpips_grayscale_support():
    """LPIPS must explicitly support 1-channel Fashion-MNIST images [B, 1, 28, 28]."""
    lpips_model = LPIPSMetric(net="squeeze", version="0.1").eval()
    x1 = torch.rand(2, 1, 28, 28)
    x2 = torch.clamp(x1 + 0.1 * torch.randn_like(x1), 0.0, 1.0)

    with torch.no_grad():
        dist = lpips_model(x1, x2)

    assert dist.shape == (2,)
    assert (dist > 0.0).all()


def test_compute_lpips_functional():
    """Functional compute_lpips should support reduction='mean' and 'none'."""
    img1 = torch.rand(2, 1, 28, 28)
    img2 = torch.clamp(img1 + 0.1 * torch.randn_like(img1), 0.0, 1.0)

    mean_dist = compute_lpips(img1, img2, reduction="mean")
    assert mean_dist.dim() == 0
    assert mean_dist.item() > 0.0

    none_dist = compute_lpips(img1, img2, reduction="none")
    assert none_dist.shape == (2,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lpips_cpu_cuda_consistency():
    """LPIPS must produce consistent distances on CPU and CUDA."""
    torch.manual_seed(42)
    x = torch.rand(2, 1, 28, 28)
    y = torch.clamp(x + 0.1 * torch.randn_like(x), 0.0, 1.0)

    lpips_cpu = LPIPSMetric(net="squeeze", version="0.1").to("cpu")
    lpips_cuda = LPIPSMetric(net="squeeze", version="0.1").to("cuda")

    d_cpu = compute_lpips(x, y, lpips_model=lpips_cpu, device=torch.device("cpu")).item()
    d_cuda = compute_lpips(x.cuda(), y.cuda(), lpips_model=lpips_cuda, device=torch.device("cuda")).item()
    assert np.isclose(d_cpu, d_cuda, atol=1e-3, rtol=1e-2)


# ==============================================================================
# 7. Computational Metrics Tests
# ==============================================================================

def test_count_parameters():
    """count_parameters must correctly count trainable vs total parameters."""
    model = nn.Sequential(
        nn.Linear(10, 20),  # 10*20 + 20 = 220
        nn.Linear(20, 5),   # 20*5 + 5 = 105 -> Total = 325
    )
    assert count_parameters(model, trainable_only=True) == 325
    assert count_parameters(model, trainable_only=False) == 325

    # Freeze second linear layer
    for p in model[1].parameters():
        p.requires_grad = False

    assert count_parameters(model, trainable_only=True) == 220
    assert count_parameters(model, trainable_only=False) == 325


def test_parameter_summary():
    """parameter_summary must return complete and accurate schema."""
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.Linear(200, 50),
    )
    # Freeze second layer
    for p in model[1].parameters():
        p.requires_grad = False

    summary = parameter_summary(model)
    assert summary["trainable_parameters"] == 100 * 200 + 200  # 20200
    assert summary["non_trainable_parameters"] == 200 * 50 + 50 # 10050
    assert summary["total_parameters"] == 30250
    assert summary["parameter_bytes"] == 30250 * 4  # float32 = 4 bytes
    assert summary["model_size_mb"] > 0.0


def test_measure_inference_runtime():
    """measure_inference_runtime must measure wall-clock latency and throughput."""
    dummy_model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
    x = torch.randn(16, 64)

    metrics = measure_inference_runtime(
        dummy_model,
        x,
        num_warmup=2,
        num_runs=5,
        batch_size=16,
    )

    assert "mean_time_seconds" in metrics
    assert "std_time_seconds" in metrics
    assert "min_time_seconds" in metrics
    assert "max_time_seconds" in metrics
    assert "total_time_seconds" in metrics
    assert metrics["num_runs"] == 5.0
    assert "samples_per_second" in metrics
    assert metrics["samples_per_second"] > 0.0
    assert metrics["mean_time_seconds"] > 0.0


# ==============================================================================
# 8. GenerationEvaluator Integration Tests
# ==============================================================================

def test_evaluator_paired_evaluation():
    """GenerationEvaluator must compute PSNR, SSIM, LPIPS, and FID for paired batches."""
    torch.manual_seed(42)
    evaluator = GenerationEvaluator(device="cpu")

    gen = torch.rand(4, 1, 28, 28)
    ref = torch.clamp(gen + 0.05 * torch.randn_like(gen), 0.0, 1.0)

    # Use a dummy model for parameter tracking
    model = nn.Linear(10, 10)

    results = evaluator.evaluate(
        gen_images=gen,
        ref_images=ref,
        paired=True,
        model=model,
        compute_fid_metric=False,  # Skip 2048-dim Inception in quick unit test
        compute_lpips_metric=True,
    )

    assert "image_metrics" in results
    assert "computational_metrics" in results

    im = results["image_metrics"]
    assert "psnr" in im
    assert "ssim" in im
    assert "lpips" in im
    assert 0.0 < im["psnr"] < 100.0
    assert 0.0 < im["ssim"] <= 1.0
    assert im["lpips"] > 0.0

    cm = results["computational_metrics"]
    assert cm["trainable_parameters"] == 110
    assert cm["total_parameters"] == 110


def test_evaluator_unpaired_evaluation():
    """Unpaired evaluation should focus on distribution metrics."""
    torch.manual_seed(42)
    evaluator = GenerationEvaluator(device="cpu")

    # Different number of generated vs reference images
    gen = torch.rand(6, 1, 28, 28)
    ref = torch.rand(8, 1, 28, 28)

    results = evaluator.evaluate(
        gen_images=gen,
        ref_images=ref,
        paired=False,
        compute_fid_metric=False,
    )

    # Paired metrics should not be computed
    assert "psnr" not in results["image_metrics"]
    assert "ssim" not in results["image_metrics"]
    assert "lpips" not in results["image_metrics"]


def test_evaluate_generation_convenience_function():
    """Convenience function evaluate_generation must return structured results."""
    gen = torch.rand(2, 1, 28, 28)
    ref = gen.clone()

    results = evaluate_generation(
        gen_images=gen,
        ref_images=ref,
        paired=True,
        compute_fid_metric=False,
        device="cpu",
    )

    assert results["image_metrics"]["psnr"] == 100.0
    assert np.isclose(results["image_metrics"]["ssim"], 1.0, atol=1e-4)
    assert np.isclose(results["image_metrics"]["lpips"], 0.0, atol=1e-4)
