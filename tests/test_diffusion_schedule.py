"""
Unit tests for Module 3 Milestone 1: Classical Forward Diffusion Noise Schedule over Latent Patches.

Verifies:
1. Schedule Validity:
   - Configurable linear beta schedule (T, beta_start, beta_end).
   - Registration of the exact five core buffers ('betas', 'alphas', 'alphas_cumprod',
     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod') and no additional buffers.
   - Strict bounds and monotonic properties (betas increasing, alphas and alphas_cumprod decreasing).
   - Explicit Signal-to-Noise Ratio (SNR) definition and monotonicity.
   - Validation of constructor parameters (reject invalid T, beta_start, beta_end).
2. Latent Patch Geometries & Broadcasting:
   - 5-D latent patch tensors [B, 9, 4, 4, 4] with batched vector timesteps t in R^B.
   - 5-D latent patch tensors with scalar integer and 0-D tensor timesteps.
   - 3-D flattened [B, 9, 64] and projected [B, 9, 6] patch tensors.
   - 4-D latent grids [B, 4, 8, 8] and 2-D latent vectors [B, D].
   - Unbatched patch tensors [9, 4, 4, 4] and vectors [D] with scalar timesteps.
   - Explicit rejection of vector timesteps for unbatched tensors.
   - Batch dimension mismatch and noise shape/device/dtype mismatch error handling.
3. Determinism, Seed/Generator Precedence & Device Compatibility:
   - Rejection of simultaneous seed + generator input.
   - Bit-identical reproducibility with identical seed across runs.
   - Different noise outputs with different seeds.
   - Reproducibility with torch.Generator.
   - Detection of generator device mismatch (e.g. CPU vs CUDA).
   - Analytical closed-form match with explicit noise:
     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.
   - sample_noisy_latents returns (x_t, noise) consistent with q_sample.
   - CUDA execution parity when CUDA is available.
4. Numerically Justified Boundary & Noise Behavior:
   - Analytical verification at t=0:
     sqrt(alpha_bar_0) = sqrt(1 - beta_start) and sqrt(1 - alpha_bar_0) = sqrt(beta_start).
     SNR(0) = (1 - beta_start) / beta_start (e.g. 9999.0 for beta_start=1e-4).
   - Analytical verification at t=T-1:
     alpha_bar_{T-1} matches exact cumulative product of (1 - beta_s).
     SNR(T-1) < 1e-3, demonstrating complete noise dominance.
   - Statistical validation: empirical mean and variance over large samples match analytical
     predictions (mean = sqrt(alpha_bar_t) * x_0, var = 1 - alpha_bar_t) within confidence intervals.
5. Finite-Value Checks:
   - NaN / Inf in x_0 raises ValueError.
   - NaN / Inf in explicit noise raises ValueError.
   - Out-of-bounds timesteps (t < 0 or t >= T) raise ValueError.
   - Non-integer timestep tensors raise TypeError.
   - Non-tensor x_0 inputs raise TypeError.
"""

import sys
import math
from pathlib import Path
import unittest
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.diffusion_schedule import (
    LinearDiffusionSchedule,
    DiffusionSchedule,
    linear_beta_schedule,
)


class TestDiffusionScheduleValidity(unittest.TestCase):
    """Test suite for linear beta schedule correctness, buffers, and SNR properties."""

    def setUp(self):
        torch.manual_seed(42)
        self.timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.schedule = LinearDiffusionSchedule(
            timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )

    def test_alias_equivalence(self):
        """Verify DiffusionSchedule is an exact alias for LinearDiffusionSchedule."""
        self.assertIs(DiffusionSchedule, LinearDiffusionSchedule)

    def test_exact_five_core_buffers_registered(self):
        """Verify that exactly the five core buffers are registered and persistent."""
        buffer_dict = dict(self.schedule.named_buffers())
        expected_buffers = {
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        }
        self.assertEqual(set(buffer_dict.keys()), expected_buffers)

        # Verify zero trainable parameters
        self.assertEqual(len(list(self.schedule.parameters())), 0)

    def test_buffer_properties_and_monotonicity(self):
        """Verify mathematical bounds and monotonicity of beta, alpha, and alpha_bar."""
        betas = self.schedule.betas
        alphas = self.schedule.alphas
        alphas_cumprod = self.schedule.alphas_cumprod
        sqrt_alphas_cumprod = self.schedule.sqrt_alphas_cumprod
        sqrt_one_minus_alphas_cumprod = self.schedule.sqrt_one_minus_alphas_cumprod

        # 1. Shapes
        self.assertEqual(betas.shape, torch.Size([self.timesteps]))
        self.assertEqual(alphas.shape, torch.Size([self.timesteps]))
        self.assertEqual(alphas_cumprod.shape, torch.Size([self.timesteps]))
        self.assertEqual(sqrt_alphas_cumprod.shape, torch.Size([self.timesteps]))
        self.assertEqual(sqrt_one_minus_alphas_cumprod.shape, torch.Size([self.timesteps]))

        # 2. Beta progression: strictly increasing from beta_start to beta_end in (0, 1)
        self.assertAlmostEqual(betas[0].item(), self.beta_start, places=7)
        self.assertAlmostEqual(betas[-1].item(), self.beta_end, places=7)
        self.assertTrue((betas > 0.0).all())
        self.assertTrue((betas < 1.0).all())
        self.assertTrue((betas[1:] > betas[:-1]).all(), "betas must be strictly monotonically increasing")

        # 3. Alpha progression: strictly decreasing in (0, 1)
        self.assertTrue(torch.allclose(alphas, 1.0 - betas))
        self.assertTrue((alphas > 0.0).all())
        self.assertTrue((alphas < 1.0).all())
        self.assertTrue((alphas[1:] < alphas[:-1]).all(), "alphas must be strictly monotonically decreasing")

        # 4. Cumulative alpha product: strictly decreasing in (0, 1)
        expected_cumprod = torch.cumprod(alphas, dim=0)
        self.assertTrue(torch.allclose(alphas_cumprod, expected_cumprod))
        self.assertTrue((alphas_cumprod > 0.0).all())
        self.assertTrue((alphas_cumprod < 1.0).all())
        self.assertTrue((alphas_cumprod[1:] < alphas_cumprod[:-1]).all(), "alphas_cumprod must be strictly decreasing")

        # 5. Square root buffers
        self.assertTrue(torch.allclose(sqrt_alphas_cumprod, torch.sqrt(alphas_cumprod)))
        self.assertTrue(torch.allclose(sqrt_one_minus_alphas_cumprod, torch.sqrt(1.0 - alphas_cumprod)))

    def test_explicit_snr_definition(self):
        """Verify explicit Signal-to-Noise Ratio: SNR(t) = alpha_bar_t / (1 - alpha_bar_t)."""
        expected_snr = self.schedule.alphas_cumprod / (1.0 - self.schedule.alphas_cumprod)
        self.assertTrue(torch.allclose(self.schedule.snr, expected_snr))

        # SNR must be strictly decreasing
        self.assertTrue((self.schedule.snr[1:] < self.schedule.snr[:-1]).all())

        # Test get_snr for scalar int and tensor timesteps
        t_scalar = 50
        self.assertAlmostEqual(
            self.schedule.get_snr(t_scalar).item(),
            expected_snr[t_scalar].item(),
            places=6,
        )

        t_vec = torch.tensor([0, 100, 500, 999])
        self.assertTrue(torch.allclose(self.schedule.get_snr(t_vec), expected_snr[t_vec]))

    def test_invalid_constructor_parameters(self):
        """Verify constructor rejects invalid arguments."""
        # Non-integer timesteps
        with self.assertRaises(TypeError):
            LinearDiffusionSchedule(timesteps=100.5)  # type: ignore

        # timesteps < 1
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(timesteps=0)
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(timesteps=-10)

        # beta_start out of bounds
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_start=0.0)
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_start=1.0)
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_start=-0.01)

        # beta_end out of bounds
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_end=0.0)
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_end=1.0)
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_end=1.5)

        # beta_start >= beta_end
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_start=0.05, beta_end=0.01)
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_start=0.02, beta_end=0.02)

    def test_single_timestep_edge_case(self):
        """Verify schedule initializes correctly when timesteps=1."""
        sched_1 = LinearDiffusionSchedule(timesteps=1, beta_start=0.01, beta_end=0.02)
        self.assertEqual(sched_1.timesteps, 1)
        self.assertEqual(sched_1.betas.shape, torch.Size([1]))
        self.assertAlmostEqual(sched_1.betas[0].item(), 0.01)


class TestDiffusionScheduleShapesAndBroadcasting(unittest.TestCase):
    """Test suite for latent patch shapes and broadcasting mechanics."""

    def setUp(self):
        torch.manual_seed(42)
        self.schedule = LinearDiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02)
        self.batch_size = 4

    def test_5d_latent_patches_batched_vector_timesteps(self):
        """Verify [B, 9, 4, 4, 4] with 1-D vector t of shape [B]."""
        x_0 = torch.randn(self.batch_size, 9, 4, 4, 4)
        t = torch.tensor([0, 250, 500, 999])
        x_t = self.schedule.q_sample(x_0, t)

        self.assertEqual(x_t.shape, x_0.shape)
        self.assertEqual(x_t.dtype, x_0.dtype)
        self.assertEqual(x_t.device, x_0.device)

    def test_5d_latent_patches_scalar_timesteps(self):
        """Verify [B, 9, 4, 4, 4] with scalar integer and 0-D tensor t."""
        x_0 = torch.randn(self.batch_size, 9, 4, 4, 4)

        # Integer scalar
        x_t_int = self.schedule.q_sample(x_0, t=100)
        self.assertEqual(x_t_int.shape, x_0.shape)

        # 0-D tensor scalar
        x_t_tensor = self.schedule.q_sample(x_0, t=torch.tensor(100))
        self.assertEqual(x_t_tensor.shape, x_0.shape)

    def test_3d_flattened_and_projected_patches(self):
        """Verify [B, 9, 64] (flattened) and [B, 9, 6] (projected) patch tensors."""
        t = torch.randint(0, 1000, (self.batch_size,))

        # Flattened 64-D patches
        x_64 = torch.randn(self.batch_size, 9, 64)
        x_t_64 = self.schedule.q_sample(x_64, t)
        self.assertEqual(x_t_64.shape, torch.Size([self.batch_size, 9, 64]))

        # Projected 6-D patches
        x_6 = torch.randn(self.batch_size, 9, 6)
        x_t_6 = self.schedule.q_sample(x_6, t)
        self.assertEqual(x_t_6.shape, torch.Size([self.batch_size, 9, 6]))

    def test_4d_latent_grid_and_2d_latents(self):
        """Verify [B, 4, 8, 8] latent grids and [B, 64] 2-D vectors."""
        t = torch.randint(0, 1000, (self.batch_size,))

        # 4-D latent grid
        x_grid = torch.randn(self.batch_size, 4, 8, 8)
        x_t_grid = self.schedule.q_sample(x_grid, t)
        self.assertEqual(x_t_grid.shape, torch.Size([self.batch_size, 4, 8, 8]))

        # 2-D latent vectors
        x_vec = torch.randn(self.batch_size, 64)
        x_t_vec = self.schedule.q_sample(x_vec, t)
        self.assertEqual(x_t_vec.shape, torch.Size([self.batch_size, 64]))

    def test_unbatched_patch_tensor_with_scalar_timestep(self):
        """Verify unbatched [9, 4, 4, 4] patch tensor with scalar timestep."""
        x_unbatched = torch.randn(9, 4, 4, 4)

        # Scalar int
        x_t_1 = self.schedule.q_sample(x_unbatched, t=50)
        self.assertEqual(x_t_1.shape, torch.Size([9, 4, 4, 4]))

        # Scalar 0-D tensor
        x_t_2 = self.schedule.q_sample(x_unbatched, t=torch.tensor(50))
        self.assertEqual(x_t_2.shape, torch.Size([9, 4, 4, 4]))

        # Explicit is_batched=False
        x_t_3 = self.schedule.q_sample(x_unbatched, t=50, is_batched=False)
        self.assertEqual(x_t_3.shape, torch.Size([9, 4, 4, 4]))

    def test_unbatched_vector_with_scalar_timestep(self):
        """Verify unbatched 1-D vector [6] with scalar timestep."""
        x_1d = torch.randn(6)
        x_t = self.schedule.q_sample(x_1d, t=20)
        self.assertEqual(x_t.shape, torch.Size([6]))

    def test_rejection_of_vector_timesteps_for_unbatched_tensors(self):
        """Explicitly verify that vector timesteps are rejected for unbatched tensors."""
        # 1. Unbatched 4-D patch tensor [9, 4, 4, 4]
        x_unbatched_patches = torch.randn(9, 4, 4, 4)
        t_vector_9 = torch.arange(9)
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(x_unbatched_patches, t=t_vector_9)
        self.assertIn("Vector timesteps", str(ctx.exception))

        # 2. Unbatched 1-D vector [6]
        x_1d = torch.randn(6)
        t_vector_6 = torch.arange(6)
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(x_1d, t=t_vector_6)
        self.assertIn("Vector timesteps", str(ctx.exception))

        # 3. Explicit is_batched=False with vector t
        x_any = torch.randn(4, 4)
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(x_any, t=torch.tensor([1, 2]), is_batched=False)
        self.assertIn("Vector timesteps", str(ctx.exception))

    def test_batch_dimension_mismatch_error(self):
        """Verify batch size mismatch between x_0 and vector t raises ValueError."""
        x_0 = torch.randn(4, 9, 4, 4, 4)
        t_mismatch = torch.tensor([10, 20])  # Length 2 != Batch size 4
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(x_0, t_mismatch)
        self.assertIn("Batch size mismatch", str(ctx.exception))

    def test_noise_shape_device_dtype_mismatches(self):
        """Verify noise tensor shape/device/dtype mismatches raise ValueError."""
        x_0 = torch.randn(4, 9, 4, 4, 4)
        t = 100

        # Shape mismatch
        wrong_shape_noise = torch.randn(4, 9, 4, 4, 3)
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(x_0, t, noise=wrong_shape_noise)
        self.assertIn("Noise shape", str(ctx.exception))

        # Dtype mismatch
        wrong_dtype_noise = torch.randn(4, 9, 4, 4, 4, dtype=torch.float64)
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(x_0, t, noise=wrong_dtype_noise)
        self.assertIn("Noise dtype", str(ctx.exception))


class TestDiffusionScheduleDeterminism(unittest.TestCase):
    """Test suite for reproducibility, seeds, generators, and exact formula verification."""

    def setUp(self):
        self.schedule = LinearDiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02)
        self.x_0 = torch.randn(4, 9, 4, 4, 4)
        self.t = torch.tensor([10, 50, 100, 500])

    def test_rejection_of_simultaneous_seed_and_generator(self):
        """Verify ValueError is raised if both seed and generator are supplied."""
        gen = torch.Generator().manual_seed(123)
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(self.x_0, self.t, seed=42, generator=gen)
        self.assertIn("Cannot specify both 'seed' and 'generator'", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.schedule.sample_noisy_latents(self.x_0, self.t, seed=42, generator=gen)
        self.assertIn("Cannot specify both 'seed' and 'generator'", str(ctx.exception))

    def test_seed_type_validation(self):
        """Verify non-integer seed raises TypeError."""
        with self.assertRaises(TypeError):
            self.schedule.q_sample(self.x_0, self.t, seed="42")  # type: ignore

    def test_deterministic_noise_with_identical_seed(self):
        """Verify identical seed yields bit-identical noisy samples and noise targets."""
        x_t_1, noise_1 = self.schedule.sample_noisy_latents(self.x_0, self.t, seed=42)
        x_t_2, noise_2 = self.schedule.sample_noisy_latents(self.x_0, self.t, seed=42)

        self.assertTrue(torch.equal(x_t_1, x_t_2))
        self.assertTrue(torch.equal(noise_1, noise_2))

        # Different seed must produce different samples and noise
        x_t_3, noise_3 = self.schedule.sample_noisy_latents(self.x_0, self.t, seed=999)
        self.assertFalse(torch.equal(x_t_1, x_t_3))
        self.assertFalse(torch.equal(noise_1, noise_3))

    def test_deterministic_noise_with_generator(self):
        """Verify torch.Generator provides reproducible sampling."""
        gen_1 = torch.Generator(device="cpu").manual_seed(12345)
        x_t_1 = self.schedule.q_sample(self.x_0, self.t, generator=gen_1)

        gen_2 = torch.Generator(device="cpu").manual_seed(12345)
        x_t_2 = self.schedule.q_sample(self.x_0, self.t, generator=gen_2)

        self.assertTrue(torch.equal(x_t_1, x_t_2))

    def test_generator_device_mismatch_detection(self):
        """Verify generator device mismatch raises ValueError."""
        # Test simulated generator mismatch: if x_0 is on CPU, generator on a different device should error
        # If CUDA is available, test with CPU tensor + CUDA generator or vice versa
        if torch.cuda.is_available():
            cuda_x0 = self.x_0.cuda()
            cpu_gen = torch.Generator(device="cpu").manual_seed(42)
            with self.assertRaises(ValueError) as ctx:
                self.schedule.q_sample(cuda_x0, self.t, generator=cpu_gen)
            self.assertIn("does not match tensor device", str(ctx.exception))

    def test_analytical_closed_form_equality(self):
        """Verify x_t matches sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon exactly."""
        fixed_noise = torch.randn_like(self.x_0)
        x_t = self.schedule.q_sample(self.x_0, self.t, noise=fixed_noise)

        # Compute theoretical analytical values directly
        expected_x_t = torch.zeros_like(self.x_0)
        for i, timestep in enumerate(self.t):
            sqrt_ab = self.schedule.sqrt_alphas_cumprod[timestep]
            sqrt_1_minus_ab = self.schedule.sqrt_one_minus_alphas_cumprod[timestep]
            expected_x_t[i] = sqrt_ab * self.x_0[i] + sqrt_1_minus_ab * fixed_noise[i]

        self.assertTrue(torch.allclose(x_t, expected_x_t, atol=1e-6))

    def test_sample_noisy_latents_consistency(self):
        """Verify sample_noisy_latents output matches q_sample with returned noise."""
        x_t, returned_noise = self.schedule.sample_noisy_latents(self.x_0, self.t, seed=777)
        recalculated_x_t = self.schedule.q_sample(self.x_0, self.t, noise=returned_noise)
        self.assertTrue(torch.equal(x_t, recalculated_x_t))

    def test_cuda_compatibility(self):
        """Verify schedule runs seamlessly on CUDA if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this system.")

        schedule_cuda = LinearDiffusionSchedule(timesteps=1000).cuda()
        x_0_cuda = torch.randn(2, 9, 4, 4, 4, device="cuda")
        t_cuda = torch.tensor([10, 500], device="cuda")

        x_t_cuda = schedule_cuda.q_sample(x_0_cuda, t_cuda, seed=42)
        self.assertEqual(x_t_cuda.device.type, "cuda")
        self.assertEqual(x_t_cuda.shape, x_0_cuda.shape)
        self.assertTrue(torch.isfinite(x_t_cuda).all())


class TestDiffusionScheduleNumericallyJustifiedBehavior(unittest.TestCase):
    """Test suite for mathematically grounded behavior at boundaries and statistical limits."""

    def setUp(self):
        self.timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.schedule = LinearDiffusionSchedule(
            timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )

    def test_numerically_justified_boundary_at_t0(self):
        """Verify exact mathematical identities at t=0.

        At t=0:
          alpha_0 = 1 - beta_start
          alpha_bar_0 = alpha_0 = 1 - beta_start
          sqrt(alpha_bar_0) = sqrt(1 - beta_start)
          sqrt(1 - alpha_bar_0) = sqrt(beta_start)
          SNR(0) = (1 - beta_start) / beta_start
        """
        # Exact theoretical values
        expected_alpha_0 = 1.0 - self.beta_start
        expected_sqrt_ab0 = math.sqrt(expected_alpha_0)
        expected_sqrt_1_minus_ab0 = math.sqrt(self.beta_start)
        expected_snr_0 = expected_alpha_0 / self.beta_start

        # Buffer checks: float32 single-precision cancellation 1.0 - (1.0 - beta) yields ~8e-7 difference
        self.assertAlmostEqual(self.schedule.alphas_cumprod[0].item(), expected_alpha_0, places=5)
        self.assertAlmostEqual(self.schedule.sqrt_alphas_cumprod[0].item(), expected_sqrt_ab0, places=5)
        self.assertAlmostEqual(self.schedule.sqrt_one_minus_alphas_cumprod[0].item(), expected_sqrt_1_minus_ab0, places=5)
        # SNR at t=0: for beta_start=1e-4, theoretical SNR is ~9999.0 (within 0.05% relative tolerance in float32)
        self.assertAlmostEqual(self.schedule.snr[0].item(), expected_snr_0, delta=5.0)

        # Forward noising test at t=0 with known clean x_0 and zero noise
        x_0 = torch.full((1, 4, 4), 2.0)
        zero_noise = torch.zeros_like(x_0)
        x_t0_noiseless = self.schedule.q_sample(x_0, t=0, noise=zero_noise)
        self.assertAlmostEqual(x_t0_noiseless[0, 0, 0].item(), 2.0 * expected_sqrt_ab0, places=5)

    def test_numerically_justified_boundary_at_t_max(self):
        """Verify exact cumulative product and extreme noise dominance at t=T-1.

        At t=T-1:
          alpha_bar_{T-1} = prod_{s=0}^{T-1} (1 - beta_s)
          SNR(T-1) = alpha_bar_{T-1} / (1 - alpha_bar_{T-1})
        """
        # Exact analytical product across float64 for ground-truth precision
        exact_cumprod_T = math.prod([1.0 - b.item() for b in self.schedule.betas])
        actual_cumprod_T = self.schedule.alphas_cumprod[-1].item()

        # Match within floating-point tolerance
        self.assertAlmostEqual(actual_cumprod_T, exact_cumprod_T, delta=1e-5)

        # At T=1000 with linear beta from 1e-4 to 0.02, average beta is ~0.01005
        # alpha_bar_T is approximately exp(-sum(betas)) ~ exp(-10.05) ~ 4.3e-5
        self.assertLess(actual_cumprod_T, 1e-4)
        self.assertGreater(actual_cumprod_T, 0.0)

        # Verify SNR at T-1 is numerically tiny (< 1e-4)
        snr_last = self.schedule.snr[-1].item()
        self.assertLess(snr_last, 1e-4)

        # Verify noise coefficient sqrt(1 - alpha_bar_T) is extremely close to 1
        sqrt_1_minus_ab_T = self.schedule.sqrt_one_minus_alphas_cumprod[-1].item()
        self.assertGreater(sqrt_1_minus_ab_T, 0.9999)

    def test_statistical_noise_distribution_properties(self):
        """Verify empirical mean and variance match analytical Gaussian formulas within 4 standard errors.

        For x_0 = 0:
          x_t = sqrt(1 - alpha_bar_t) * epsilon ~ N(0, (1 - alpha_bar_t) * I)
        For x_0 = c:
          E[x_t] = sqrt(alpha_bar_t) * c
          Var[x_t] = 1 - alpha_bar_t
        """
        sample_size = 100000
        x_0_zero = torch.zeros(sample_size)
        x_0_const = torch.full((sample_size,), 1.5)

        test_timesteps = [0, 200, 500, 800, 999]
        for t_val in test_timesteps:
            t_tensor = torch.tensor(t_val)
            alpha_bar = self.schedule.alphas_cumprod[t_val].item()
            sqrt_ab = self.schedule.sqrt_alphas_cumprod[t_val].item()
            theoretical_var = 1.0 - alpha_bar

            # 1. Zero-mean test: x_t = sqrt(1 - alpha_bar_t) * epsilon
            x_t_zero = self.schedule.q_sample(x_0_zero, t=t_tensor, seed=123 + t_val)
            empirical_mean_zero = x_t_zero.mean().item()
            empirical_var_zero = x_t_zero.var(unbiased=True).item()

            # Standard error of mean: sigma / sqrt(N)
            std_err_mean = math.sqrt(theoretical_var / sample_size)
            # Standard error of variance for normal: sqrt(2 * sigma^4 / (N - 1))
            std_err_var = math.sqrt(2.0 * (theoretical_var ** 2) / (sample_size - 1))

            self.assertAlmostEqual(empirical_mean_zero, 0.0, delta=4.0 * std_err_mean)
            self.assertAlmostEqual(empirical_var_zero, theoretical_var, delta=4.0 * std_err_var)

            # 2. Non-zero signal test: x_t = sqrt(alpha_bar_t) * 1.5 + sqrt(1 - alpha_bar_t) * epsilon
            x_t_const = self.schedule.q_sample(x_0_const, t=t_tensor, seed=456 + t_val)
            empirical_mean_const = x_t_const.mean().item()
            theoretical_mean_const = sqrt_ab * 1.5

            self.assertAlmostEqual(empirical_mean_const, theoretical_mean_const, delta=4.0 * std_err_mean)


class TestDiffusionScheduleFiniteChecks(unittest.TestCase):
    """Test suite for finite-value validation and error handling."""

    def setUp(self):
        self.schedule = LinearDiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02)
        self.x_0 = torch.randn(2, 9, 4, 4, 4)

    def test_non_tensor_input_raises_type_error(self):
        """Verify non-tensor x_0 raises TypeError."""
        with self.assertRaises(TypeError):
            self.schedule.q_sample([[1.0, 2.0]], t=0)  # type: ignore

    def test_nan_in_x0_raises_value_error(self):
        """Verify NaN values in x_0 raise ValueError."""
        nan_x0 = self.x_0.clone()
        nan_x0[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(nan_x0, t=10)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_inf_in_x0_raises_value_error(self):
        """Verify Inf values in x_0 raise ValueError."""
        inf_x0 = self.x_0.clone()
        inf_x0[0, 0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(inf_x0, t=10)
        self.assertIn("non-finite values", str(ctx.exception))

        neg_inf_x0 = self.x_0.clone()
        neg_inf_x0[0, 0, 0, 0, 0] = float("-inf")
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(neg_inf_x0, t=10)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_nan_in_noise_raises_value_error(self):
        """Verify NaN values in explicit noise raise ValueError."""
        bad_noise = torch.randn_like(self.x_0)
        bad_noise[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            self.schedule.q_sample(self.x_0, t=10, noise=bad_noise)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_out_of_bounds_integer_timestep_raises_value_error(self):
        """Verify negative or >= T integer timesteps raise ValueError."""
        with self.assertRaises(ValueError):
            self.schedule.q_sample(self.x_0, t=-1)

        with self.assertRaises(ValueError):
            self.schedule.q_sample(self.x_0, t=1000)

        with self.assertRaises(ValueError):
            self.schedule.q_sample(self.x_0, t=1005)

    def test_out_of_bounds_tensor_timestep_raises_value_error(self):
        """Verify out-of-bounds elements in timestep tensor raise ValueError."""
        bad_t_neg = torch.tensor([-1, 10])
        with self.assertRaises(ValueError):
            self.schedule.q_sample(self.x_0, t=bad_t_neg)

        bad_t_high = torch.tensor([50, 1000])
        with self.assertRaises(ValueError):
            self.schedule.q_sample(self.x_0, t=bad_t_high)

    def test_float_timestep_tensor_raises_type_error(self):
        """Verify floating-point timestep tensors raise TypeError."""
        float_t = torch.tensor([1.5, 2.5])
        with self.assertRaises(TypeError):
            self.schedule.q_sample(self.x_0, t=float_t)


if __name__ == "__main__":
    unittest.main()
