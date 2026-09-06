"""
Unit tests for Module 3 Milestone 2A: Diffusion-to-PQC Denoiser Interface.

Verifies:
1. Input & Output Shapes:
   - [B, 9, 4, 4, 4] -> [B, 9, 6] for various batch sizes B.
   - [B, 9, 64] -> [B, 9, 6].
   - Explicit pre-projected [B, 9, 6] -> [B, 9, 6] via is_pre_projected=True.
   - Explicit definition that output is a denoised 6D projected-space representation [B, 9, 6]
     and no inverse projection / reconstruction is performed in Milestone 2A.
   - project_patches strictly accepts [B, 9, 4, 4, 4] or [B, 9, 64] and rejects [B, 9, 6].
2. Trainable Parameter Counts:
   - Exactly 18 trainable parameters in PQC for depth 3 (6 qubits x depth 3 = 18).
   - Exactly 18 trainable parameters in the entire model.
   - Projection has strictly zero trainable parameters.
   - Schedule has strictly zero trainable parameters.
   - Parameter counts tested using public PQCDenoiser APIs without coupling to internal attributes.
3. Quantum Weight Sharing & Patch Distinguishability:
   - PQC parameters are shared across all 9 patch positions.
   - Perturbing PQC parameters alters predictions across all 9 patch positions.
   - Evaluating individual patch slices through model.pqc matches denoise_patches.
   - Different input patches produce different denoised outputs (distinguishable patches).
4. Timestep Configurations:
   - Scalar integer timesteps work across the batch.
   - Scalar 0-D tensor timesteps work.
   - Per-batch 1-D vector timesteps work.
5. Error Handling & Input Validation:
   - Rejects invalid patch count (!= 9).
   - Rejects invalid patch spatial / feature dimensions.
   - Rejects batch size vs timestep vector length mismatch.
   - Rejects out-of-bounds timesteps (t < 0 or t >= T).
   - Rejects non-finite inputs (NaN / Inf).
   - Rejects non-tensor inputs.
6. Gradient Flow:
   - Gradients propagate from denoised output loss to all trainable PQC parameters.
   - Projection receives zero gradients.
7. Determinism & Finite Outputs:
   - Fixed initialization and seed produce bit-identical outputs.
   - Denoised outputs are strictly finite and in [-1, 1] (Pauli-Z expectation values).
8. Integration with LinearDiffusionSchedule:
   - Operates seamlessly with existing LinearDiffusionSchedule and custom schedules.
   - forward_with_intermediates returns all required intermediate representations.
"""

import sys
from pathlib import Path
import unittest
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.diffusion_pqc_denoiser import DiffusionPQCDenoiser
from src.models.projection import FixedPatchProjector
from src.training.diffusion_schedule import LinearDiffusionSchedule


class TestDiffusionPQCDenoiserShapesAndOutputs(unittest.TestCase):
    """Test suite for tensor geometries, shapes, and project_patches boundaries."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3, timesteps=1000)

    def test_forward_output_shape_from_5d_raw_patches(self):
        """Verify [B, 9, 4, 4, 4] -> [B, 9, 6] across multiple batch sizes."""
        for b in [1, 2, 4]:
            x_5d = torch.randn(b, 9, 4, 4, 4)
            out = self.model(x_5d, t=50, seed=42)
            self.assertEqual(
                out.shape,
                torch.Size([b, 9, 6]),
                f"Expected [B, 9, 6], got {out.shape}",
            )
            self.assertEqual(out.dtype, torch.float32)

    def test_forward_output_shape_from_3d_flattened_patches(self):
        """Verify [B, 9, 64] -> [B, 9, 6]."""
        b = 3
        x_64 = torch.randn(b, 9, 64)
        out = self.model(x_64, t=100, seed=42)
        self.assertEqual(out.shape, torch.Size([b, 9, 6]))

    def test_forward_output_explicit_pre_projected(self):
        """Verify explicit pre-projected path [B, 9, 6] -> [B, 9, 6] via is_pre_projected=True."""
        b = 2
        z_6 = torch.randn(b, 9, 6)
        out = self.model(z_6, t=25, seed=42, is_pre_projected=True)
        self.assertEqual(out.shape, torch.Size([b, 9, 6]))

    def test_output_is_explicitly_6d_projected_space_representation(self):
        """Verify output is strictly in 6-D projected space [B, 9, 6] with no inverse reconstruction."""
        x_5d = torch.randn(2, 9, 4, 4, 4)
        out = self.model(x_5d, t=0, seed=42)

        # Must be exactly 3D tensor [B, 9, 6]
        self.assertEqual(out.dim(), 3)
        self.assertEqual(out.shape[-1], 6)
        # Must not be a reconstructed spatial patch [B, 9, 4, 4, 4]
        self.assertNotEqual(out.shape, x_5d.shape)
        # Values must fall within Pauli-Z expectation range [-1.0, 1.0]
        self.assertTrue((out >= -1.0 - 1e-5).all() and (out <= 1.0 + 1e-5).all())

    def test_project_patches_strictly_rejects_6d_tensors(self):
        """Verify project_patches rejects [B, 9, 6] tensors with ValueError."""
        z_6 = torch.randn(2, 9, 6)
        with self.assertRaises(ValueError) as ctx:
            self.model.project_patches(z_6)
        self.assertIn("project_patches requires input shape [B, 9, 4, 4, 4] or [B, 9, 64]", str(ctx.exception))
        self.assertIn("Pre-projected", str(ctx.exception))

    def test_project_patches_valid_inputs(self):
        """Verify project_patches accepts only [B, 9, 4, 4, 4] and [B, 9, 64]."""
        x_5d = torch.randn(2, 9, 4, 4, 4)
        z_from_5d = self.model.project_patches(x_5d)
        self.assertEqual(z_from_5d.shape, torch.Size([2, 9, 6]))

        x_64 = x_5d.reshape(2, 9, 64)
        z_from_64 = self.model.project_patches(x_64)
        self.assertEqual(z_from_64.shape, torch.Size([2, 9, 6]))

        # Projection of identical 5D and 64D representations must match
        self.assertTrue(torch.allclose(z_from_5d, z_from_64, atol=1e-6))


class TestDiffusionPQCDenoiserParameters(unittest.TestCase):
    """Test suite verifying exact trainable parameter counts without PennyLane internal coupling."""

    def setUp(self):
        torch.manual_seed(42)

    def test_trainable_parameter_count_at_depth_3(self):
        """Verify exactly 18 trainable parameters in PQC and entire model at depth 3 (6 qubits x 3 layers)."""
        model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3)

        # 1. PQC parameter count via public property and public PQCDenoiser parameters
        self.assertEqual(model.num_pqc_parameters, 18)
        pqc_grad_params = sum(p.numel() for p in model.pqc.parameters() if p.requires_grad)
        self.assertEqual(pqc_grad_params, 18)

        # 2. Total model parameter count
        self.assertEqual(model.num_trainable_parameters, 18)
        total_model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(total_model_grad_params, 18)

    def test_projection_and_schedule_have_zero_trainable_parameters(self):
        """Verify projection and schedule components add strictly 0 trainable parameters."""
        model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3)

        self.assertEqual(model.num_projection_parameters, 0)
        proj_trainable = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
        self.assertEqual(proj_trainable, 0)

        sched_trainable = sum(p.numel() for p in model.schedule.parameters() if p.requires_grad)
        self.assertEqual(sched_trainable, 0)

    def test_configurable_depth_and_qubits(self):
        """Verify parameter count scales strictly as n_qubits * pqc_depth."""
        model_depth_2 = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=2)
        self.assertEqual(model_depth_2.num_pqc_parameters, 12)
        self.assertEqual(model_depth_2.num_trainable_parameters, 12)

        model_depth_4 = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=4)
        self.assertEqual(model_depth_4.num_pqc_parameters, 24)
        self.assertEqual(model_depth_4.num_trainable_parameters, 24)

    def test_scientific_integrity_projector_distinction(self):
        """Verify explicit distinction between experimental pre-fitted projector and test fallback."""
        # 1. Fallback uncalibrated placeholder projector (default when projector=None)
        model_default = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3)
        self.assertTrue(model_default.uses_placeholder_projector)
        self.assertTrue(torch.equal(model_default.projector.mean_64, torch.zeros(64)))
        self.assertTrue(torch.equal(model_default.projector.std_6, torch.ones(6)))

        # 2. Experimental path: user supplies pre-fitted projector derived strictly from training patches
        synthetic_train_data = torch.randn(10, 64) + 2.5
        experimental_projector = FixedPatchProjector(in_dim=64, out_dim=6, seed=42)
        experimental_projector.fit(synthetic_train_data)

        model_experimental = DiffusionPQCDenoiser(
            n_qubits=6,
            pqc_depth=3,
            projector=experimental_projector,
        )
        self.assertFalse(model_experimental.uses_placeholder_projector)
        self.assertIs(model_experimental.projector, experimental_projector)
        # Verify training-fitted statistics were preserved and not overwritten
        self.assertTrue(torch.equal(model_experimental.projector.mean_64, experimental_projector.mean_64))
        self.assertTrue(torch.equal(model_experimental.projector.std_6, experimental_projector.std_6))


class TestDiffusionPQCDenoiserWeightSharing(unittest.TestCase):
    """Test suite verifying quantum weight sharing across the 9 patch locations."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3)

    def test_pqc_parameters_shared_across_all_9_positions(self):
        """Verify denoise_patches evaluates each patch position with the exact same shared PQC."""
        z_noisy = torch.randn(2, 9, 6)
        denoised_batched = self.model.denoise_patches(z_noisy)

        # Evaluate each patch position independently through model.pqc
        for p in range(9):
            single_patch = z_noisy[:, p, :]  # [B, 6]
            expected_patch_out = self.model.pqc(single_patch)  # [B, 6]
            actual_patch_out = denoised_batched[:, p, :]  # [B, 6]
            self.assertTrue(
                torch.allclose(actual_patch_out, expected_patch_out, atol=1e-6),
                f"Patch position {p} did not match independent PQC evaluation",
            )

    def test_weight_perturbation_affects_all_patch_positions(self):
        """Verify modifying the shared PQC weights changes predictions across all 9 patch positions."""
        z_noisy = torch.randn(2, 9, 6)
        out_before = self.model.denoise_patches(z_noisy)

        # Perturb all trainable PQC parameters
        with torch.no_grad():
            for p in self.model.pqc.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        out_after = self.model.denoise_patches(z_noisy)

        # Every patch position must have changed
        for p in range(9):
            diff = (out_after[:, p, :] - out_before[:, p, :]).abs().max().item()
            self.assertGreater(diff, 1e-3, f"Patch position {p} was unaffected by PQC weight update")

    def test_different_patch_inputs_produce_different_outputs(self):
        """Verify patch distinguishability: different patch inputs yield distinct outputs."""
        # Create input where patch 0 and patch 1 have different values
        z = torch.zeros(1, 9, 6)
        z[:, 0, :] = 1.0
        z[:, 1, :] = -1.0

        out = self.model.denoise_patches(z)
        self.assertFalse(
            torch.allclose(out[:, 0, :], out[:, 1, :], atol=1e-4),
            "Different patch inputs produced identical outputs",
        )


class TestDiffusionPQCDenoiserTimesteps(unittest.TestCase):
    """Test suite verifying scalar and per-batch vector timestep execution."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3)
        self.batch_size = 4
        self.x_5d = torch.randn(self.batch_size, 9, 4, 4, 4)

    def test_scalar_integer_timestep(self):
        """Verify scalar integer timestep applies across the batch."""
        out = self.model(self.x_5d, t=100, seed=42)
        self.assertEqual(out.shape, torch.Size([self.batch_size, 9, 6]))
        self.assertTrue(torch.isfinite(out).all())

    def test_scalar_tensor_timestep(self):
        """Verify 0-D scalar tensor timestep applies across the batch."""
        out = self.model(self.x_5d, t=torch.tensor(100), seed=42)
        self.assertEqual(out.shape, torch.Size([self.batch_size, 9, 6]))
        self.assertTrue(torch.isfinite(out).all())

    def test_per_batch_vector_timesteps(self):
        """Verify 1-D vector of timesteps with distinct values per batch item."""
        t_vector = torch.tensor([0, 250, 500, 999])
        out = self.model(self.x_5d, t=t_vector, seed=42)
        self.assertEqual(out.shape, torch.Size([self.batch_size, 9, 6]))
        self.assertTrue(torch.isfinite(out).all())


class TestDiffusionPQCDenoiserValidationAndRejections(unittest.TestCase):
    """Test suite for error handling, shape rejections, and out-of-bounds guards."""

    def setUp(self):
        self.model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3, timesteps=1000)

    def test_reject_invalid_patch_counts(self):
        """Verify patch count != 9 is rejected."""
        # 8 patches instead of 9
        bad_5d = torch.randn(2, 8, 4, 4, 4)
        with self.assertRaises(ValueError):
            self.model(bad_5d, t=10)

        bad_64 = torch.randn(2, 10, 64)
        with self.assertRaises(ValueError):
            self.model(bad_64, t=10)

    def test_reject_invalid_patch_spatial_dims(self):
        """Verify spatial dimensions != (4, 4, 4) are rejected."""
        bad_spatial = torch.randn(2, 9, 3, 4, 4)
        with self.assertRaises(ValueError):
            self.model(bad_spatial, t=10)

    def test_reject_batch_size_timestep_vector_mismatch(self):
        """Verify mismatch between batch size B and timestep vector length raises ValueError."""
        x_5d = torch.randn(4, 9, 4, 4, 4)
        t_mismatch = torch.tensor([10, 20])  # Length 2 != Batch size 4
        with self.assertRaises(ValueError):
            self.model(x_5d, t=t_mismatch)

    def test_reject_out_of_bounds_timesteps(self):
        """Verify timesteps < 0 or >= T raise ValueError."""
        x_5d = torch.randn(2, 9, 4, 4, 4)
        with self.assertRaises(ValueError):
            self.model(x_5d, t=-1)

        with self.assertRaises(ValueError):
            self.model(x_5d, t=1000)

    def test_reject_non_tensor_inputs(self):
        """Verify non-tensor inputs raise TypeError."""
        with self.assertRaises(TypeError):
            self.model([[1.0]], t=0)  # type: ignore

    def test_reject_non_finite_inputs(self):
        """Verify NaN / Inf inputs raise ValueError."""
        nan_x = torch.randn(2, 9, 4, 4, 4)
        nan_x[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(nan_x, t=10)


class TestDiffusionPQCDenoiserGradients(unittest.TestCase):
    """Test suite verifying autograd backpropagation reaches PQC parameters."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3)

    def test_gradients_reach_pqc_parameters(self):
        """Verify loss computed on denoised output propagates valid gradients to PQC weights."""
        x_5d = torch.randn(2, 9, 4, 4, 4)
        t = 50

        out = self.model(x_5d, t=t, seed=42)
        loss = out.sum()
        loss.backward()

        # Check all PQC parameters have valid, non-zero, finite gradients
        pqc_params = [p for p in self.model.pqc.parameters() if p.requires_grad]
        self.assertGreater(len(pqc_params), 0)

        for p in pqc_params:
            self.assertIsNotNone(p.grad, "PQC parameter gradient was None")
            self.assertTrue(torch.isfinite(p.grad).all(), "PQC gradient contained non-finite values")
            self.assertTrue((p.grad != 0).any(), "PQC gradient was entirely zero")

    def test_projection_receives_zero_gradients(self):
        """Verify frozen projection parameters have requires_grad=False."""
        for p in self.model.projector.parameters():
            self.assertFalse(p.requires_grad)


class TestDiffusionPQCDenoiserDeterminismAndIntermediates(unittest.TestCase):
    """Test suite for reproducibility, forward_with_intermediates, and schedule integration."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3)
        self.x_5d = torch.randn(2, 9, 4, 4, 4)

    def test_deterministic_behavior_with_fixed_seed(self):
        """Verify fixed seed produces bit-identical results; different seeds produce different results."""
        out1 = self.model(self.x_5d, t=50, seed=123)
        out2 = self.model(self.x_5d, t=50, seed=123)
        self.assertTrue(torch.equal(out1, out2))

        out3 = self.model(self.x_5d, t=50, seed=456)
        self.assertFalse(torch.equal(out1, out3))

    def test_forward_with_intermediates_structure(self):
        """Verify forward_with_intermediates returns clean, noisy, denoised, and noise tensors."""
        intermediates = self.model.forward_with_intermediates(self.x_5d, t=100, seed=42)

        expected_keys = {"clean_projected", "noisy_projected", "denoised_projected", "noise"}
        self.assertEqual(set(intermediates.keys()), expected_keys)

        expected_shape = torch.Size([2, 9, 6])
        for key in expected_keys:
            self.assertEqual(intermediates[key].shape, expected_shape, f"Mismatch in {key} shape")
            self.assertTrue(torch.isfinite(intermediates[key]).all())

        # Clean + noise should relate analytically through q_sample
        sqrt_ab = self.model.schedule.sqrt_alphas_cumprod[100].item()
        sqrt_1_ab = self.model.schedule.sqrt_one_minus_alphas_cumprod[100].item()
        analytical_noisy = sqrt_ab * intermediates["clean_projected"] + sqrt_1_ab * intermediates["noise"]
        self.assertTrue(torch.allclose(intermediates["noisy_projected"], analytical_noisy, atol=1e-5))

    def test_integration_with_custom_schedule(self):
        """Verify model works with an externally provided LinearDiffusionSchedule."""
        custom_sched = LinearDiffusionSchedule(timesteps=500, beta_start=1e-3, beta_end=0.03)
        model = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3, schedule=custom_sched)

        self.assertEqual(model.timesteps, 500)
        out = model(self.x_5d, t=200, seed=42)
        self.assertEqual(out.shape, torch.Size([2, 9, 6]))

    def test_cuda_compatibility(self):
        """Verify model runs on CUDA if CUDA is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this system.")

        model_cuda = DiffusionPQCDenoiser(n_qubits=6, pqc_depth=3).cuda()
        x_cuda = self.x_5d.cuda()
        out_cuda = model_cuda(x_cuda, t=50, seed=42)

        self.assertEqual(out_cuda.device.type, "cuda")
        self.assertEqual(out_cuda.shape, torch.Size([2, 9, 6]))
        self.assertTrue(torch.isfinite(out_cuda).all())


if __name__ == "__main__":
    unittest.main()
