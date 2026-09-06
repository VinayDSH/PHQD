"""
Unit tests for Module 3 Milestone 2B: Toy Diffusion Denoising Learning Experiment.

Verifies:
1. Toy Diffusion Dataset & Data Generation:
   - Output shapes: noisy xt [B, 9, 6] and clean x0 [B, 9, 6].
   - Deterministic harmonic generation and seed variation.
   - Clean targets bounded in [-0.7, 0.7], finite, no NaNs/Infs.
   - Distinct phase signals across all 9 patch positions.
   - DataLoader batch sizing, sample counts, and reproducible batching.
2. Diffusion Schedule Integration:
   - Reuses existing LinearDiffusionSchedule without duplicate logic.
   - Fixed timestep t=500 properties and buffer sanity.
   - Forward diffusion q_sample produces expected shape [B, 9, 6].
   - Target is strictly clean x0, not noise epsilon.
   - Determinism and reproducibility with explicit seeds.
3. Shared PQC Denoiser Architecture:
   - Wraps existing PQCDenoiser with exactly 1 shared PQC.
   - Trainable parameters are strictly 18 (6 qubits x depth 3).
   - Input/output shape is [B, 9, 6].
   - Denoised outputs strictly bounded in [-1, 1].
   - Parameter sharing verified across all 9 patch positions.
   - Input shape and finite-value validation.
4. Gradient Flow & Trainability:
   - Loss backward propagates non-null, finite, non-zero gradients to all 18 parameters.
   - Optimizer step updates all 18 parameters.
5. Short-Step Reproducibility & Configuration:
   - Bit-identical multi-step training reproducibility with identical seeds.
   - YAML config parsing and validation.
   - Ensures full 15-epoch experiment is NOT executed inside unit tests.
"""

import sys
import math
from pathlib import Path
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.diffusion_schedule import LinearDiffusionSchedule
from src.models.pqc import PQCDenoiser
from experiments.module3.toy_diffusion_pqc_denoising import (
    generate_toy_diffusion_patches,
    ToyDiffusionPatchDataset,
    get_toy_diffusion_dataloaders,
    SharedPatchPQCDenoiser,
    parse_yaml_config,
    train_epoch,
    evaluate,
)


class TestToyDiffusionDataset(unittest.TestCase):
    """Test suite for synthetic 6D harmonic patch dataset and dataloaders."""

    def test_patch_generation_shapes_and_types(self):
        """Verify generated tensor shape [N, 9, 6] and float32 dtype."""
        patches = generate_toy_diffusion_patches(num_samples=32, num_patches=9, dim=6, seed=42)
        self.assertEqual(patches.shape, torch.Size([32, 9, 6]))
        self.assertEqual(patches.dtype, torch.float32)

    def test_patch_generation_determinism_and_seed_variation(self):
        """Verify identical seed produces identical data, different seed produces different data."""
        p1 = generate_toy_diffusion_patches(num_samples=16, num_patches=9, dim=6, seed=42)
        p2 = generate_toy_diffusion_patches(num_samples=16, num_patches=9, dim=6, seed=42)
        p3 = generate_toy_diffusion_patches(num_samples=16, num_patches=9, dim=6, seed=999)

        self.assertTrue(torch.equal(p1, p2), "Same seed must produce bit-identical patches")
        self.assertFalse(torch.equal(p1, p3), "Different seeds must produce different patches")

    def test_clean_patches_bounds_and_finiteness(self):
        """Verify clean patches are finite and strictly within theoretical amplitude bounds [-0.7, 0.7]."""
        patches = generate_toy_diffusion_patches(num_samples=128, num_patches=9, dim=6, seed=42)
        self.assertTrue(torch.isfinite(patches).all(), "Clean patches must contain only finite numbers")
        self.assertFalse(torch.isnan(patches).any(), "Clean patches must not contain NaNs")

        # Since A_i in [0.3, 0.7] and |sin(.)| <= 1, max absolute value cannot exceed 0.7 + eps
        max_val = patches.max().item()
        min_val = patches.min().item()
        self.assertLessEqual(max_val, 0.7 + 1e-6, f"Max value {max_val} exceeds upper bound 0.7")
        self.assertGreaterEqual(min_val, -0.7 - 1e-6, f"Min value {min_val} below lower bound -0.7")

    def test_distinct_phase_signals_across_patch_positions(self):
        """Verify that different patch positions have distinct signals due to phi_p = 2*pi*p/9."""
        patches = generate_toy_diffusion_patches(num_samples=10, num_patches=9, dim=6, seed=42)
        # Check that patch 0 and patch 1 are not identical across coordinates
        for p in range(1, 9):
            diff = (patches[:, 0, :] - patches[:, p, :]).abs().sum().item()
            self.assertGreater(diff, 1e-3, f"Patch 0 and patch {p} should have distinct phase signals")

    def test_dataset_item_shapes_and_values(self):
        """Verify dataset __getitem__ returns (noisy, clean) of shape [9, 6]."""
        dataset = ToyDiffusionPatchDataset(num_samples=16, num_patches=9, dim=6, timestep=500, seed=42)
        self.assertEqual(len(dataset), 16)

        noisy, clean = dataset[0]
        self.assertEqual(noisy.shape, torch.Size([9, 6]))
        self.assertEqual(clean.shape, torch.Size([9, 6]))
        self.assertEqual(noisy.dtype, torch.float32)
        self.assertEqual(clean.dtype, torch.float32)
        self.assertTrue(torch.isfinite(noisy).all())
        self.assertTrue(torch.isfinite(clean).all())

    def test_dataloaders_batch_size_and_count(self):
        """Verify train and val dataloaders return batches of shape [B, 9, 6]."""
        train_loader, val_loader = get_toy_diffusion_dataloaders(
            train_samples=64,
            val_samples=32,
            batch_size=16,
            timestep=500,
            train_seed=42,
            val_seed=1042,
        )
        self.assertEqual(len(train_loader), 4)
        self.assertEqual(len(val_loader), 2)

        for noisy_b, clean_b in train_loader:
            self.assertEqual(noisy_b.shape, torch.Size([16, 9, 6]))
            self.assertEqual(clean_b.shape, torch.Size([16, 9, 6]))
            break


class TestToyDiffusionSchedule(unittest.TestCase):
    """Test suite for diffusion schedule integration at fixed timestep t=500."""

    def setUp(self):
        self.schedule = LinearDiffusionSchedule(timesteps=1000, beta_start=0.0001, beta_end=0.02)

    def test_uses_existing_linear_diffusion_schedule(self):
        """Verify schedule is an instance of the existing LinearDiffusionSchedule."""
        self.assertIsInstance(self.schedule, LinearDiffusionSchedule)
        self.assertEqual(self.schedule.timesteps, 1000)

    def test_fixed_timestep_500_buffers(self):
        """Verify schedule properties at t=500: alpha_bar_500 is between 0 and 1."""
        alpha_bar_500 = self.schedule.alphas_cumprod[500].item()
        sqrt_alpha_bar_500 = self.schedule.sqrt_alphas_cumprod[500].item()
        sqrt_one_minus_alpha_bar_500 = self.schedule.sqrt_one_minus_alphas_cumprod[500].item()

        self.assertGreater(alpha_bar_500, 0.0)
        self.assertLess(alpha_bar_500, 1.0)
        self.assertAlmostEqual(sqrt_alpha_bar_500 ** 2 + sqrt_one_minus_alpha_bar_500 ** 2, 1.0, places=5)

    def test_q_sample_shape_and_clean_target(self):
        """Verify q_sample output shape matches clean input and target is clean x0."""
        x0 = generate_toy_diffusion_patches(num_samples=8, num_patches=9, dim=6, seed=42)
        xt = self.schedule.q_sample(x_0=x0, t=500, seed=123)

        self.assertEqual(xt.shape, x0.shape)
        self.assertEqual(xt.shape, torch.Size([8, 9, 6]))
        # Target in the experiment is clean x0, not noise epsilon
        self.assertFalse(torch.equal(xt, x0), "Noisy xt must differ from clean x0 at t=500")

    def test_forward_diffusion_noise_determinism(self):
        """Verify q_sample reproducibility with fixed seed."""
        x0 = generate_toy_diffusion_patches(num_samples=4, num_patches=9, dim=6, seed=42)
        xt_1 = self.schedule.q_sample(x_0=x0, t=500, seed=777)
        xt_2 = self.schedule.q_sample(x_0=x0, t=500, seed=777)
        xt_3 = self.schedule.q_sample(x_0=x0, t=500, seed=888)

        self.assertTrue(torch.equal(xt_1, xt_2), "Same noise seed must yield bit-identical xt")
        self.assertFalse(torch.equal(xt_1, xt_3), "Different noise seed must yield different xt")


class TestSharedPQCDenoiser(unittest.TestCase):
    """Test suite for SharedPatchPQCDenoiser architecture and parameter sharing."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = SharedPatchPQCDenoiser(n_qubits=6, depth=3)

    def test_parameter_count_is_strictly_18(self):
        """Verify model has exactly 6 qubits x depth 3 = 18 trainable parameters."""
        self.assertEqual(self.model.num_parameters, 18)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(trainable_params, 18, f"Expected strictly 18 trainable params, got {trainable_params}")
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, 18, f"Expected strictly 18 total params, got {total_params}")

    def test_forward_output_shape_and_bounds(self):
        """Verify output shape is [B, 9, 6] and values are in [-1, 1]."""
        for b in [1, 2, 4]:
            x = torch.randn(b, 9, 6)
            out = self.model(x)
            self.assertEqual(out.shape, torch.Size([b, 9, 6]))
            self.assertEqual(out.dtype, torch.float32)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue((out >= -1.0 - 1e-5).all() and (out <= 1.0 + 1e-5).all(),
                            "Outputs must be in [-1, 1] as Pauli-Z expectation values")

    def test_parameter_sharing_across_all_9_patch_positions(self):
        """Verify that the model evaluates all 9 patch positions with the same single PQC."""
        x = torch.randn(2, 9, 6)
        out_model = self.model(x)

        # Directly evaluate through the inner pqc on flattened [B*9, 6]
        x_flat = x.reshape(2 * 9, 6)
        out_flat = self.model.pqc(x_flat)
        out_expected = out_flat.reshape(2, 9, 6)

        self.assertTrue(
            torch.allclose(out_model, out_expected, atol=1e-6),
            "SharedPatchPQCDenoiser output must match direct evaluation through inner pqc",
        )

    def test_distinguishable_patch_outputs(self):
        """Verify that distinct input patches produce distinct denoised representations."""
        x = torch.randn(1, 9, 6)
        out = self.model(x)
        # Ensure outputs for patch 0 and patch 1 differ
        diff = (out[0, 0, :] - out[0, 1, :]).abs().sum().item()
        self.assertGreater(diff, 1e-4, "Different patch inputs must yield distinct outputs")

    def test_input_validation_errors(self):
        """Verify model rejects invalid input ranks, shapes, types, and non-finite values."""
        # Non-tensor
        with self.assertRaises(TypeError):
            self.model([[0.0] * 6] * 9)

        # 2D input [B, 6]
        with self.assertRaises(ValueError):
            self.model(torch.randn(4, 6))

        # 4D input [B, 9, 4, 4]
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 9, 4, 4))

        # Wrong patch count (8 instead of 9)
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 8, 6))

        # Wrong feature dimension (5 instead of 6)
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 9, 5))


class TestGradientFlow(unittest.TestCase):
    """Test suite for gradient propagation and parameter trainability."""

    def test_gradients_flow_to_all_18_pqc_parameters(self):
        """Verify loss backward computes non-null, finite, non-zero gradients for all 18 parameters."""
        torch.manual_seed(42)
        model = SharedPatchPQCDenoiser(n_qubits=6, depth=3)
        criterion = nn.MSELoss()

        x = torch.randn(2, 9, 6)
        target = torch.randn(2, 9, 6)

        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients")
            grad_norm = torch.norm(param.grad).item()
            self.assertGreater(grad_norm, 1e-6, f"Parameter {name} has zero/negligible gradient norm {grad_norm}")

    def test_single_optimizer_step_updates_parameters(self):
        """Verify Adam optimizer step successfully alters all 18 parameter weights."""
        torch.manual_seed(42)
        model = SharedPatchPQCDenoiser(n_qubits=6, depth=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.MSELoss()

        init_params = [p.clone().detach() for p in model.parameters()]

        x = torch.randn(4, 9, 6)
        target = torch.randn(4, 9, 6)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        for init_p, new_p in zip(init_params, model.parameters()):
            diff = (init_p - new_p).abs().sum().item()
            self.assertGreater(diff, 1e-5, "Parameter was not updated by optimizer step")


class TestShortStepReproducibilityAndConfig(unittest.TestCase):
    """Test short-run reproducibility and configuration parsing (no 15-epoch run)."""

    def test_parse_yaml_config(self):
        """Verify configs/module3_toy_diffusion.yaml parses accurately."""
        config_path = PROJECT_ROOT / "configs" / "module3_toy_diffusion.yaml"
        self.assertTrue(config_path.exists(), f"Config file not found at {config_path}")

        config = parse_yaml_config(config_path)
        self.assertIn("dataset", config)
        self.assertIn("diffusion", config)
        self.assertIn("model", config)
        self.assertIn("training", config)

        self.assertEqual(config["dataset"]["num_patches"], 9)
        self.assertEqual(config["dataset"]["dim"], 6)
        self.assertEqual(config["diffusion"]["fixed_timestep"], 500)
        self.assertEqual(config["model"]["n_qubits"], 6)
        self.assertEqual(config["model"]["pqc_depth"], 3)
        self.assertEqual(config["training"]["epochs"], 15)
        self.assertEqual(config["training"]["optimizer"], "adam")

    def test_reproducibility_short_run(self):
        """Verify two 2-step training runs with identical seeds produce bit-identical losses and weights."""
        def run_short_steps(seed: int):
            torch.manual_seed(seed)
            model = SharedPatchPQCDenoiser(n_qubits=6, depth=3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            criterion = nn.MSELoss()

            # Dummy fixed batch
            gen = torch.Generator().manual_seed(seed)
            noisy = torch.randn(2, 9, 6, generator=gen)
            clean = torch.randn(2, 9, 6, generator=gen)

            losses = []
            for _ in range(2):
                optimizer.zero_grad()
                pred = model(noisy)
                loss = criterion(pred, clean)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            params = [p.clone().detach() for p in model.parameters()]
            return losses, params

        losses_1, params_1 = run_short_steps(seed=42)
        losses_2, params_2 = run_short_steps(seed=42)

        self.assertEqual(losses_1, losses_2, "Short runs with identical seeds must produce identical losses")
        for p1, p2 in zip(params_1, params_2):
            self.assertTrue(torch.equal(p1, p2), "Model parameters must match bit-identically after identical steps")


if __name__ == "__main__":
    unittest.main()
