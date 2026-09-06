"""
Unit tests for Module 3 Milestone 5: Combined Loss and lambda_seam Ablation.

Verifies:
1. Exact loss formulation: L_total == L_diff + lambda_seam * L_seam + lambda_recon * L_recon.
2. Behavior when lambda_seam = 0 (baseline setting).
3. Behavior when lambda_seam > 0 (proportional seam contribution).
4. Detailed component decomposition via forward_with_components.
5. Strictly zero trainable parameters in CombinedLoss.
6. Finite-value and dimension validation (rejects NaNs, Infs, dimension mismatches).
7. Batch handling: supports B in {1, 2, 4} and unbatched inputs.
8. Full gradient flow through toy patch denoiser.
9. Deterministic pregeneration of shared ablation datasets and paired model initializations.
10. Configuration parsing and validation of lambda ablation parameters.
"""

import sys
import copy
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patch_extractor import extract_latent_patches
from src.losses.combined_loss import CombinedLoss
from src.losses.seam_consistency import SeamConsistencyLoss
from src.models.stitching_cnn import OverlapAverageStitcher
from experiments.module3.toy_combined_loss_ablation import (
    ToyPatchDenoiser,
    generate_synthetic_latents,
    pregenerate_paired_datasets,
    parse_yaml_config,
)


class TestCombinedLossFormulation(unittest.TestCase):
    """Test suite for CombinedLoss formulation, component decomposition, and parameter count."""

    def setUp(self):
        torch.manual_seed(42)
        self.loss_fn = CombinedLoss(lambda_seam=0.05, lambda_recon=1.0)
        self.clean_z = generate_synthetic_latents(num_samples=2, seed=42)
        self.clean_p = extract_latent_patches(self.clean_z, patch_size=4, stride=2)
        self.pred_p = self.clean_p + 0.1 * torch.randn_like(self.clean_p)

    def test_exact_loss_formulation_and_manual_parity(self):
        """Verify L_total matches manually computed sum: L_diff + lambda_seam*L_seam + lambda_recon*L_recon."""
        l_total = self.loss_fn(self.pred_p, self.clean_p, self.clean_z)

        # Manual computation of components
        l_diff = F.mse_loss(self.pred_p, self.clean_p)
        l_seam = SeamConsistencyLoss()(self.pred_p)
        z_rec = OverlapAverageStitcher()(self.pred_p)
        l_recon = F.mse_loss(z_rec, self.clean_z)
        l_expected = l_diff + (0.05 * l_seam) + (1.0 * l_recon)

        self.assertTrue(torch.allclose(l_total, l_expected, atol=1e-6))

    def test_lambda_seam_zero_behavior(self):
        """Verify that when lambda_seam = 0, seam loss is completely excluded from total loss."""
        loss_fn_zero = CombinedLoss(lambda_seam=0.0, lambda_recon=1.0)
        l_total = loss_fn_zero(self.pred_p, self.clean_p, self.clean_z)

        l_diff = F.mse_loss(self.pred_p, self.clean_p)
        z_rec = OverlapAverageStitcher()(self.pred_p)
        l_recon = F.mse_loss(z_rec, self.clean_z)
        l_expected = l_diff + l_recon

        self.assertTrue(torch.allclose(l_total, l_expected, atol=1e-6))

    def test_lambda_seam_positive_scaling(self):
        """Verify total loss increases monotonically as lambda_seam increases on perturbed patches."""
        l_0 = CombinedLoss(lambda_seam=0.0, lambda_recon=1.0)(self.pred_p, self.clean_p, self.clean_z)
        l_01 = CombinedLoss(lambda_seam=0.01, lambda_recon=1.0)(self.pred_p, self.clean_p, self.clean_z)
        l_05 = CombinedLoss(lambda_seam=0.05, lambda_recon=1.0)(self.pred_p, self.clean_p, self.clean_z)
        l_10 = CombinedLoss(lambda_seam=0.10, lambda_recon=1.0)(self.pred_p, self.clean_p, self.clean_z)

        self.assertLess(l_0.item(), l_01.item())
        self.assertLess(l_01.item(), l_05.item())
        self.assertLess(l_05.item(), l_10.item())

    def test_forward_with_components_keys_and_values(self):
        """Verify forward_with_components returns all expected keys and matching total."""
        comp = self.loss_fn.forward_with_components(self.pred_p, self.clean_p, self.clean_z)
        for key in ["total", "diffusion", "seam", "horizontal_seam", "vertical_seam", "recon", "lambda_seam", "lambda_recon"]:
            self.assertIn(key, comp)

        total_forward = self.loss_fn(self.pred_p, self.clean_p, self.clean_z)
        self.assertTrue(torch.allclose(comp["total"], total_forward, atol=1e-6))

    def test_zero_trainable_parameters(self):
        """Verify CombinedLoss has strictly 0 trainable parameters."""
        self.assertEqual(self.loss_fn.num_parameters, 0)
        self.assertEqual(sum(p.numel() for p in self.loss_fn.parameters()), 0)

    def test_unbatched_and_batched_tensor_support(self):
        """Verify loss computes accurately for B in {1, 2, 4} and unbatched inputs."""
        for b in [1, 2, 4]:
            z = generate_synthetic_latents(num_samples=b, seed=42)
            p_clean = extract_latent_patches(z, patch_size=4, stride=2)
            p_pred = p_clean + 0.05 * torch.randn_like(p_clean)

            loss = self.loss_fn(p_pred, p_clean, z)
            self.assertEqual(loss.dim(), 0)
            self.assertTrue(torch.isfinite(loss))

        # Unbatched [9, 4, 4, 4] and [4, 8, 8]
        z_unb = self.clean_z[0]
        p_clean_unb = self.clean_p[0]
        p_pred_unb = self.pred_p[0]
        loss_unb = self.loss_fn(p_pred_unb, p_clean_unb, z_unb)
        self.assertEqual(loss_unb.dim(), 0)
        self.assertTrue(torch.isfinite(loss_unb))

    def test_input_validation_errors(self):
        """Verify loss rejects non-tensors, mismatched shapes, and non-finite values."""
        # Non-tensor
        with self.assertRaises(TypeError):
            self.loss_fn([[0.0]], self.clean_p, self.clean_z)

        # Batch size mismatch
        with self.assertRaises(ValueError):
            self.loss_fn(self.pred_p, self.clean_p[:1], self.clean_z)

        # NaN / Inf
        nan_p = self.pred_p.clone()
        nan_p[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.loss_fn(nan_p, self.clean_p, self.clean_z)

        # Negative lambda
        with self.assertRaises(ValueError):
            CombinedLoss(lambda_seam=-0.1)
        with self.assertRaises(ValueError):
            CombinedLoss(lambda_recon=-0.1)


class TestCombinedLossGradientsAndOptimization(unittest.TestCase):
    """Test suite for gradient flow and trainability under combined loss."""

    def test_gradient_flow_to_toy_denoiser(self):
        """Verify combined loss backpropagation computes non-null, finite gradients on all parameters."""
        torch.manual_seed(42)
        model = ToyPatchDenoiser()
        loss_fn = CombinedLoss(lambda_seam=0.05, lambda_recon=1.0)

        z = generate_synthetic_latents(num_samples=2, seed=42)
        clean_p = extract_latent_patches(z, patch_size=4, stride=2)
        noisy_p = clean_p + 0.25 * torch.randn_like(clean_p)

        pred_p = model(noisy_p)
        loss = loss_fn(pred_p, clean_p, z)
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradient")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient")
            self.assertGreater(torch.norm(param.grad).item(), 1e-6, f"Parameter {name} has zero gradient")

    def test_training_reduces_total_loss(self):
        """Verify optimizer steps strictly reduce total loss."""
        torch.manual_seed(42)
        model = ToyPatchDenoiser()
        loss_fn = CombinedLoss(lambda_seam=0.05, lambda_recon=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        z = generate_synthetic_latents(num_samples=4, seed=42)
        clean_p = extract_latent_patches(z, patch_size=4, stride=2)
        noisy_p = clean_p + 0.25 * torch.randn_like(clean_p)

        with torch.no_grad():
            init_loss = loss_fn(model(noisy_p), clean_p, z).item()

        for _ in range(5):
            optimizer.zero_grad()
            pred = model(noisy_p)
            loss = loss_fn(pred, clean_p, z)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_loss = loss_fn(model(noisy_p), clean_p, z).item()

        self.assertLess(final_loss, init_loss)


class TestAblationSetupAndConfig(unittest.TestCase):
    """Test suite for paired ablation determinism and configuration parsing."""

    def test_pregenerated_paired_datasets_consistency(self):
        """Verify pregenerate_paired_datasets produces bit-identical tensors with same seed."""
        d1 = pregenerate_paired_datasets(train_samples=16, val_samples=8, train_seed=42, val_seed=1042)
        d2 = pregenerate_paired_datasets(train_samples=16, val_samples=8, train_seed=42, val_seed=1042)

        for k in ["train_z0", "train_x0", "train_xt", "val_z0", "val_x0", "val_xt"]:
            self.assertTrue(torch.equal(d1[k], d2[k]), f"Tensor {k} must be bit-identical across runs")

    def test_paired_model_initialization_parity(self):
        """Verify canonical state dict can be restored identically across runs."""
        torch.manual_seed(42)
        canonical = ToyPatchDenoiser()
        state_dict = copy.deepcopy(canonical.state_dict())

        m1 = ToyPatchDenoiser()
        m1.load_state_dict(copy.deepcopy(state_dict))

        m2 = ToyPatchDenoiser()
        m2.load_state_dict(copy.deepcopy(state_dict))

        x = torch.randn(2, 9, 4, 4, 4)
        out1 = m1(x)
        out2 = m2(x)
        self.assertTrue(torch.equal(out1, out2), "Identically initialized models must produce bit-identical output")

    def test_parse_yaml_config(self):
        """Verify configs/module3_combined_loss.yaml parses correctly with all 4 lambda values."""
        config_path = PROJECT_ROOT / "configs" / "module3_combined_loss.yaml"
        self.assertTrue(config_path.exists(), f"Config file not found at {config_path}")

        config = parse_yaml_config(config_path)
        self.assertIn("ablation", config)
        self.assertEqual(config["ablation"]["lambda_recon"], 1.0)
        self.assertEqual(config["ablation"]["lambda_seam_values"], [0.0, 0.01, 0.05, 0.10])
        self.assertEqual(config["training"]["epochs"], 15)
        self.assertEqual(config["training"]["batch_size"], 32)
        self.assertEqual(config["training"]["learning_rate"], 0.001)


if __name__ == "__main__":
    unittest.main()
