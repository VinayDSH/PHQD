"""
Unit tests for Module 3 Milestone 4: Seam-Consistency Loss.

Verifies:
1. Zero seam loss for perfectly matching overlaps (clean extracted patches).
2. Positive seam loss for deliberately mismatched overlaps.
3. Horizontal-only mismatch isolation (L_H > 0, L_V == 0).
4. Vertical-only mismatch isolation (L_V > 0, L_H == 0).
5. Exact component formulation: L_seam == (L_H + L_V) / 2.
6. Correct neighbor indexing for all 6 horizontal and 6 vertical pairs.
7. Batch handling: supports B in {1, 2, 4} and unbatched [9, 4, 4, 4].
8. Finite-value and dimension validation (rejects NaNs, Infs, wrong ranks/dims).
9. Strictly zero trainable parameters in SeamConsistencyLoss.
10. Deterministic results.
11. Full gradient flow through the toy correction network.
12. Training optimization reduces seam loss.
13. Config parsing and validation.
"""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patch_extractor import extract_latent_patches
from src.losses.seam_consistency import (
    SeamConsistencyLoss,
    HORIZONTAL_PAIRS,
    VERTICAL_PAIRS,
)
from experiments.module3.toy_seam_consistency import (
    PatchSeamCorrectionNet,
    parse_yaml_config,
    generate_synthetic_latents,
)


class TestSeamConsistencyLossGeometry(unittest.TestCase):
    """Test suite for SeamConsistencyLoss geometry, mathematical properties, and pair isolation."""

    def setUp(self):
        self.loss_fn = SeamConsistencyLoss(num_patches=9, channels=4, patch_size=4, overlap=2)
        # Generate clean synthetic patches from clean continuous latent
        self.clean_z = generate_synthetic_latents(num_samples=4, seed=42)
        self.clean_patches = extract_latent_patches(self.clean_z, patch_size=4, stride=2)

    def test_zero_seam_loss_for_matching_overlaps(self):
        """Verify clean patches extracted from smooth continuous latents produce exact 0.0 seam loss."""
        total_loss = self.loss_fn(self.clean_patches)
        lh = self.loss_fn.compute_horizontal_seam_loss(self.clean_patches)
        lv = self.loss_fn.compute_vertical_seam_loss(self.clean_patches)

        self.assertAlmostEqual(total_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(lh.item(), 0.0, places=6)
        self.assertAlmostEqual(lv.item(), 0.0, places=6)

    def test_positive_loss_for_mismatched_overlaps(self):
        """Verify noisy/mismatched patches produce strictly positive seam loss."""
        noise = 0.25 * torch.randn_like(self.clean_patches)
        noisy_patches = self.clean_patches + noise

        total_loss = self.loss_fn(noisy_patches)
        self.assertGreater(total_loss.item(), 0.01)

    def test_horizontal_only_mismatch(self):
        """Verify perturbing only horizontal overlap region makes L_H > 0 while L_V == 0.0."""
        # Patch 0 right overlap is W in [2:4].
        # If we perturb P0 at row 0, col 2:
        # - It affects horizontal pair (P0, P1) because W=2 is in horizontal overlap.
        # - Does it affect vertical pair (P0, P3)? P0 vertical overlap is H in [2:4]. Row 0 is NOT in [2:4]!
        # Thus row 0, col 2 is in horizontal overlap but NOT in vertical overlap!
        p_mismatched = self.clean_patches.clone()
        p_mismatched[:, 0, :, 0, 2] += 2.0  # row 0 (not vertical overlap), col 2 (horizontal overlap)

        lh = self.loss_fn.compute_horizontal_seam_loss(p_mismatched)
        lv = self.loss_fn.compute_vertical_seam_loss(p_mismatched)

        self.assertGreater(lh.item(), 0.0, "Horizontal loss must be positive")
        self.assertAlmostEqual(lv.item(), 0.0, places=6, msg="Vertical loss must remain exactly zero")

    def test_vertical_only_mismatch(self):
        """Verify perturbing only vertical overlap region makes L_V > 0 while L_H == 0.0."""
        # Patch 6 (bottom-left) top overlap with P3 is H in [0:2].
        # Patch 6 right overlap with P7 is W in [2:4].
        # If we perturb P6 at row 0, col 0:
        # - Row 0 is in vertical overlap H in [0:2] with P3.
        # - Col 0 is NOT in horizontal overlap W in [2:4] with P7.
        # Thus row 0, col 0 of P6 affects ONLY vertical seam!
        p_mismatched = self.clean_patches.clone()
        p_mismatched[:, 6, :, 0, 0] += 2.0

        lh = self.loss_fn.compute_horizontal_seam_loss(p_mismatched)
        lv = self.loss_fn.compute_vertical_seam_loss(p_mismatched)

        self.assertAlmostEqual(lh.item(), 0.0, places=6, msg="Horizontal loss must remain exactly zero")
        self.assertGreater(lv.item(), 0.0, "Vertical loss must be positive")

    def test_exact_components_formulation(self):
        """Verify total loss is strictly (L_horizontal + L_vertical) / 2."""
        noise = 0.2 * torch.randn_like(self.clean_patches)
        noisy = self.clean_patches + noise

        comp = self.loss_fn.forward_with_components(noisy)
        expected_total = (comp["horizontal"] + comp["vertical"]) / 2.0

        self.assertTrue(torch.allclose(comp["total"], expected_total, atol=1e-6))
        self.assertEqual(len(comp["h_pair_losses"]), 6)
        self.assertEqual(len(comp["v_pair_losses"]), 6)

    def test_correct_neighbor_indexing_horizontal_pairs(self):
        """Verify each of the 6 horizontal neighbor pairs individually."""
        expected_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
        self.assertEqual(HORIZONTAL_PAIRS, expected_pairs)

        for left_idx, right_idx in expected_pairs:
            p = self.clean_patches.clone()
            # Perturb right overlap of left patch
            p[:, left_idx, :, 0, 3] += 1.0

            comp = self.loss_fn.forward_with_components(p)
            for item in comp["h_pair_losses"]:
                pair = item["pair"]
                loss = item["loss"]
                if pair == (left_idx, right_idx):
                    self.assertGreater(loss, 0.05, f"Pair {pair} should have positive loss")
                else:
                    self.assertAlmostEqual(loss, 0.0, places=6, msg=f"Pair {pair} should have zero loss")

    def test_correct_neighbor_indexing_vertical_pairs(self):
        """Verify each of the 6 vertical neighbor pairs individually."""
        expected_pairs = [(0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]
        self.assertEqual(VERTICAL_PAIRS, expected_pairs)

        for top_idx, bottom_idx in expected_pairs:
            p = self.clean_patches.clone()
            # Perturb bottom overlap of top patch at col 0 (which is not horizontal overlap for patch 0, 3, 6)
            # Use col 1 (not horizontal overlap for rightmost, col 0 for left)
            col = 0 if top_idx in [0, 3] else (1 if top_idx in [1, 4] else 1)
            p[:, top_idx, :, 3, col] += 1.0

            comp = self.loss_fn.forward_with_components(p)
            for item in comp["v_pair_losses"]:
                pair = item["pair"]
                loss = item["loss"]
                if pair == (top_idx, bottom_idx):
                    self.assertGreater(loss, 0.05, f"Pair {pair} should have positive loss")
                else:
                    self.assertAlmostEqual(loss, 0.0, places=6, msg=f"Pair {pair} should have zero loss")

    def test_batch_handling_and_unbatched_support(self):
        """Verify loss computes correctly for B in {1, 2, 4} and unbatched [9, 4, 4, 4]."""
        for b in [1, 2, 4]:
            p = torch.randn(b, 9, 4, 4, 4)
            loss = self.loss_fn(p)
            self.assertEqual(loss.dim(), 0)
            self.assertTrue(torch.isfinite(loss))

        # Unbatched [9, 4, 4, 4]
        p_unb = torch.randn(9, 4, 4, 4)
        loss_unb = self.loss_fn(p_unb)
        self.assertEqual(loss_unb.dim(), 0)
        self.assertTrue(torch.isfinite(loss_unb))

    def test_zero_trainable_parameters(self):
        """Verify SeamConsistencyLoss has strictly zero trainable parameters."""
        self.assertEqual(self.loss_fn.num_parameters, 0)
        total_p = sum(p.numel() for p in self.loss_fn.parameters())
        self.assertEqual(total_p, 0)

    def test_input_validation_errors(self):
        """Verify loss rejects non-tensors, invalid ranks, wrong patch/channel/spatial dimensions, and NaNs."""
        # Non-tensor
        with self.assertRaises(TypeError):
            self.loss_fn([[0.0] * 4] * 9)

        # 3D tensor
        with self.assertRaises(ValueError):
            self.loss_fn(torch.randn(9, 4, 4))

        # 6D tensor
        with self.assertRaises(ValueError):
            self.loss_fn(torch.randn(1, 1, 9, 4, 4, 4))

        # Wrong patch count (8 instead of 9)
        with self.assertRaises(ValueError):
            self.loss_fn(torch.randn(2, 8, 4, 4, 4))

        # Wrong channels (3 instead of 4)
        with self.assertRaises(ValueError):
            self.loss_fn(torch.randn(2, 9, 3, 4, 4))

        # Wrong spatial size (5x5 instead of 4x4)
        with self.assertRaises(ValueError):
            self.loss_fn(torch.randn(2, 9, 4, 5, 5))

        # NaN / Inf
        nan_p = torch.randn(2, 9, 4, 4, 4)
        nan_p[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.loss_fn(nan_p)

        inf_p = torch.randn(2, 9, 4, 4, 4)
        inf_p[0, 0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            self.loss_fn(inf_p)

    def test_deterministic_loss_computation(self):
        """Verify identical inputs produce bit-identical seam losses."""
        p1 = torch.randn(2, 9, 4, 4, 4)
        p2 = p1.clone()

        l1 = self.loss_fn(p1)
        l2 = self.loss_fn(p2)
        self.assertEqual(l1.item(), l2.item())


class TestSeamCorrectionTrainability(unittest.TestCase):
    """Test suite for gradient propagation and seam loss optimization using toy network."""

    def test_gradient_flow_to_toy_network(self):
        """Verify loss backward generates non-null, finite gradients on all correction network parameters."""
        torch.manual_seed(42)
        net = PatchSeamCorrectionNet(num_patches=9, channels=4, hidden_dim=32)
        loss_fn = SeamConsistencyLoss()

        x = torch.randn(2, 9, 4, 4, 4)
        out = net(x)
        loss = loss_fn(out)
        loss.backward()

        for name, param in net.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient")
            self.assertGreater(torch.norm(param.grad).item(), 1e-6, f"Parameter {name} has zero gradient")

    def test_training_reduces_seam_loss(self):
        """Verify a few gradient descent steps strictly reduce seam loss on mismatched patches."""
        torch.manual_seed(42)
        net = PatchSeamCorrectionNet(num_patches=9, channels=4, hidden_dim=32)
        loss_fn = SeamConsistencyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        x = torch.randn(4, 9, 4, 4, 4)

        with torch.no_grad():
            initial_loss = loss_fn(net(x)).item()

        for _ in range(5):
            optimizer.zero_grad()
            out = net(x)
            loss = loss_fn(out)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_loss = loss_fn(net(x)).item()

        self.assertLess(final_loss, initial_loss, "Optimization must reduce seam loss")


class TestToySeamExperimentDatasetAndConfig(unittest.TestCase):
    """Test suite for configuration parsing and dataset generation."""

    def test_parse_yaml_config(self):
        """Verify configs/module3_seam_loss.yaml parses correctly with locked parameters."""
        config_path = PROJECT_ROOT / "configs" / "module3_seam_loss.yaml"
        self.assertTrue(config_path.exists(), f"Config file not found at {config_path}")

        config = parse_yaml_config(config_path)
        self.assertIn("dataset", config)
        self.assertIn("model", config)
        self.assertIn("training", config)

        self.assertEqual(config["training"]["epochs"], 15)
        self.assertEqual(config["training"]["batch_size"], 32)
        self.assertEqual(config["training"]["learning_rate"], 0.001)
        self.assertEqual(config["dataset"]["noise_std"], 0.25)
        self.assertEqual(config["dataset"]["train_samples"], 512)
        self.assertEqual(config["dataset"]["val_samples"], 128)


if __name__ == "__main__":
    unittest.main()
