"""
Unit tests for Module 3 Milestone 3: Classical Stitching CNN.

Verifies:
1. OverlapAverageStitcher Geometry & Mathematical Identity:
   - Coverage count matrix values {1.0, 2.0, 4.0} across corners, edges, center.
   - Total count sum equals 144 = 9 patches x 4 x 4 elements.
   - Exact Overlap Reconstruction Identity: clean patches extracted from clean latents
     reconstruct the latent grid with zero error (max abs diff = 0.0).
   - Linearity of the deterministic stitcher operator.
   - Batched and unbatched tensor handling.
   - Strictly zero trainable parameters.
   - Robust input validation and error rejections (NaN, Inf, wrong dimensions).

2. ResidualRefinementCNN:
   - Architecture: 4 -> 32 -> 32 -> 4 residual conv block.
   - Parameter count: strictly 11,588 trainable parameters.
   - Input and output shapes [B, 4, 8, 8].
   - Gradient flow to all weight and bias parameters.
   - Input validation and finiteness checks.

3. ClassicalStitchingCNN Pipeline:
   - End-to-end forward pass: [B, 9, 4, 4, 4] -> [B, 4, 8, 8].
   - Parameter separation: 11,588 model trainable parameters, 0 stitcher parameters.
   - forward_with_intermediates returns overlap_average and refined tensors.
   - stitch() and refine() standalone methods.
   - Determinism and reproducibility.

4. Toy Stitching Dataset & Configuration:
   - Config parsing from configs/module3_toy_stitching.yaml.
   - Dataset tensor shapes and finite values.
   - Difference between exact overlap reconstruction and toy patch-noise refinement.
   - Multi-step training reproducibility with identical seeds.
"""

import sys
import math
from pathlib import Path
import unittest
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patch_extractor import extract_latent_patches
from src.models.stitching_cnn import (
    compute_overlap_weight_counts,
    OverlapAverageStitcher,
    ResidualRefinementCNN,
    ClassicalStitchingCNN,
)
from experiments.module3.toy_stitching_cnn import (
    generate_synthetic_latents,
    ToyStitchingDataset,
    get_toy_stitching_dataloaders,
    parse_yaml_config,
)


class TestOverlapAverageStitcherGeometry(unittest.TestCase):
    """Test suite for deterministic 3x3 overlap averaging geometry and mathematical identities."""

    def setUp(self):
        self.stitcher = OverlapAverageStitcher(grid_size=8, patch_size=4, stride=2, channels=4)

    def test_weight_counts_grid_values_and_sum(self):
        """Verify the overlap count matrix geometry, cell values in {1, 2, 4}, and total sum 144."""
        counts = self.stitcher.weight_counts
        self.assertEqual(counts.shape, torch.Size([1, 1, 8, 8]))

        counts_2d = counts.squeeze()
        unique_vals = torch.unique(counts_2d).tolist()
        self.assertEqual(sorted(unique_vals), [1.0, 2.0, 4.0])

        # Corners: count = 1.0 (2x2 at 4 corners = 16 cells)
        for r in [0, 1, 6, 7]:
            for c in [0, 1, 6, 7]:
                self.assertEqual(counts_2d[r, c].item(), 1.0, f"Corner ({r}, {c}) count != 1.0")

        # Central 4x4: count = 4.0 (rows 2..5, cols 2..5 = 16 cells)
        for r in range(2, 6):
            for c in range(2, 6):
                self.assertEqual(counts_2d[r, c].item(), 4.0, f"Center ({r}, {c}) count != 4.0")

        # Edge overlaps: count = 2.0 (remaining 32 cells)
        # Total sum must equal 16*1 + 32*2 + 16*4 = 144
        total_sum = counts_2d.sum().item()
        self.assertEqual(total_sum, 144.0)

    def test_exact_overlap_reconstruction_identity(self):
        """Verify clean patches extracted from clean latents reconstruct the latent grid bit-identically."""
        # Test across multiple batch sizes and random seeds
        for b in [1, 2, 4]:
            z_clean = torch.randn(b, 4, 8, 8)
            patches = extract_latent_patches(z_clean, patch_size=4, stride=2)
            z_reconstructed = self.stitcher(patches)

            self.assertEqual(z_reconstructed.shape, z_clean.shape)
            diff = (z_clean - z_reconstructed).abs().max().item()
            self.assertAlmostEqual(
                diff, 0.0, places=6,
                msg=f"Extract followed by stitch must reconstruct bit-identically; max abs diff: {diff}",
            )

    def test_stitcher_linearity(self):
        """Verify stitcher is a strictly linear operator: stitch(a*P1 + b*P2) == a*stitch(P1) + b*stitch(P2)."""
        p1 = torch.randn(2, 9, 4, 4, 4)
        p2 = torch.randn(2, 9, 4, 4, 4)
        alpha, beta = 2.5, -1.3

        out_linear = self.stitcher(alpha * p1 + beta * p2)
        out_expected = alpha * self.stitcher(p1) + beta * self.stitcher(p2)

        self.assertTrue(torch.allclose(out_linear, out_expected, atol=1e-6))

    def test_unbatched_and_batched_tensor_support(self):
        """Verify stitcher accepts both 4D unbatched [9, 4, 4, 4] and 5D batched [B, 9, 4, 4, 4]."""
        p_unbatched = torch.randn(9, 4, 4, 4)
        out_unbatched = self.stitcher(p_unbatched)
        self.assertEqual(out_unbatched.shape, torch.Size([4, 8, 8]))

        p_batched = torch.randn(3, 9, 4, 4, 4)
        out_batched = self.stitcher(p_batched)
        self.assertEqual(out_batched.shape, torch.Size([3, 4, 8, 8]))

    def test_stitcher_has_zero_trainable_parameters(self):
        """Verify deterministic stitcher operator has strictly zero trainable parameters."""
        self.assertEqual(self.stitcher.num_parameters, 0)
        param_count = sum(p.numel() for p in self.stitcher.parameters())
        self.assertEqual(param_count, 0)

    def test_stitcher_input_validation_errors(self):
        """Verify stitcher rejects non-tensors, invalid ranks, wrong patch/channel counts, and NaNs."""
        # Non-tensor
        with self.assertRaises(TypeError):
            self.stitcher([[0.0] * 4] * 9)

        # 3D tensor
        with self.assertRaises(ValueError):
            self.stitcher(torch.randn(9, 4, 4))

        # 6D tensor
        with self.assertRaises(ValueError):
            self.stitcher(torch.randn(1, 1, 9, 4, 4, 4))

        # Wrong patch count (8 instead of 9)
        with self.assertRaises(ValueError):
            self.stitcher(torch.randn(2, 8, 4, 4, 4))

        # Wrong channel count (3 instead of 4)
        with self.assertRaises(ValueError):
            self.stitcher(torch.randn(2, 9, 3, 4, 4))

        # Wrong patch spatial size (5x5 instead of 4x4)
        with self.assertRaises(ValueError):
            self.stitcher(torch.randn(2, 9, 4, 5, 5))

        # Non-finite inputs
        nan_patches = torch.randn(2, 9, 4, 4, 4)
        nan_patches[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.stitcher(nan_patches)

        inf_patches = torch.randn(2, 9, 4, 4, 4)
        inf_patches[0, 0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            self.stitcher(inf_patches)


class TestResidualRefinementCNN(unittest.TestCase):
    """Test suite for 4 -> 32 -> 32 -> 4 residual convolutional network."""

    def setUp(self):
        torch.manual_seed(42)
        self.cnn = ResidualRefinementCNN(in_channels=4, hidden_channels=32, out_channels=4, kernel_size=3)

    def test_parameter_count_is_strictly_11588(self):
        """Verify refinement network has exactly 11,588 trainable parameters."""
        # Conv1: 4*32*3*3 + 32 = 1184
        # Conv2: 32*32*3*3 + 32 = 9248
        # Conv3: 32*4*3*3 + 4 = 1156
        # Total = 1184 + 9248 + 1156 = 11588
        self.assertEqual(self.cnn.num_parameters, 11588)
        trainable = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
        self.assertEqual(trainable, 11588)

    def test_forward_output_shapes(self):
        """Verify forward output shape matches input shape [B, 4, 8, 8] and [4, 8, 8]."""
        for b in [1, 2, 4]:
            x = torch.randn(b, 4, 8, 8)
            out = self.cnn(x)
            self.assertEqual(out.shape, torch.Size([b, 4, 8, 8]))
            self.assertEqual(out.dtype, torch.float32)

        # Unbatched [4, 8, 8]
        x_unb = torch.randn(4, 8, 8)
        out_unb = self.cnn(x_unb)
        self.assertEqual(out_unb.shape, torch.Size([4, 8, 8]))

    def test_gradient_flow_to_all_parameters(self):
        """Verify loss backward computes non-null, finite, non-zero gradients on all CNN parameters."""
        x = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)

        out = self.cnn(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()

        for name, param in self.cnn.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients")
            self.assertGreater(torch.norm(param.grad).item(), 1e-6, f"Parameter {name} has zero gradient")

    def test_input_validation_errors(self):
        """Verify CNN rejects invalid input types, channels, and non-finite values."""
        with self.assertRaises(TypeError):
            self.cnn([[0.0] * 8] * 8)

        with self.assertRaises(ValueError):
            self.cnn(torch.randn(2, 3, 8, 8))  # 3 channels instead of 4

        with self.assertRaises(ValueError):
            self.cnn(torch.randn(2, 4, 8))  # 3D rank with wrong channel layout

        nan_x = torch.randn(2, 4, 8, 8)
        nan_x[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.cnn(nan_x)


class TestClassicalStitchingCNN(unittest.TestCase):
    """Test suite for integrated ClassicalStitchingCNN model."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = ClassicalStitchingCNN(grid_size=8, patch_size=4, stride=2, channels=4, hidden_channels=32)

    def test_parameter_counts(self):
        """Verify model total trainable parameters is 11,588 and stitcher parameters is 0."""
        self.assertEqual(self.model.num_trainable_parameters, 11588)
        self.assertEqual(self.model.num_stitcher_parameters, 0)

    def test_forward_pipeline(self):
        """Verify end-to-end forward [B, 9, 4, 4, 4] -> [B, 4, 8, 8]."""
        for b in [1, 2, 4]:
            patches = torch.randn(b, 9, 4, 4, 4)
            out = self.model(patches)
            self.assertEqual(out.shape, torch.Size([b, 4, 8, 8]))
            self.assertTrue(torch.isfinite(out).all())

    def test_forward_with_intermediates(self):
        """Verify forward_with_intermediates returns overlap_average and refined tensors."""
        patches = torch.randn(2, 9, 4, 4, 4)
        intermediates = self.model.forward_with_intermediates(patches)

        self.assertIn("overlap_average", intermediates)
        self.assertIn("refined", intermediates)
        self.assertEqual(intermediates["overlap_average"].shape, torch.Size([2, 4, 8, 8]))
        self.assertEqual(intermediates["refined"].shape, torch.Size([2, 4, 8, 8]))

        # Compare with standalone stitch and forward
        z_overlap_expected = self.model.stitch(patches)
        self.assertTrue(torch.equal(intermediates["overlap_average"], z_overlap_expected))

        refined_expected = self.model.forward(patches)
        self.assertTrue(torch.equal(intermediates["refined"], refined_expected))

    def test_refine_standalone(self):
        """Verify refine() applies CNN to a pre-stitched [B, 4, 8, 8] tensor."""
        z_grid = torch.randn(2, 4, 8, 8)
        refined = self.model.refine(z_grid)
        self.assertEqual(refined.shape, torch.Size([2, 4, 8, 8]))


class TestToyStitchingDatasetAndConfig(unittest.TestCase):
    """Test suite for toy dataset generation, config parsing, and short-step reproducibility."""

    def test_parse_yaml_config(self):
        """Verify configs/module3_toy_stitching.yaml parses correctly with locked parameters."""
        config_path = PROJECT_ROOT / "configs" / "module3_toy_stitching.yaml"
        self.assertTrue(config_path.exists(), f"Config file not found at {config_path}")

        config = parse_yaml_config(config_path)
        self.assertIn("dataset", config)
        self.assertIn("model", config)
        self.assertIn("training", config)

        # Locked parameters per approved plan
        self.assertEqual(config["training"]["epochs"], 15)
        self.assertEqual(config["training"]["batch_size"], 32)
        self.assertEqual(config["training"]["learning_rate"], 0.001)
        self.assertEqual(config["training"]["optimizer"], "adam")
        self.assertEqual(config["training"]["loss"], "mse")
        self.assertEqual(config["dataset"]["noise_std"], 0.25)
        self.assertEqual(config["dataset"]["train_samples"], 512)
        self.assertEqual(config["dataset"]["val_samples"], 128)
        self.assertEqual(config["model"]["hidden_channels"], 32)

    def test_synthetic_latents_generation(self):
        """Verify synthetic latent generator returns smooth finite [N, 4, 8, 8] tensors."""
        latents = generate_synthetic_latents(num_samples=16, channels=4, height=8, width=8, seed=42)
        self.assertEqual(latents.shape, torch.Size([16, 4, 8, 8]))
        self.assertTrue(torch.isfinite(latents).all())

        # Determinism
        latents_2 = generate_synthetic_latents(num_samples=16, channels=4, height=8, width=8, seed=42)
        self.assertTrue(torch.equal(latents, latents_2))

    def test_dataset_item_shapes_and_perturbation(self):
        """Verify dataset returns (noisy_patches [9, 4, 4, 4], clean_latents [4, 8, 8])."""
        dataset = ToyStitchingDataset(num_samples=16, seed=42)
        self.assertEqual(len(dataset), 16)

        noisy_p, clean_z = dataset[0]
        self.assertEqual(noisy_p.shape, torch.Size([9, 4, 4, 4]))
        self.assertEqual(clean_z.shape, torch.Size([4, 8, 8]))
        self.assertTrue(torch.isfinite(noisy_p).all())
        self.assertTrue(torch.isfinite(clean_z).all())

        # Clean patches from clean latents have 0 overlap MSE
        stitcher = OverlapAverageStitcher()
        clean_reconstructed = stitcher(dataset.clean_patches[0:1])
        clean_mse = nn.MSELoss()(clean_reconstructed, dataset.clean_latents[0:1]).item()
        self.assertAlmostEqual(clean_mse, 0.0, places=6, msg="Clean patches must have 0.0 overlap MSE")

        # Perturbed patches have strictly positive overlap MSE
        noisy_reconstructed = stitcher(dataset.noisy_patches[0:1])
        noisy_mse = nn.MSELoss()(noisy_reconstructed, dataset.clean_latents[0:1]).item()
        self.assertGreater(noisy_mse, 0.01, msg="Perturbed patches must have strictly positive overlap MSE")

    def test_dataloaders_batch_size_and_count(self):
        """Verify train and val dataloaders return batches of shape [32, 9, 4, 4, 4]."""
        train_loader, val_loader = get_toy_stitching_dataloaders(
            train_samples=64,
            val_samples=32,
            batch_size=32,
            train_seed=42,
            val_seed=1042,
        )
        self.assertEqual(len(train_loader), 2)
        self.assertEqual(len(val_loader), 1)

        for noisy_b, clean_b in train_loader:
            self.assertEqual(noisy_b.shape, torch.Size([32, 9, 4, 4, 4]))
            self.assertEqual(clean_b.shape, torch.Size([32, 4, 8, 8]))
            break

    def test_short_run_reproducibility(self):
        """Verify two 2-step training runs with identical seeds produce bit-identical losses and weights."""
        def run_short_steps(seed: int):
            torch.manual_seed(seed)
            model = ClassicalStitchingCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            gen = torch.Generator().manual_seed(seed)
            noisy = torch.randn(2, 9, 4, 4, 4, generator=gen)
            clean = torch.randn(2, 4, 8, 8, generator=gen)

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

        self.assertEqual(losses_1, losses_2, "Identical seeds must yield identical losses")
        for p1, p2 in zip(params_1, params_2):
            self.assertTrue(torch.equal(p1, p2), "Model parameters must match bit-identically after identical steps")


if __name__ == "__main__":
    unittest.main()
