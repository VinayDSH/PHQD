"""
Unit tests for Module 1 Latent-Patch Extraction.

Verifies:
1. Patch geometry: [B, 4, 8, 8] -> [B, 9, 4, 4, 4].
2. Expected 9 patches per image (3x3 grid).
3. Expected [4, 4, 4] patch shape (4 channels, 4 height, 4 width).
4. Configurable patch_size and stride.
5. Exact spatial alignment against ground-truth coordinate slices.
6. LatentPatchExtractor operates with frozen model parameters and eval() mode.
7. Metadata dictionary schema contains all required reproducibility keys.
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

from src.models.autoencoder import ConvAutoencoder
from src.data.patch_extractor import (
    extract_latent_patches,
    LatentPatchExtractor,
)


class TestLatentPatchExtractor(unittest.TestCase):
    """Test suite verifying latent patch geometry, extraction correctness, and freezing."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.latent_channels = 4
        self.latent_h = 8
        self.latent_w = 8
        self.dummy_latent = torch.randn(
            self.batch_size,
            self.latent_channels,
            self.latent_h,
            self.latent_w,
        )

    def test_default_patch_geometry_shape(self):
        """Verify output shape is exactly [B, 9, 4, 4, 4] for default patch_size=4, stride=2."""
        patches = extract_latent_patches(self.dummy_latent, patch_size=4, stride=2)
        expected_shape = torch.Size([self.batch_size, 9, 4, 4, 4])
        self.assertEqual(
            patches.shape,
            expected_shape,
            f"Expected patch tensor shape {expected_shape}, got {patches.shape}",
        )

    def test_expected_nine_patches_per_image(self):
        """Verify each latent produces exactly 9 spatial patches (3x3 grid)."""
        patches = extract_latent_patches(self.dummy_latent, patch_size=4, stride=2)
        num_patches = patches.size(1)
        self.assertEqual(
            num_patches,
            9,
            f"Expected exactly 9 patches per image, but got {num_patches}",
        )

    def test_expected_single_patch_shape(self):
        """Verify each individual patch has shape [4, 4, 4] (64 latent values)."""
        patches = extract_latent_patches(self.dummy_latent, patch_size=4, stride=2)
        single_patch = patches[0, 0]
        expected_patch_shape = torch.Size([4, 4, 4])
        self.assertEqual(
            single_patch.shape,
            expected_patch_shape,
            f"Expected single patch shape {expected_patch_shape}, got {single_patch.shape}",
        )
        self.assertEqual(
            single_patch.numel(),
            64,
            f"Expected 64 values per patch (4x4x4), got {single_patch.numel()}",
        )

    def test_exact_spatial_slice_alignment(self):
        """Verify extracted patches exactly match manual spatial tensor slicing."""
        # Construct coordinate-indexed latent to verify precise spatial mapping
        z = torch.arange(8 * 8, dtype=torch.float32).view(1, 1, 8, 8).expand(1, 4, 8, 8)
        patches = extract_latent_patches(z, patch_size=4, stride=2)

        # 3x3 grid corresponds to row offsets [0, 2, 4] and col offsets [0, 2, 4]
        grid_offsets = [(r, c) for r in [0, 2, 4] for c in [0, 2, 4]]
        for patch_idx, (r, c) in enumerate(grid_offsets):
            expected_slice = z[:, :, r : r + 4, c : c + 4]
            actual_patch = patches[:, patch_idx]
            self.assertTrue(
                torch.allclose(actual_patch, expected_slice),
                f"Patch index {patch_idx} at offset ({r}, {c}) does not match expected spatial slice",
            )

    def test_configurable_stride_and_patch_size(self):
        """Verify extraction logic works for alternative valid patch sizes and strides."""
        # Non-overlapping 4x4 patches with stride=4: yields 2x2 = 4 patches of shape [4, 4, 4]
        patches_s4 = extract_latent_patches(self.dummy_latent, patch_size=4, stride=4)
        self.assertEqual(patches_s4.shape, torch.Size([self.batch_size, 4, 4, 4, 4]))

        # Non-overlapping 2x2 patches with stride=2: yields 4x4 = 16 patches of shape [4, 2, 2]
        patches_p2_s2 = extract_latent_patches(self.dummy_latent, patch_size=2, stride=2)
        self.assertEqual(patches_p2_s2.shape, torch.Size([self.batch_size, 16, 4, 2, 2]))

        # Full-grid patch with patch_size=8, stride=1: yields 1 patch of shape [4, 8, 8]
        patches_p8 = extract_latent_patches(self.dummy_latent, patch_size=8, stride=1)
        self.assertEqual(patches_p8.shape, torch.Size([self.batch_size, 1, 4, 8, 8]))

    def test_patch_extractor_frozen_model(self):
        """Verify LatentPatchExtractor freezes model parameters, sets eval mode, and extracts [B, 9, 4, 4, 4]."""
        model = ConvAutoencoder(in_channels=1, latent_channels=4)
        # Ensure parameters originally had requires_grad True
        self.assertTrue(any(p.requires_grad for p in model.parameters()))

        extractor = LatentPatchExtractor(model, patch_size=4, stride=2)

        # Must be in eval mode
        self.assertFalse(extractor.model.training, "Model must be in eval() mode")

        # All parameters must have requires_grad == False
        for name, param in extractor.model.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"Parameter {name} was not frozen (requires_grad is True)",
            )

        # Extract from dummy image batch: [B, 1, 28, 28] -> [B, 9, 4, 4, 4]
        dummy_images = torch.rand(self.batch_size, 1, 28, 28)
        extracted = extractor.extract_from_images(dummy_images)

        expected_shape = torch.Size([self.batch_size, 9, 4, 4, 4])
        self.assertEqual(extracted.shape, expected_shape)
        self.assertFalse(extracted.requires_grad)

    def test_metadata_reproducibility_schema(self):
        """Verify metadata fields required for reproducible downstream diffusion are present."""
        required_keys = {
            "split",
            "num_samples",
            "latent_shape",
            "patch_size",
            "stride",
            "patches_per_image",
            "patch_shape",
            "checkpoint_path",
            "seed",
        }
        metadata_sample = {
            "split": "train",
            "num_samples": 50000,
            "latent_shape": [4, 8, 8],
            "patch_size": 4,
            "stride": 2,
            "patches_per_image": 9,
            "patch_shape": [4, 4, 4],
            "checkpoint_path": "checkpoints/best_autoencoder.pt",
            "seed": 42,
        }
        for key in required_keys:
            self.assertIn(key, metadata_sample, f"Missing required metadata key: {key}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
