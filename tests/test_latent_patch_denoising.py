"""
Unit tests for Module 2 Stage 2: Latent-Patch Denoising Dataset & Preprocessing Pipeline.

Verifies:
1. Dataset shapes for sample [6], batch [B, 6], and patch flattening [B, 9, 4, 4, 4] -> B*9.
2. Deterministic noise generation across identical vs different seeds.
3. Training-only normalization fitting (zero leakage from validation/test data).
4. Finite values across clean and noisy vectors.
5. Clean/noisy pairing and residual noise distribution (mean ~ 0, std ~ 0.2).
6. Zero trainable parameters in preprocessing and dataset components.
7. Explicit documentation and test check confirming lossy 64->6 compression:
   - Arbitrary 64-D patches cannot be exactly reconstructed via pseudo-inverse.
   - Primary metric for Stage 2 is 6-D projected denoising MSE.
"""

import tempfile
from pathlib import Path
import unittest
import torch
from torch.utils.data import DataLoader

from src.models.projection import FixedPatchProjector
from src.data.latent_patch_denoising import (
    LatentPatchDenoisingDataset,
    prepare_latent_patch_denoising_datasets,
    get_latent_patch_denoising_dataloaders,
    flatten_latent_patches,
)


class TestLatentPatchDenoising(unittest.TestCase):
    """Test suite for Module 2 Stage 2 dataset and preprocessing."""

    def setUp(self):
        torch.manual_seed(42)
        self.num_images = 10
        self.patches_per_image = 9
        self.total_patches = self.num_images * self.patches_per_image
        self.noise_std = 0.2

        # Synthetic latent patch tensors matching Module 1 export shape [B, 9, 4, 4, 4]
        self.train_patches = torch.randn(self.num_images, 9, 4, 4, 4) + 1.5
        self.val_patches = torch.randn(5, 9, 4, 4, 4) - 3.0

        self.projector = FixedPatchProjector(seed=42, norm_type="standard")
        train_flat = flatten_latent_patches(self.train_patches)
        self.projector.fit(train_flat)

    def test_flatten_latent_patches_shapes(self):
        """Verify [B, 9, 4, 4, 4] flattens to [B * 9, 64]."""
        flat = flatten_latent_patches(self.train_patches)
        self.assertEqual(flat.shape, torch.Size([self.total_patches, 64]))

    def test_dataset_item_and_batch_shapes(self):
        """Verify sample shapes (6,) and DataLoader batch shapes [B, 6]."""
        dataset = LatentPatchDenoisingDataset(
            patches=self.train_patches,
            projector=self.projector,
            noise_std=self.noise_std,
            seed=42,
        )
        self.assertEqual(len(dataset), self.total_patches)

        noisy, clean = dataset[0]
        self.assertEqual(noisy.shape, torch.Size([6]))
        self.assertEqual(clean.shape, torch.Size([6]))

        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        batch_noisy, batch_clean = next(iter(loader))
        self.assertEqual(batch_noisy.shape, torch.Size([16, 6]))
        self.assertEqual(batch_clean.shape, torch.Size([16, 6]))

    def test_deterministic_noise_generation(self):
        """Verify identical seed produces identical noisy vectors; different seed produces different noise."""
        d1 = LatentPatchDenoisingDataset(self.train_patches, self.projector, noise_std=0.2, seed=42)
        d2 = LatentPatchDenoisingDataset(self.train_patches, self.projector, noise_std=0.2, seed=42)
        d3 = LatentPatchDenoisingDataset(self.train_patches, self.projector, noise_std=0.2, seed=999)

        # Same seed must yield identical clean and noisy tensors
        self.assertTrue(torch.equal(d1.clean, d2.clean))
        self.assertTrue(torch.equal(d1.noisy, d2.noisy))

        # Different seed must have identical clean targets but different noisy inputs
        self.assertTrue(torch.equal(d1.clean, d3.clean))
        self.assertFalse(torch.equal(d1.noisy, d3.noisy))

    def test_training_only_normalization_fitting(self):
        """Verify normalization statistics are fitted strictly on training data."""
        train_ds, val_ds, frozen_proj = prepare_latent_patch_denoising_datasets(
            train_patches=self.train_patches,
            val_patches=self.val_patches,
            noise_std=self.noise_std,
            projector_seed=42,
            dataset_seed=42,
        )

        train_flat = flatten_latent_patches(self.train_patches)
        expected_mean_64 = train_flat.mean(dim=0)

        # Stored mean_64 must match training set exactly
        self.assertTrue(torch.allclose(frozen_proj.mean_64, expected_mean_64, atol=1e-5))

        # Train and validation datasets must reference the exact same frozen projector
        self.assertIs(train_ds.projector, frozen_proj)
        self.assertIs(val_ds.projector, frozen_proj)

        # Varying validation patches must NOT alter frozen training statistics
        other_val = torch.randn(5, 9, 4, 4, 4) * 10.0
        _, _, frozen_proj_2 = prepare_latent_patch_denoising_datasets(
            train_patches=self.train_patches,
            val_patches=other_val,
            projector_seed=42,
            dataset_seed=42,
        )
        self.assertTrue(torch.equal(frozen_proj.mean_64, frozen_proj_2.mean_64))
        self.assertTrue(torch.equal(frozen_proj.mean_6, frozen_proj_2.mean_6))
        self.assertTrue(torch.equal(frozen_proj.std_6, frozen_proj_2.std_6))

    def test_finite_values(self):
        """Verify all elements in clean and noisy vectors are finite (no NaN, no Inf)."""
        dataset = LatentPatchDenoisingDataset(
            patches=self.train_patches,
            projector=self.projector,
            noise_std=self.noise_std,
            seed=42,
        )
        self.assertTrue(torch.isfinite(dataset.clean).all())
        self.assertTrue(torch.isfinite(dataset.noisy).all())

    def test_clean_noisy_pairing_and_noise_distribution(self):
        """Verify clean/noisy pairing and that residual follows N(0, 0.2^2)."""
        large_train = torch.randn(100, 9, 4, 4, 4)
        large_proj = FixedPatchProjector(seed=42).fit(flatten_latent_patches(large_train))
        dataset = LatentPatchDenoisingDataset(
            patches=large_train,
            projector=large_proj,
            noise_std=0.2,
            seed=42,
        )

        # Check pairing: clean target must match projector output on corresponding patch
        flat_patches = flatten_latent_patches(large_train)
        sample_idx = 17
        expected_clean = large_proj(flat_patches[sample_idx])
        self.assertTrue(torch.allclose(dataset.clean[sample_idx], expected_clean, atol=1e-5))

        # Check noise statistics across all samples
        residuals = dataset.noisy - dataset.clean
        residual_mean = residuals.mean().item()
        residual_std = residuals.std().item()

        self.assertAlmostEqual(residual_mean, 0.0, delta=0.03)
        self.assertAlmostEqual(residual_std, 0.2, delta=0.02)

    def test_zero_trainable_parameters(self):
        """Verify dataset and projector have strictly 0 trainable parameters."""
        dataset = LatentPatchDenoisingDataset(
            patches=self.train_patches,
            projector=self.projector,
            noise_std=self.noise_std,
            seed=42,
        )
        self.assertEqual(dataset.projector.num_trainable_parameters, 0)
        self.assertEqual(len(list(dataset.projector.parameters())), 0)

    def test_file_path_loading(self):
        """Verify dataset correctly loads exported .pt payload dictionaries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_patches.pt"
            payload = {
                "patches": self.train_patches,
                "labels": torch.zeros(self.num_images),
                "metadata": {"split": "train"},
            }
            torch.save(payload, file_path)

            ds_from_file = LatentPatchDenoisingDataset(
                patches=file_path,
                projector=self.projector,
                noise_std=self.noise_std,
                seed=42,
            )
            ds_from_tensor = LatentPatchDenoisingDataset(
                patches=self.train_patches,
                projector=self.projector,
                noise_std=self.noise_std,
                seed=42,
            )
            self.assertTrue(torch.equal(ds_from_file.clean, ds_from_tensor.clean))
            self.assertTrue(torch.equal(ds_from_file.noisy, ds_from_tensor.noisy))

    def test_lossy_compression_pseudo_inverse_documentation_check(self):
        """Verify that 64-D -> 6-D projection is intentionally lossy and arbitrary 64-D
        patches cannot be exactly reconstructed via pseudo-inverse back-projection."""
        flat_patches = flatten_latent_patches(self.train_patches)
        clean_6d = self.projector(flat_patches)
        recon_64d = self.projector.pseudo_inverse_back_project(clean_6d)

        # Arbitrary patches have substantial reconstruction error in 64-D
        recon_error = torch.mean((flat_patches - recon_64d) ** 2).item()
        self.assertGreater(
            recon_error,
            0.1,
            f"Expected non-zero lossy reconstruction error, got {recon_error}",
        )

        # But re-projecting into 6-D preserves exact 6-D clean coordinates
        clean_reprojected = self.projector(recon_64d)
        self.assertTrue(torch.allclose(clean_6d, clean_reprojected, atol=1e-5))

    def test_projector_buffers_unchanged_after_dataset_construction(self):
        """Snapshot projector buffers and assert they remain unchanged after dataset construction."""
        # 1. Snapshot all persistent buffers before constructing dataset
        snapshots = {
            name: buf.clone()
            for name, buf in self.projector.named_buffers()
        }

        # 2. Construct dataset with distinct out-of-distribution patches
        novel_patches = torch.randn(8, 9, 4, 4, 4) * 20.0 - 50.0
        _ = LatentPatchDenoisingDataset(
            patches=novel_patches,
            projector=self.projector,
            noise_std=self.noise_std,
            seed=1234,
        )

        # 3. Assert all buffers are strictly unchanged bit-for-bit
        for name, buf in self.projector.named_buffers():
            self.assertTrue(
                torch.equal(buf, snapshots[name]),
                f"Buffer '{name}' was modified during dataset construction",
            )


if __name__ == "__main__":
    unittest.main()

