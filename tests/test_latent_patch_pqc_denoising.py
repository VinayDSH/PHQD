"""
Focused unit tests for Module 2 Stage 2 Experiment Wiring.

Verifies:
1. End-to-end execution of run_latent_patch_denoising_experiment on synthetic patch inputs.
2. Strict parameter count parity between PQC and circulant baseline (12 vs 12).
3. Independent recording of PQC and classical initialization seeds.
4. Calculation and finiteness of primary 6-D MSE and secondary 64-D reconstruction MSE.
5. Identical and reproducible train DataLoader batch ordering for PQC and classical runs.
6. Non-destructive JSON result appending without overwriting previous runs or Stage 1 logs.
7. Error handling when latent patch files do not exist.
"""

import sys
import tempfile
import json
from pathlib import Path
import unittest
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.module2.latent_patch_pqc_denoising import (
    run_latent_patch_denoising_experiment,
    create_reproducible_train_loader,
)
from src.data.latent_patch_denoising import prepare_latent_patch_denoising_datasets


class TestStage2ExperimentWiring(unittest.TestCase):
    """Test suite verifying Module 2 Stage 2 experiment wiring."""

    def setUp(self):
        torch.manual_seed(42)
        self.num_train_images = 4
        self.num_val_images = 2
        # Synthetic Module 1 patch tensors [B, 9, 4, 4, 4]
        self.train_patches = torch.randn(self.num_train_images, 9, 4, 4, 4) + 1.0
        self.val_patches = torch.randn(self.num_val_images, 9, 4, 4, 4) - 1.0

    def test_end_to_end_wiring_synthetic_run(self):
        """Verify complete 1-epoch training flow for PQC and circulant baseline on synthetic patches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_latent_patch_denoising_experiment(
                train_patches=self.train_patches,
                val_patches=self.val_patches,
                qubits=6,
                layers=2,
                noise_std=0.2,
                epochs=1,
                batch_size=16,
                lr=0.05,
                pqc_seed=111,
                classical_seed=222,
                dataloader_seed=333,
                compare_classical=True,
                output_dir=tmp_dir,
            )

            self.assertEqual(summary["experiment"], "latent_patch_pqc_denoising")
            self.assertEqual(summary["primary_metric"], "6d_projected_denoising_mse")
            self.assertEqual(summary["dataloader_seed"], 333)
            self.assertEqual(summary["model_initialization"]["pqc_init_seed"], 111)
            self.assertEqual(summary["model_initialization"]["classical_init_seed"], 222)

            # Both models must have strictly 12 trainable parameters
            self.assertEqual(summary["pqc_results"]["parameter_count"], 12)
            self.assertEqual(summary["classical_results"]["parameter_count"], 12)

            # Check primary 6-D metrics
            self.assertTrue(torch.isfinite(torch.tensor(summary["pqc_results"]["final_val_loss"])))
            self.assertTrue(torch.isfinite(torch.tensor(summary["classical_results"]["final_val_loss"])))

            # Check secondary 64-D metrics
            pqc_sec = summary["pqc_results"]["secondary_metrics"]["val_recon_mse_64d"]
            class_sec = summary["classical_results"]["secondary_metrics"]["val_recon_mse_64d"]
            self.assertTrue(torch.isfinite(torch.tensor(pqc_sec)))
            self.assertTrue(torch.isfinite(torch.tensor(class_sec)))

    def test_train_dataloader_batch_ordering_identical_for_pqc_and_classical(self):
        """Verify that train DataLoaders for PQC and classical runs iterate over
        bit-for-bit identical batches in the exact same shuffle order."""
        datasets_and_proj = prepare_latent_patch_denoising_datasets(
            train_patches=self.train_patches,
            noise_std=0.2,
            dataset_seed=42,
        )
        train_ds = datasets_and_proj[0]

        dataloader_seed = 777
        loader_pqc = create_reproducible_train_loader(train_ds, batch_size=8, seed=dataloader_seed)
        loader_classical = create_reproducible_train_loader(train_ds, batch_size=8, seed=dataloader_seed)

        # Multi-epoch check: verify each epoch produces identical batches
        epochs = 3
        for epoch in range(epochs):
            batches_pqc = list(loader_pqc)
            batches_classical = list(loader_classical)
            self.assertEqual(len(batches_pqc), len(batches_classical))

            for (pqc_noisy, pqc_clean), (cls_noisy, cls_clean) in zip(batches_pqc, batches_classical):
                self.assertTrue(
                    torch.equal(pqc_noisy, cls_noisy),
                    f"Batch noisy mismatch at epoch {epoch}",
                )
                self.assertTrue(
                    torch.equal(pqc_clean, cls_clean),
                    f"Batch clean mismatch at epoch {epoch}",
                )

    def test_non_destructive_results_logging(self):
        """Verify subsequent experiment runs append records without overwriting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "latent_patch_pqc_denoising_results.json"

            # Run 1
            run_latent_patch_denoising_experiment(
                train_patches=self.train_patches,
                val_patches=self.val_patches,
                epochs=1,
                batch_size=16,
                compare_classical=False,
                output_dir=tmp_dir,
            )
            with open(log_file, "r") as f:
                data_run1 = json.load(f)
            self.assertEqual(len(data_run1), 1)
            self.assertEqual(data_run1[0]["run_id"], 1)

            # Run 2
            run_latent_patch_denoising_experiment(
                train_patches=self.train_patches,
                val_patches=self.val_patches,
                epochs=1,
                batch_size=16,
                compare_classical=False,
                output_dir=tmp_dir,
            )
            with open(log_file, "r") as f:
                data_run2 = json.load(f)
            self.assertEqual(len(data_run2), 2)
            self.assertEqual(data_run2[1]["run_id"], 2)

    def test_missing_patch_files_raises_filenotfound(self):
        """Verify clear FileNotFoundError when patch files do not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            run_latent_patch_denoising_experiment(
                train_patches_path="nonexistent_train.pt",
                val_patches_path="nonexistent_val.pt",
            )
        self.assertIn("Training latent patches not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
