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
    build_parser,
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
            self.assertEqual(summary["pqc_results"]["best_val_mse"], summary["pqc_results"]["final_val_loss"])
            self.assertEqual(summary["pqc_results"]["best_epoch"], 1)
            self.assertEqual(summary["classical_results"]["best_val_mse"], summary["classical_results"]["final_val_loss"])
            self.assertEqual(summary["classical_results"]["best_epoch"], 1)

            # Check secondary 64-D metrics
            pqc_sec = summary["pqc_results"]["secondary_metrics"]["val_recon_mse_64d"]
            class_sec = summary["classical_results"]["secondary_metrics"]["val_recon_mse_64d"]
            self.assertTrue(torch.isfinite(torch.tensor(pqc_sec)))
            self.assertTrue(torch.isfinite(torch.tensor(class_sec)))

    def test_validation_mse_tracking_and_best_epoch(self):
        """Verify validation MSE is tracked per epoch, best_val_mse equals min(history),
        and best_epoch corresponds to that minimum for both PQC and classical models."""
        epochs = 3
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_latent_patch_denoising_experiment(
                train_patches=self.train_patches,
                val_patches=self.val_patches,
                epochs=epochs,
                batch_size=16,
                compare_classical=True,
                output_dir=tmp_dir,
            )

            for model_key in ["pqc_results", "classical_results"]:
                res = summary[model_key]

                val_history = res["history"]["val_loss"]
                train_history = res["history"]["train_loss"]

                # 1. Per-epoch history tracked for all epochs
                self.assertEqual(len(val_history), epochs)
                self.assertEqual(len(train_history), epochs)

                for val_mse in val_history:
                    self.assertTrue(torch.isfinite(torch.tensor(val_mse)))
                for train_mse in train_history:
                    self.assertTrue(torch.isfinite(torch.tensor(train_mse)))

                # 2. best_val_mse equals min(val_history)
                expected_best_val_mse = min(val_history)
                self.assertEqual(res["best_val_mse"], expected_best_val_mse)

                # 3. best_epoch corresponds to that minimum (1-indexed)
                expected_best_epoch = val_history.index(expected_best_val_mse) + 1
                self.assertEqual(res["best_epoch"], expected_best_epoch)
                self.assertGreaterEqual(res["best_epoch"], 1)
                self.assertLessEqual(res["best_epoch"], epochs)

                # 4. final_val_mse equals last element in history
                self.assertEqual(res["final_val_mse"], val_history[-1])
                self.assertEqual(res["final_train_mse"], train_history[-1])

            # 5. Verify saved JSON summary contains the required keys
            log_file = Path(tmp_dir) / "latent_patch_pqc_denoising_results.json"
            self.assertTrue(log_file.exists())
            with open(log_file, "r") as f:
                saved_runs = json.load(f)
            saved_pqc = saved_runs[0]["pqc_results"]
            saved_cls = saved_runs[0]["classical_results"]

            self.assertIn("best_val_mse", saved_pqc)
            self.assertIn("best_epoch", saved_pqc)
            self.assertIn("final_val_mse", saved_pqc)
            self.assertIn("final_train_mse", saved_pqc)

            self.assertIn("best_val_mse", saved_cls)
            self.assertIn("best_epoch", saved_cls)
            self.assertIn("final_val_mse", saved_cls)
            self.assertIn("final_train_mse", saved_cls)

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

    def test_cli_parser_defaults_and_explicit_limits(self):
        """Verify CLI argument parser defaults to None and accepts integer limits."""
        parser = build_parser()

        # Defaults must be None
        args_default = parser.parse_args([])
        self.assertIsNone(args_default.max_train_samples)
        self.assertIsNone(args_default.max_val_samples)

        # Explicit integer limits
        args_explicit = parser.parse_args([
            "--max-train-samples", "100",
            "--max-val-samples", "25",
        ])
        self.assertEqual(args_explicit.max_train_samples, 100)
        self.assertEqual(args_explicit.max_val_samples, 25)

    def test_max_samples_limits_respected(self):
        """Verify max_train_samples and max_val_samples restrict image counts correctly."""
        num_train_images = 6
        num_val_images = 4
        train_patches = torch.randn(num_train_images, 9, 4, 4, 4)
        val_patches = torch.randn(num_val_images, 9, 4, 4, 4)

        limit_train = 3
        limit_val = 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_latent_patch_denoising_experiment(
                train_patches=train_patches,
                val_patches=val_patches,
                epochs=1,
                batch_size=16,
                max_train_samples=limit_train,
                max_val_samples=limit_val,
                compare_classical=False,
                output_dir=tmp_dir,
            )

            self.assertEqual(summary["max_train_samples"], limit_train)
            self.assertEqual(summary["max_val_samples"], limit_val)
            self.assertEqual(summary["num_train_images"], limit_train)
            self.assertEqual(summary["num_val_images"], limit_val)
            self.assertEqual(summary["num_train_patches"], limit_train * 9)
            self.assertEqual(summary["num_val_patches"], limit_val * 9)

    def test_default_none_preserves_full_dataset(self):
        """Verify default None preserves all images and patches from full dataset."""
        num_train_images = 5
        num_val_images = 3
        train_patches = torch.randn(num_train_images, 9, 4, 4, 4)
        val_patches = torch.randn(num_val_images, 9, 4, 4, 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_latent_patch_denoising_experiment(
                train_patches=train_patches,
                val_patches=val_patches,
                epochs=1,
                batch_size=16,
                max_train_samples=None,
                max_val_samples=None,
                compare_classical=False,
                output_dir=tmp_dir,
            )

            self.assertIsNone(summary["max_train_samples"])
            self.assertIsNone(summary["max_val_samples"])
            self.assertEqual(summary["num_train_images"], num_train_images)
            self.assertEqual(summary["num_val_images"], num_val_images)
            self.assertEqual(summary["num_train_patches"], num_train_images * 9)
            self.assertEqual(summary["num_val_patches"], num_val_images * 9)

    def test_max_samples_with_saved_patch_files(self):
        """Verify that loading from disk files also respects sample limits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_file = Path(tmp_dir) / "train.pt"
            val_file = Path(tmp_dir) / "val.pt"

            torch.save({"patches": torch.randn(8, 9, 4, 4, 4)}, train_file)
            torch.save({"patches": torch.randn(4, 9, 4, 4, 4)}, val_file)

            summary = run_latent_patch_denoising_experiment(
                train_patches_path=str(train_file),
                val_patches_path=str(val_file),
                epochs=1,
                batch_size=16,
                max_train_samples=2,
                max_val_samples=1,
                compare_classical=False,
                output_dir=tmp_dir,
            )

            self.assertEqual(summary["max_train_samples"], 2)
            self.assertEqual(summary["max_val_samples"], 1)
            self.assertEqual(summary["num_train_images"], 2)
            self.assertEqual(summary["num_val_images"], 1)
            self.assertEqual(summary["num_train_patches"], 18)
            self.assertEqual(summary["num_val_patches"], 9)


if __name__ == "__main__":
    unittest.main()
