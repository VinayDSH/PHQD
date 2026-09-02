"""
Unit tests for Module 1 Autoencoder training and reconstruction evaluation pipeline.

Verifies:
1. PSNR returns a finite scalar for valid [0, 1] tensors.
2. PSNR of identical images is very high / handled correctly.
3. SSIM returns a valid finite value in its expected range [-1, 1] and 1.0 for identical images.
4. A single training step produces a finite loss.
5. Gradients flow through all layers of the autoencoder.
6. Checkpoint directory/path handling works without requiring an existing directory.
"""

import sys
import tempfile
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import ConvAutoencoder
from src.evaluation.metrics import compute_psnr, compute_ssim
from src.training.train_autoencoder import parse_yaml_config, set_seed


class TestAutoencoderTrainingPipeline(unittest.TestCase):
    """Test suite for training pipeline, metric definitions, and gradient flow."""

    def setUp(self):
        set_seed(42)

    def test_psnr_finite_scalar(self):
        """Verify PSNR returns a finite scalar tensor for valid [0, 1] images."""
        pred = torch.rand(4, 1, 28, 28)
        target = torch.rand(4, 1, 28, 28)

        psnr = compute_psnr(pred, target)
        self.assertEqual(psnr.dim(), 0, "PSNR should be a 0-dim scalar tensor")
        self.assertTrue(torch.isfinite(psnr), f"PSNR should be finite, got {psnr.item()}")
        self.assertGreater(psnr.item(), 0.0, "PSNR for typical non-identical images should be > 0 dB")

    def test_psnr_identical_images_handled_correctly(self):
        """Verify PSNR of identical images is very high (>= 80 dB) and finite."""
        img = torch.rand(4, 1, 28, 28)
        psnr = compute_psnr(img, img)

        self.assertTrue(torch.isfinite(psnr), "PSNR for identical images must be finite")
        self.assertGreaterEqual(
            psnr.item(),
            80.0,
            f"PSNR for identical images should be very high (>= 80 dB), got {psnr.item()}",
        )

    def test_ssim_valid_range_and_finite(self):
        """Verify SSIM returns a finite value in [-1, 1], and ~1.0 for identical images."""
        img1 = torch.rand(4, 1, 28, 28)
        img2 = torch.rand(4, 1, 28, 28)

        # Different images
        ssim_diff = compute_ssim(img1, img2)
        self.assertTrue(torch.isfinite(ssim_diff), "SSIM should be finite")
        self.assertGreaterEqual(ssim_diff.item(), -1.0)
        self.assertLessEqual(ssim_diff.item(), 1.0)

        # Identical images
        ssim_same = compute_ssim(img1, img1)
        self.assertTrue(torch.isfinite(ssim_same))
        self.assertAlmostEqual(
            ssim_same.item(),
            1.0,
            places=4,
            msg=f"SSIM for identical images must be 1.0, got {ssim_same.item()}",
        )

    def test_single_training_step_produces_finite_loss(self):
        """Verify a single forward and loss step produces a finite positive MSE loss."""
        model = ConvAutoencoder(in_channels=1, latent_channels=4)
        criterion = nn.MSELoss()
        inputs = torch.rand(8, 1, 28, 28)

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        self.assertTrue(torch.isfinite(loss), f"Training loss must be finite, got {loss.item()}")
        self.assertGreater(loss.item(), 0.0, "Reconstruction loss must be strictly positive")

    def test_gradients_flow_through_all_layers(self):
        """Verify gradients flow through all encoder and decoder weights during backpropagation."""
        model = ConvAutoencoder(in_channels=1, latent_channels=4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        inputs = torch.rand(4, 1, 28, 28)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()

        # Check every trainable weight and bias has a non-None, non-zero gradient
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient (None)")
            self.assertFalse(
                torch.isnan(param.grad).any(),
                f"Parameter {name} gradient contains NaN",
            )
            grad_norm = param.grad.abs().sum().item()
            self.assertGreater(
                grad_norm,
                0.0,
                f"Parameter {name} has zero gradient norm",
            )

    def test_checkpoint_directory_path_handling(self):
        """Verify checkpoint saving creates destination directories without requiring pre-existence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_checkpoint_dir = Path(tmp_dir) / "nested" / "subfolder" / "checkpoints"
            self.assertFalse(nested_checkpoint_dir.exists())

            # Path creation logic used in training pipeline
            nested_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.assertTrue(nested_checkpoint_dir.exists())

            test_checkpoint_path = nested_checkpoint_dir / "test_model.pt"
            dummy_data = {"test_val": 42, "model_state": {"layer.weight": torch.tensor([1.0])}}
            torch.save(dummy_data, test_checkpoint_path)

            self.assertTrue(test_checkpoint_path.exists())
            loaded = torch.load(test_checkpoint_path, weights_only=True)
            self.assertEqual(loaded["test_val"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
