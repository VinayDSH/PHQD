"""
Unit tests for Module 1 baseline Convolutional Autoencoder.

Verifies:
1. Input shape [B, 1, 28, 28] produces latent shape [B, 4, 8, 8].
2. decode() maps [B, 4, 8, 8] to [B, 1, 28, 28].
3. forward() maps [B, 1, 28, 28] to [B, 1, 28, 28].
4. Output values are strictly within [0, 1].
5. Latent representation has no forced [0, 1] or non-negativity constraint
   (verifies no Sigmoid or ReLU is applied to the latent).
6. Total trainable parameter count is exactly 46,277.
"""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import ConvAutoencoder


class TestConvAutoencoder(unittest.TestCase):
    """Test suite for ConvAutoencoder architecture and tensor geometries."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = ConvAutoencoder(in_channels=1, latent_channels=4)
        self.batch_sizes = [1, 4, 16]

    def test_encode_latent_shape(self):
        """Verify input [B, 1, 28, 28] encodes to exactly [B, 4, 8, 8]."""
        for b in self.batch_sizes:
            x = torch.randn(b, 1, 28, 28)
            z = self.model.encode(x)
            self.assertEqual(
                z.shape,
                torch.Size([b, 4, 8, 8]),
                f"Expected latent shape ({b}, 4, 8, 8), got {z.shape}",
            )

    def test_decode_output_shape(self):
        """Verify latent [B, 4, 8, 8] decodes to exactly [B, 1, 28, 28]."""
        for b in self.batch_sizes:
            z = torch.randn(b, 4, 8, 8)
            rec = self.model.decode(z)
            self.assertEqual(
                rec.shape,
                torch.Size([b, 1, 28, 28]),
                f"Expected reconstructed shape ({b}, 1, 28, 28), got {rec.shape}",
            )

    def test_forward_shape(self):
        """Verify forward(x) maps [B, 1, 28, 28] to [B, 1, 28, 28]."""
        for b in self.batch_sizes:
            x = torch.rand(b, 1, 28, 28)
            out = self.model(x)
            self.assertEqual(
                out.shape,
                torch.Size([b, 1, 28, 28]),
                f"Expected forward shape ({b}, 1, 28, 28), got {out.shape}",
            )

    def test_output_values_within_zero_one(self):
        """Verify output values from decode() and forward() fall strictly within [0, 1]."""
        # Test random inputs covering extreme negative and positive values
        x = torch.randn(8, 1, 28, 28) * 10.0
        out = self.model(x)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

        # Direct decode of extreme latent values
        z = torch.randn(8, 4, 8, 8) * 50.0
        out_z = self.model.decode(z)
        self.assertGreaterEqual(out_z.min().item(), 0.0)
        self.assertLessEqual(out_z.max().item(), 1.0)

    def test_latent_has_no_forced_activation_constraint(self):
        """Verify that the latent layer has no forced [0, 1] or non-negativity constraint.

        Confirms neither Sigmoid nor ReLU is attached to the final encoder layer.
        """
        # Structural check: final encoder layer must be Conv2d, not an activation
        self.assertIsInstance(
            self.model.encoder[-1],
            nn.Conv2d,
            "The final layer of the encoder must be Conv2d without an activation.",
        )

        # Functional check: ensure latent can take both negative values and values > 1
        # Set bias to negative value: latent must preserve negative values without ReLU clamping
        with torch.no_grad():
            saved_bias = self.model.encoder[-1].bias.clone()
            self.model.encoder[-1].bias.fill_(-5.0)
            zeros_in = torch.zeros(2, 1, 28, 28)
            z_neg = self.model.encode(zeros_in)
            self.assertTrue(
                (z_neg < 0).all(),
                "Latent representation unexpectedly clamped negative values (possible ReLU).",
            )

            # Set bias to > 1: latent must preserve values > 1 without Sigmoid squashing
            self.model.encoder[-1].bias.fill_(5.0)
            z_pos = self.model.encode(zeros_in)
            self.assertTrue(
                (z_pos > 1).all(),
                "Latent representation unexpectedly squashed values to [0, 1] (possible Sigmoid).",
            )

            # Restore original bias
            self.model.encoder[-1].bias.copy_(saved_bias)

        # Dynamic check with natural forward pass
        x = torch.randn(16, 1, 28, 28) * 5.0
        z = self.model.encode(x)
        self.assertTrue(
            (z < 0).any().item(),
            "Expected unconstrained latent to contain negative values with random inputs.",
        )
        self.assertTrue(
            (z > 1).any().item(),
            "Expected unconstrained latent to contain values > 1 with random inputs.",
        )

    def test_exact_trainable_parameter_count(self):
        """Verify the total trainable parameter count is exactly 46,277."""
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.assertEqual(
            trainable_params,
            46277,
            f"Expected exactly 46,277 parameters, but got {trainable_params}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
