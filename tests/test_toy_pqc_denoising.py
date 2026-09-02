"""
Unit tests for Module 2 Stage 1: Toy PQC Gaussian Denoising.

Verifies:
1. Toy dataset shapes (noisy and clean tensors match [N, dim]).
2. Deterministic seed behavior across independent dataset generations.
3. PQC output shape matches input dimension for batched and single-vector inputs.
4. Finite outputs bounded in [-1, 1] matching Pauli-Z expectation values.
5. Finite non-zero gradients generated via backpropagation.
6. A single training optimization step produces finite loss and updates weights.
7. Model parameter count reporting for quantum model and classical baseline.
"""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pqc import PQCDenoiser, ClassicalToyBaseline
from src.data.toy_denoising import (
    generate_toy_denoising_data,
    ToyGaussianDenoisingDataset,
    get_toy_denoising_dataloaders,
)


class TestToyPQCDenoising(unittest.TestCase):
    """Test suite for toy Gaussian denoising dataset, PQC model, and training flow."""

    def setUp(self):
        torch.manual_seed(42)
        self.n_qubits = 6
        self.n_layers = 2
        self.model = PQCDenoiser(n_qubits=self.n_qubits, n_layers=self.n_layers)

    def test_toy_dataset_shapes(self):
        """Verify toy dataset returns tensors of shape [num_samples, dim] and batches of [B, dim]."""
        num_samples = 40
        dim = 6
        noisy, clean = generate_toy_denoising_data(
            num_samples=num_samples,
            dim=dim,
            noise_std=0.2,
            seed=42,
        )
        self.assertEqual(noisy.shape, torch.Size([num_samples, dim]))
        self.assertEqual(clean.shape, torch.Size([num_samples, dim]))

        dataset = ToyGaussianDenoisingDataset(num_samples=20, dim=dim, seed=42)
        sample_noisy, sample_clean = dataset[0]
        self.assertEqual(sample_noisy.shape, torch.Size([dim]))
        self.assertEqual(sample_clean.shape, torch.Size([dim]))

        train_loader, val_loader = get_toy_denoising_dataloaders(
            num_train=32,
            num_val=16,
            dim=dim,
            batch_size=8,
            seed=42,
        )
        batch_noisy, batch_clean = next(iter(train_loader))
        self.assertEqual(batch_noisy.shape, torch.Size([8, dim]))
        self.assertEqual(batch_clean.shape, torch.Size([8, dim]))

    def test_deterministic_seed_behavior(self):
        """Verify generation is strictly reproducible given the same random seed."""
        n1, c1 = generate_toy_denoising_data(30, 6, noise_std=0.2, seed=42)
        n2, c2 = generate_toy_denoising_data(30, 6, noise_std=0.2, seed=42)
        self.assertTrue(torch.equal(n1, n2), "Noisy tensors with seed 42 must be identical")
        self.assertTrue(torch.equal(c1, c2), "Clean tensors with seed 42 must be identical")

        n_diff, _ = generate_toy_denoising_data(30, 6, noise_std=0.2, seed=999)
        self.assertFalse(
            torch.equal(n1, n_diff),
            "Different seeds must generate different noisy vectors",
        )

    def test_pqc_output_shape(self):
        """Verify PQC maps [B, n_qubits] to [B, n_qubits] and 1D [n_qubits] to [n_qubits]."""
        batch_size = 4
        x_batched = torch.randn(batch_size, self.n_qubits)
        out_batched = self.model(x_batched)
        self.assertEqual(out_batched.shape, torch.Size([batch_size, self.n_qubits]))

        x_single = torch.randn(self.n_qubits)
        out_single = self.model(x_single)
        self.assertEqual(out_single.shape, torch.Size([self.n_qubits]))

    def test_finite_outputs(self):
        """Verify PQC produces finite outputs strictly in [-1, 1] matching Pauli-Z observables."""
        x = torch.randn(8, self.n_qubits) * 5.0  # test strong dynamic ranges
        out = self.model(x)

        self.assertTrue(torch.isfinite(out).all(), "Outputs must contain no NaNs or Infs")
        self.assertGreaterEqual(
            out.min().item(),
            -1.0 - 1e-6,
            "Pauli-Z expectation must be >= -1.0",
        )
        self.assertLessEqual(
            out.max().item(),
            1.0 + 1e-6,
            "Pauli-Z expectation must be <= 1.0",
        )

    def test_finite_nonzero_gradients(self):
        """Verify backpropagation produces finite, non-zero gradients on circuit parameters."""
        x = torch.randn(4, self.n_qubits)
        target = torch.randn(4, self.n_qubits)

        out = self.model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} gradient is None")
            self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients")
            self.assertGreater(
                param.grad.abs().sum().item(),
                0.0,
                f"Parameter {name} has zero gradient",
            )

    def test_one_training_step_produces_finite_loss(self):
        """Verify one optimization step executes, computes finite loss, and updates parameters."""
        optimizer = optim.Adam(self.model.parameters(), lr=0.05)
        criterion = nn.MSELoss()

        x = torch.randn(4, self.n_qubits)
        target = torch.randn(4, self.n_qubits)

        # Record initial weights
        initial_weights = next(self.model.parameters()).clone()

        optimizer.zero_grad()
        out = self.model(x)
        loss = criterion(out, target)
        self.assertTrue(torch.isfinite(loss), f"Loss must be finite, got {loss.item()}")
        loss.backward()
        optimizer.step()

        updated_weights = next(self.model.parameters())
        self.assertFalse(
            torch.equal(initial_weights, updated_weights),
            "Model parameters must be updated after optimizer.step()",
        )

    def test_model_parameter_count_reporting(self):
        """Verify parameter count calculation for PQC (n_qubits * n_layers) and classical baseline."""
        # 6 qubits * 2 layers = 12 parameters
        self.assertEqual(
            self.model.num_parameters,
            12,
            f"Expected 12 parameters for 6-qubit 2-layer PQC, got {self.model.num_parameters}",
        )

        # 6 qubits * 3 layers = 18 parameters
        model_3l = PQCDenoiser(n_qubits=6, n_layers=3)
        self.assertEqual(model_3l.num_parameters, 18)

        # Classical baseline with hidden_dim=1, bias=False has 6 + 6 = 12 parameters
        classical_model = ClassicalToyBaseline(dim=6, hidden_dim=1, bias=False)
        self.assertEqual(
            classical_model.num_parameters,
            12,
            f"Expected 12 parameters for classical baseline, got {classical_model.num_parameters}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
