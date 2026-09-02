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
import tempfile
import json
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pqc import (
    PQCDenoiser,
    ClassicalToyBaseline,
    ParameterMatchedClassicalBaseline,
    CirculantLinear,
)
from src.data.toy_denoising import (
    generate_toy_denoising_data,
    ToyGaussianDenoisingDataset,
    get_toy_denoising_dataloaders,
)
from experiments.module2.toy_pqc_denoising import (
    run_toy_denoising_experiment,
    set_seed,
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

    def test_independent_seed_initialization(self):
        """Verify PQC and classical models initialize deterministically with independent seeds."""
        # PQC deterministic initialization with same seed
        set_seed(101)
        pqc_1 = PQCDenoiser(n_qubits=6, n_layers=2)
        w_pqc_1 = next(pqc_1.parameters()).clone()

        set_seed(101)
        pqc_2 = PQCDenoiser(n_qubits=6, n_layers=2)
        w_pqc_2 = next(pqc_2.parameters()).clone()

        set_seed(102)
        pqc_3 = PQCDenoiser(n_qubits=6, n_layers=2)
        w_pqc_3 = next(pqc_3.parameters()).clone()

        self.assertTrue(torch.equal(w_pqc_1, w_pqc_2), "Same PQC init seed must produce identical weights")
        self.assertFalse(torch.equal(w_pqc_1, w_pqc_3), "Different PQC init seed must produce distinct weights")

        # Classical deterministic initialization with same seed
        set_seed(201)
        c_1 = ClassicalToyBaseline(dim=6, hidden_dim=1, bias=False)
        w_c_1 = next(c_1.parameters()).clone()

        set_seed(201)
        c_2 = ClassicalToyBaseline(dim=6, hidden_dim=1, bias=False)
        w_c_2 = next(c_2.parameters()).clone()

        set_seed(202)
        c_3 = ClassicalToyBaseline(dim=6, hidden_dim=1, bias=False)
        w_c_3 = next(c_3.parameters()).clone()

        self.assertTrue(torch.equal(w_c_1, w_c_2), "Same classical init seed must produce identical weights")
        self.assertFalse(torch.equal(w_c_1, w_c_3), "Different classical init seed must produce distinct weights")

        # Parameter-matched classical baseline deterministic initialization
        set_seed(301)
        pm_1 = ParameterMatchedClassicalBaseline(dim=6, layers=2)
        w_pm_1 = next(pm_1.parameters()).clone()

        set_seed(301)
        pm_2 = ParameterMatchedClassicalBaseline(dim=6, layers=2)
        w_pm_2 = next(pm_2.parameters()).clone()

        set_seed(302)
        pm_3 = ParameterMatchedClassicalBaseline(dim=6, layers=2)
        w_pm_3 = next(pm_3.parameters()).clone()

        self.assertTrue(torch.equal(w_pm_1, w_pm_2), "Same parameter-matched init seed must produce identical weights")
        self.assertFalse(torch.equal(w_pm_1, w_pm_3), "Different parameter-matched init seed must produce distinct weights")

    def test_multiple_experiment_results_retention(self):
        """Verify that sequential experiment runs are preserved rather than overwritten in JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "toy_pqc_denoising_results.json"

            # Run 1
            res_1 = run_toy_denoising_experiment(
                qubits=6,
                layers=2,
                noise_std=0.2,
                seed=42,
                epochs=1,
                num_train=16,
                num_val=8,
                compare_classical=True,
                output_dir=tmp_dir,
                pqc_seed=111,
                classical_seed=222,
            )

            self.assertTrue(log_file.exists(), "Log file should exist after run 1")
            with open(log_file, "r", encoding="utf-8") as f:
                logs_after_run1 = json.load(f)

            self.assertIsInstance(logs_after_run1, list, "Logs should be stored as a list of runs")
            self.assertEqual(len(logs_after_run1), 1, "Expected exactly 1 run entry after run 1")
            entry_1 = logs_after_run1[0]
            self.assertEqual(entry_1["run_id"], 1)
            self.assertEqual(entry_1["qubits"], 6)
            self.assertEqual(entry_1["layers"], 2)
            self.assertEqual(entry_1["noise_std"], 0.2)
            self.assertEqual(entry_1["seed"], 42)
            self.assertEqual(entry_1["pqc_init_seed"], 111)
            self.assertEqual(entry_1["classical_init_seed"], 222)
            self.assertIn("pqc_results", entry_1)
            self.assertIn("classical_results", entry_1)

            # Run 2 (different seed/params)
            res_2 = run_toy_denoising_experiment(
                qubits=6,
                layers=2,
                noise_std=0.3,
                seed=43,
                epochs=1,
                num_train=16,
                num_val=8,
                compare_classical=True,
                output_dir=tmp_dir,
                pqc_seed=333,
                classical_seed=444,
            )

            with open(log_file, "r", encoding="utf-8") as f:
                logs_after_run2 = json.load(f)

            self.assertEqual(len(logs_after_run2), 2, "Expected exactly 2 run entries after run 2")

            # Verify run 1 metrics were preserved and not overwritten
            self.assertEqual(logs_after_run2[0]["seed"], 42)
            self.assertEqual(logs_after_run2[0]["noise_std"], 0.2)
            self.assertEqual(logs_after_run2[0]["pqc_init_seed"], 111)
            self.assertEqual(logs_after_run2[0]["run_id"], 1)

            # Verify run 2 metrics
            entry_2 = logs_after_run2[1]
            self.assertEqual(entry_2["run_id"], 2)
            self.assertEqual(entry_2["seed"], 43)
            self.assertEqual(entry_2["noise_std"], 0.3)
            self.assertEqual(entry_2["pqc_init_seed"], 333)
            self.assertEqual(entry_2["classical_init_seed"], 444)

            # Test backward compatibility: pre-existing legacy single-dict log
            legacy_file = Path(tmp_dir) / "legacy_test" / "toy_pqc_denoising_results.json"
            legacy_file.parent.mkdir(parents=True, exist_ok=True)
            legacy_dict = {"experiment": "toy_gaussian_denoising", "seed": 99, "qubit_count": 6}
            with open(legacy_file, "w", encoding="utf-8") as f:
                json.dump(legacy_dict, f)

            run_toy_denoising_experiment(
                qubits=6,
                layers=2,
                noise_std=0.2,
                seed=100,
                epochs=1,
                num_train=16,
                num_val=8,
                compare_classical=False,
                output_dir=str(legacy_file.parent),
            )

            with open(legacy_file, "r", encoding="utf-8") as f:
                converted_logs = json.load(f)
            self.assertIsInstance(converted_logs, list)
            self.assertEqual(len(converted_logs), 2)
            self.assertEqual(converted_logs[0]["seed"], 99)
            self.assertEqual(converted_logs[1]["seed"], 100)

    def test_parameter_matched_exact_equality_and_properties(self):
        """Verify exact parameter count equality between PQC and classical baseline for D in {4, 6, 8} and L in {1, 2, 3},

        along with output shape, finite bounds, Tanh output range, and gradient backpropagation.
        """
        dims = [4, 6, 8]
        layers_list = [1, 2, 3]

        for D in dims:
            # Check single CirculantLinear layer has strictly D parameters
            circ_layer = CirculantLinear(dim=D)
            self.assertEqual(
                circ_layer.num_parameters,
                D,
                f"CirculantLinear({D}) must have exactly {D} parameters",
            )

            for L in layers_list:
                expected_params = D * L

                # 1. PQC Denoiser
                pqc = PQCDenoiser(n_qubits=D, n_layers=L)
                self.assertEqual(
                    pqc.num_parameters,
                    expected_params,
                    f"PQC({D}, {L}) expected {expected_params} parameters, got {pqc.num_parameters}",
                )

                # 2. Parameter-Matched Classical Baseline
                classical = ParameterMatchedClassicalBaseline(dim=D, layers=L)
                self.assertEqual(
                    classical.num_parameters,
                    expected_params,
                    f"Classical({D}, {L}) expected {expected_params} parameters, got {classical.num_parameters}",
                )

                # 3. Exact 1:1 Parameter Equality
                self.assertEqual(
                    classical.num_parameters,
                    pqc.num_parameters,
                    f"Parameter mismatch for D={D}, L={L}: Classical={classical.num_parameters}, PQC={pqc.num_parameters}",
                )

                # 4. Output Shape Verification (Batched and 1D)
                batch_size = 5
                x_batched = torch.randn(batch_size, D)
                out_batched = classical(x_batched)
                self.assertEqual(
                    out_batched.shape,
                    torch.Size([batch_size, D]),
                    f"Batched output shape mismatch for D={D}, L={L}",
                )

                x_1d = torch.randn(D)
                out_1d = classical(x_1d)
                self.assertEqual(
                    out_1d.shape,
                    torch.Size([D]),
                    f"1D output shape mismatch for D={D}, L={L}",
                )

                # 5. Finite Outputs and Tanh Range Bound [-1, 1] for both batched and 1D
                self.assertTrue(
                    torch.isfinite(out_batched).all(),
                    f"Batched outputs must be finite for D={D}, L={L}",
                )
                self.assertTrue(
                    torch.isfinite(out_1d).all(),
                    f"1D outputs must be finite for D={D}, L={L}",
                )
                self.assertGreaterEqual(
                    out_batched.min().item(),
                    -1.0 - 1e-6,
                    f"Batched Tanh output must be >= -1.0 for D={D}, L={L}",
                )
                self.assertLessEqual(
                    out_batched.max().item(),
                    1.0 + 1e-6,
                    f"Batched Tanh output must be <= 1.0 for D={D}, L={L}",
                )
                self.assertGreaterEqual(
                    out_1d.min().item(),
                    -1.0 - 1e-6,
                    f"1D Tanh output must be >= -1.0 for D={D}, L={L}",
                )
                self.assertLessEqual(
                    out_1d.max().item(),
                    1.0 + 1e-6,
                    f"1D Tanh output must be <= 1.0 for D={D}, L={L}",
                )

                # 6. Backward Pass and Gradient Flow
                target = torch.randn_like(out_batched)
                loss = nn.MSELoss()(out_batched, target)
                loss.backward()

                for name, param in classical.named_parameters():
                    self.assertIsNotNone(
                        param.grad,
                        f"Gradient is None for parameter {name} (D={D}, L={L})",
                    )
                    self.assertFalse(
                        torch.isnan(param.grad).any(),
                        f"Gradient has NaNs for parameter {name} (D={D}, L={L})",
                    )
                    self.assertGreater(
                        param.grad.abs().sum().item(),
                        0.0,
                        f"Gradient is zero for parameter {name} (D={D}, L={L})",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
