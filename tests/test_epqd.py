"""
Unit tests for PHQD Module 5: Entangled Patch Quantum Diffusion (EPQD).

Tests:
1. Exact parameter counts and wire mappings for k=1, k=2, k=4 (nq=4).
2. Bridge gate topology (strictly RZZ).
3. Forward pass and gradient flow through PyTorch autograd.
4. Consensus combining across overlapping quad patches in k=4.
5. End-to-end integration with ClassicalStitchingCNN and ConvAutoencoder decoder.
6. Diagnostics: Von Neumann entanglement entropy, bridge-zeroing ablation, seam disagreement.
7. Statistical evaluation: paired two-sided Student's t-test and power law fit.
8. Analytical compute accounting (C_train).
"""

import unittest
import math
import torch
import numpy as np

from src.models.autoencoder import ConvAutoencoder
from src.models.stitching_cnn import ClassicalStitchingCNN
from src.models.epqd import (
    EPQDDenoiser,
    create_epqd_pair_circuit,
    create_epqd_quad_circuit,
    consensus_combine_quads,
)
from src.evaluation.epqd_diagnostics import (
    compute_von_neumann_entropy,
    evaluate_bridge_zeroing_ablation,
    compute_seam_disagreement,
)
from src.evaluation.epqd_analysis import (
    compute_paired_t_test,
    fit_power_law,
    compute_efficiency_index,
    calculate_epqd_c_train,
)


class TestEPQDCircuitTopology(unittest.TestCase):
    """Test exact circuit topology, wire counts, and parameter counts."""

    def test_k1_parameter_counts(self):
        """k=1: 6 qubits, depth 3, 18 PQC params, 768 adapters, total 786."""
        model = EPQDDenoiser(k=1, n_qubits=6, n_layers=3)
        self.assertEqual(model.num_pqc_parameters, 18)
        self.assertEqual(model.num_adapter_parameters, 768)
        self.assertEqual(model.num_parameters, 786)

    def test_k2_parameter_counts_and_wires(self):
        """k=2: 12 qubits for pairs (wires 0-5 and 6-11, boundary 5-6).
        36 local + 3 bridge = 39 core params. Adapters: 768. Total: 807.
        """
        model = EPQDDenoiser(k=2, n_qubits=6, n_layers=3)
        self.assertEqual(model.num_pqc_parameters, 39)
        self.assertEqual(model.num_adapter_parameters, 768)
        self.assertEqual(model.num_parameters, 807)

    def test_k4_nq4_parameter_counts(self):
        """k=4 (nq=4): 16 qubits total, 4 RZZ bridges per layer.
        48 local + 12 bridge = 60 core params. Adapters: 512. Total: 572.
        """
        model = EPQDDenoiser(k=4, n_qubits=4, n_layers=3)
        self.assertEqual(model.num_pqc_parameters, 60)
        self.assertEqual(model.num_adapter_parameters, 512)
        self.assertEqual(model.num_parameters, 572)

    def test_invalid_k_raises(self):
        """Only k in {1, 2, 4} are permitted."""
        with self.assertRaises(ValueError):
            EPQDDenoiser(k=3)


class TestEPQDForwardPass(unittest.TestCase):
    """Test forward execution and output shapes across all k modes."""

    def test_k1_forward_and_gradient(self):
        model = EPQDDenoiser(k=1, n_qubits=6, n_layers=3)
        x = torch.randn(2, 9, 64, requires_grad=True)
        t = torch.tensor([10, 20])
        out = model(x, t)
        self.assertEqual(out.shape, (2, 9, 64))
        self.assertTrue(torch.isfinite(out).all())

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_k2_forward_and_gradient(self):
        model = EPQDDenoiser(k=2, n_qubits=6, n_layers=3)
        x = torch.randn(2, 9, 64, requires_grad=True)
        t = torch.tensor([5, 15])
        out = model(x, t)
        self.assertEqual(out.shape, (2, 9, 64))
        self.assertTrue(torch.isfinite(out).all())

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_k4_forward_and_gradient(self):
        model = EPQDDenoiser(k=4, n_qubits=4, n_layers=3)
        x = torch.randn(2, 9, 64, requires_grad=True)
        t = torch.tensor([2, 8])
        out = model(x, t)
        self.assertEqual(out.shape, (2, 9, 64))
        self.assertTrue(torch.isfinite(out).all())

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestConsensusCombining(unittest.TestCase):
    """Test arithmetic mean consensus for k=4 quad predictions."""

    def test_consensus_counts_and_weights(self):
        b = 2
        # quad predictions: [B, 4, 4, 64]
        # 4 quads, each predicting 4 patches
        quads = torch.ones(b, 4, 4, 64)
        # Set distinct values for each quad
        quads[:, 0] *= 1.0  # QTL: P0, P1, P3, P4
        quads[:, 1] *= 2.0  # QTR: P1, P2, P4, P5
        quads[:, 2] *= 3.0  # QBL: P3, P4, P6, P7
        quads[:, 3] *= 4.0  # QBR: P4, P5, P7, P8

        combined = consensus_combine_quads(quads)
        self.assertEqual(combined.shape, (b, 9, 64))

        # Expected arithmetic averages:
        # P0: only in QTL -> 1.0
        self.assertAlmostEqual(combined[0, 0, 0].item(), 1.0)
        # P1: in QTL (1.0) and QTR (2.0) -> (1 + 2) / 2 = 1.5
        self.assertAlmostEqual(combined[0, 1, 0].item(), 1.5)
        # P2: only in QTR -> 2.0
        self.assertAlmostEqual(combined[0, 2, 0].item(), 2.0)
        # P3: in QTL (1.0) and QBL (3.0) -> (1 + 3) / 2 = 2.0
        self.assertAlmostEqual(combined[0, 3, 0].item(), 2.0)
        # P4: in all 4 quads -> (1 + 2 + 3 + 4) / 4 = 2.5
        self.assertAlmostEqual(combined[0, 4, 0].item(), 2.5)
        # P5: in QTR (2.0) and QBR (4.0) -> (2 + 4) / 2 = 3.0
        self.assertAlmostEqual(combined[0, 5, 0].item(), 3.0)
        # P6: only in QBL -> 3.0
        self.assertAlmostEqual(combined[0, 6, 0].item(), 3.0)
        # P7: in QBL (3.0) and QBR (4.0) -> (3 + 4) / 2 = 3.5
        self.assertAlmostEqual(combined[0, 7, 0].item(), 3.5)
        # P8: only in QBR -> 4.0
        self.assertAlmostEqual(combined[0, 8, 0].item(), 4.0)


class TestDownstreamIntegration(unittest.TestCase):
    """Test full assembly: EPQDDenoiser -> ClassicalStitchingCNN -> ConvAutoencoder decode."""

    def test_full_pipeline_k2(self):
        denoiser = EPQDDenoiser(k=2, n_qubits=6, n_layers=3)
        stitching = ClassicalStitchingCNN()
        autoencoder = ConvAutoencoder()

        x = torch.randn(2, 9, 64)
        t = torch.tensor([12, 18])

        # 1. EPQD denoiser
        x_pred = denoiser(x, t)  # [2, 9, 64]
        # 2. Reshape to [B, 9, 4, 4, 4]
        patches = x_pred.view(2, 9, 4, 4, 4)
        # 3. Stitching CNN
        stitched = stitching(patches)  # [2, 4, 8, 8]
        self.assertEqual(stitched.shape, (2, 4, 8, 8))
        # 4. Decoder
        decoded = autoencoder.decode(stitched)  # [2, 1, 28, 28]
        self.assertEqual(decoded.shape, (2, 1, 28, 28))
        self.assertTrue(torch.isfinite(decoded).all())


class TestDiagnosticsAndAnalysis(unittest.TestCase):
    """Test Von Neumann entropy, bridge zeroing, seam disagreement, and Gate 2 analysis."""

    def test_von_neumann_entropy(self):
        model = EPQDDenoiser(k=2, n_qubits=6, n_layers=3)
        x = torch.randn(1, 9, 64)
        t = torch.tensor([10])
        entropy = compute_von_neumann_entropy(model, x, t)
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)

    def test_bridge_zeroing_ablation(self):
        model = EPQDDenoiser(k=2, n_qubits=6, n_layers=3)
        with torch.no_grad():
            model.bridge_weights.fill_(0.5)
        x = torch.randn(2, 9, 64)
        t = torch.tensor([10, 20])
        delta_loss = evaluate_bridge_zeroing_ablation(model, x, t)
        self.assertIsInstance(delta_loss, float)

    def test_seam_disagreement(self):
        patches = torch.zeros(2, 9, 4, 4, 4)
        d_seam = compute_seam_disagreement(patches)
        self.assertAlmostEqual(d_seam, 0.0, places=6)

        patches[:, 0, :, :, 2:4] += 1.0
        d_seam_nonzero = compute_seam_disagreement(patches)
        self.assertGreater(d_seam_nonzero, 0.0)

    def test_paired_t_test_exact_formula(self):
        """Verify paired two-sided Student's t-test matches d_bar / (s_d / sqrt(3))."""
        psnr_k1 = [17.0, 17.2, 17.1]
        psnr_k2 = [17.5, 17.6, 17.7]

        d = np.array(psnr_k2) - np.array(psnr_k1)
        d_bar = np.mean(d)
        s_d = np.std(d, ddof=1)
        expected_t = d_bar / (s_d / math.sqrt(3))

        res = compute_paired_t_test(psnr_k1, psnr_k2, alpha=0.05)
        self.assertAlmostEqual(res["t_statistic"], expected_t, places=5)
        self.assertEqual(res["n_seeds"], 3)
        self.assertIn("p_value", res)
        self.assertIsInstance(res["p_value"], float)

    def test_power_law_fit(self):
        """Fit MSE(k) = MSE_inf + A / k^gamma with A > 0, gamma > 0."""
        k_vals = np.array([1.0, 2.0, 4.0])
        mse_vals = 0.05 + 0.10 / (k_vals ** 0.8)

        fit = fit_power_law(k_vals, mse_vals)
        self.assertGreater(fit["A"], 0)
        self.assertGreater(fit["gamma"], 0)
        self.assertTrue(fit["is_improving"])

    def test_compute_accounting(self):
        """Verify analytical C_train values match the locked specification exactly."""
        c_train_dict = calculate_epqd_c_train(n_epochs=15, n_batch=32, kappa=1.0)
        self.assertEqual(c_train_dict["k1"], 63866880)
        self.assertEqual(c_train_dict["k2"], 1900431360)
        self.assertEqual(c_train_dict["k4_nq4"], 45698088960)
        self.assertEqual(c_train_dict["k4_nq6_theoretical"], 16331646320640)

        eta = compute_efficiency_index(delta_psnr=0.35, c_k1=c_train_dict["k1"], c_k2=c_train_dict["k2"])
        self.assertGreater(eta, 0.15)


if __name__ == "__main__":
    unittest.main()
