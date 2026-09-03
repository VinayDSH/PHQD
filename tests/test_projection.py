"""
Unit tests for Module 2 Stage 2: Fixed 64-D to 6-D Projection and Normalization.

Verifies:
1. Output shape for single vectors, 2D batches, 3D patch grids, and unflattened spatial patches.
2. Determinism and global seed independence across runs and random seeds.
3. Finiteness of forward outputs, numerical stability under extreme values, and input gradient flow.
4. Frozen parameter guarantees (zero trainable parameters, zero nn.Parameters, no buffer updates).
5. Reproducibility, strict training-data isolation, and semi-orthogonality (P @ P.T = I_6).
6. Methodological verification of intentionally lossy 64-D -> 6-D projection:
   - Arbitrary 64-D patches cannot be exactly reconstructed (non-zero reconstruction error).
   - Vectors within the 6-D projection subspace are reconstructed faithfully.
   - Subspace projection operator is idempotent (projecting reconstructed patches recovers exact 6-D coordinates).
"""

import math
import tempfile
from pathlib import Path
import unittest
import torch

from src.models.projection import FixedPatchProjector


class TestFixedPatchProjector(unittest.TestCase):
    """Comprehensive test suite for FixedPatchProjector."""

    def setUp(self):
        torch.manual_seed(42)
        self.in_dim = 64
        self.out_dim = 6
        self.seed = 42

        # Synthetic training data
        self.train_data = torch.randn(128, self.in_dim) + 2.5
        self.projector = FixedPatchProjector(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            seed=self.seed,
            norm_type="standard",
        )
        self.projector.fit(self.train_data)

    # -------------------------------------------------------------
    # 1. SHAPE TESTS
    # -------------------------------------------------------------
    def test_shape_single_vector_1d(self):
        """Verify 1D input (64,) maps to (6,) and inverts back to (64,)."""
        x = torch.randn(self.in_dim)
        z = self.projector(x)
        self.assertEqual(z.shape, torch.Size([self.out_dim]))
        x_rec = self.projector.pseudo_inverse_back_project(z)
        self.assertEqual(x_rec.shape, torch.Size([self.in_dim]))

    def test_shape_batched_2d(self):
        """Verify batched 2D inputs [B, 64] map to [B, 6] across diverse batch sizes."""
        for b in [1, 7, 32, 64]:
            x = torch.randn(b, self.in_dim)
            z = self.projector(x)
            self.assertEqual(z.shape, torch.Size([b, self.out_dim]))
            x_rec = self.projector.pseudo_inverse_back_project(z)
            self.assertEqual(x_rec.shape, torch.Size([b, self.in_dim]))

    def test_shape_patch_grid_3d(self):
        """Verify 3D patch grid [B, 9, 64] maps to [B, 9, 6]."""
        x = torch.randn(8, 9, self.in_dim)
        z = self.projector(x)
        self.assertEqual(z.shape, torch.Size([8, 9, self.out_dim]))
        x_rec = self.projector.pseudo_inverse_back_project(z)
        self.assertEqual(x_rec.shape, torch.Size([8, 9, self.in_dim]))

    def test_shape_unflattened_spatial_patch(self):
        """Verify unflattened spatial patches [B, 4, 4, 4] map to [B, 6] and back-project."""
        x = torch.randn(8, 4, 4, 4)
        z = self.projector(x)
        self.assertEqual(z.shape, torch.Size([8, self.out_dim]))
        x_rec = self.projector.pseudo_inverse_back_project(z, target_shape=(4, 4, 4))
        self.assertEqual(x_rec.shape, torch.Size([8, 4, 4, 4]))

    def test_shape_unflattened_grid(self):
        """Verify unflattened grid [B, 9, 4, 4, 4] maps to [B, 9, 6] and back-projects."""
        x = torch.randn(4, 9, 4, 4, 4)
        z = self.projector(x)
        self.assertEqual(z.shape, torch.Size([4, 9, self.out_dim]))
        x_rec = self.projector.pseudo_inverse_back_project(z, target_shape=(4, 4, 4))
        self.assertEqual(x_rec.shape, torch.Size([4, 9, 4, 4, 4]))

    # -------------------------------------------------------------
    # 2. DETERMINISM TESTS
    # -------------------------------------------------------------
    def test_determinism_same_seed(self):
        """Verify identical seed produces bit-for-bit identical projection matrix."""
        p1 = FixedPatchProjector(seed=42)
        p2 = FixedPatchProjector(seed=42)
        self.assertTrue(torch.equal(p1.projection_matrix, p2.projection_matrix))

    def test_determinism_different_seed(self):
        """Verify distinct seeds produce different projection matrices."""
        p1 = FixedPatchProjector(seed=42)
        p2 = FixedPatchProjector(seed=999)
        self.assertFalse(torch.equal(p1.projection_matrix, p2.projection_matrix))

    def test_determinism_forward_output(self):
        """Verify forward pass on identical inputs yields bit-for-bit identical output."""
        p1 = FixedPatchProjector(seed=123).fit(self.train_data)
        p2 = FixedPatchProjector(seed=123).fit(self.train_data)
        x = torch.randn(10, self.in_dim)
        z1 = p1(x)
        z2 = p2(x)
        self.assertTrue(torch.equal(z1, z2))

    def test_global_rng_isolation(self):
        """Verify instantiating projector does not perturb global PyTorch RNG."""
        torch.manual_seed(777)
        expected_rand = torch.randn(5)
        torch.manual_seed(777)
        _ = FixedPatchProjector(seed=42)
        actual_rand = torch.randn(5)
        self.assertTrue(torch.equal(expected_rand, actual_rand))

    # -------------------------------------------------------------
    # 3. FINITENESS TESTS
    # -------------------------------------------------------------
    def test_finiteness_standard_inputs(self):
        """Verify outputs on Gaussian inputs are finite without NaN or Inf."""
        x = torch.randn(50, self.in_dim)
        z = self.projector(x)
        self.assertTrue(torch.isfinite(z).all())
        x_rec = self.projector.pseudo_inverse_back_project(z)
        self.assertTrue(torch.isfinite(x_rec).all())

    def test_finiteness_extreme_inputs(self):
        """Verify stability on extreme values (all zeros, 1e5, -1e5, 1e-6)."""
        extremes = [
            torch.zeros(10, self.in_dim),
            torch.full((10, self.in_dim), 1e5),
            torch.full((10, self.in_dim), -1e5),
            torch.full((10, self.in_dim), 1e-6),
        ]
        for x in extremes:
            z = self.projector(x)
            self.assertTrue(torch.isfinite(z).all())

    def test_finiteness_constant_feature_epsilon_protection(self):
        """Verify epsilon prevents division by zero when training feature has zero variance."""
        constant_data = torch.ones(50, self.in_dim)
        p = FixedPatchProjector().fit(constant_data)
        z = p(constant_data)
        self.assertTrue(torch.isfinite(z).all())

    def test_finite_gradients_to_inputs(self):
        """Verify backward pass through projector computes finite, non-NaN input gradients."""
        x = torch.randn(16, self.in_dim, requires_grad=True)
        z = self.projector(x)
        loss = z.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    # -------------------------------------------------------------
    # 4. FROZEN PARAMETERS TESTS
    # -------------------------------------------------------------
    def test_zero_trainable_parameters(self):
        """Verify model has strictly zero trainable parameters."""
        self.assertEqual(self.projector.num_trainable_parameters, 0)
        self.assertEqual(self.projector.num_parameters, 0)

    def test_empty_parameters_iterator(self):
        """Verify list(model.parameters()) is empty (all tensors are persistent buffers)."""
        params = list(self.projector.parameters())
        self.assertEqual(len(params), 0)

    def test_buffers_have_requires_grad_false(self):
        """Verify all registered buffers have requires_grad=False."""
        for name, buf in self.projector.named_buffers():
            self.assertFalse(buf.requires_grad, f"Buffer {name} has requires_grad=True")

    def test_backward_does_not_populate_buffer_grads(self):
        """Verify loss backward pass leaves zero gradients on projector buffers."""
        x = torch.randn(8, self.in_dim, requires_grad=True)
        z = self.projector(x)
        loss = (z ** 2).sum()
        loss.backward()
        for name, buf in self.projector.named_buffers():
            self.assertIsNone(buf.grad, f"Buffer {name} received a gradient")

    def test_optimizer_compatibility(self):
        """Verify PyTorch optimizer raises ValueError when passed model.parameters() because list is empty."""
        with self.assertRaises(ValueError):
            _ = torch.optim.Adam(self.projector.parameters(), lr=1e-3)

    # -------------------------------------------------------------
    # 5. REPRODUCIBILITY & TRAINING-DATA-ONLY NORMALIZATION TESTS
    # -------------------------------------------------------------
    def test_training_data_only_isolation(self):
        """Verify test data evaluation does NOT alter stored training statistics."""
        initial_mean_64 = self.projector.mean_64.clone()
        initial_mean_6 = self.projector.mean_6.clone()
        initial_std_6 = self.projector.std_6.clone()

        # Pass completely different distribution (e.g., mean -100, std 50)
        test_data = torch.randn(100, self.in_dim) * 50.0 - 100.0
        _ = self.projector(test_data)

        self.assertTrue(torch.equal(self.projector.mean_64, initial_mean_64))
        self.assertTrue(torch.equal(self.projector.mean_6, initial_mean_6))
        self.assertTrue(torch.equal(self.projector.std_6, initial_std_6))

    def test_unfitted_raises_runtime_error(self):
        """Verify calling forward before fit raises RuntimeError for normalization."""
        unfitted = FixedPatchProjector(norm_type="standard")
        x = torch.randn(4, self.in_dim)
        with self.assertRaises(RuntimeError):
            _ = unfitted(x)

    def test_state_dict_save_and_restore(self):
        """Verify saving state_dict and restoring to a new module reproduces exact output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "projector_state.pt"
            torch.save(self.projector.state_dict(), save_path)

            loaded_p = FixedPatchProjector(seed=999)  # different initial seed
            loaded_p.load_state_dict(torch.load(save_path, weights_only=True))

            x = torch.randn(16, self.in_dim)
            z_orig = self.projector(x)
            z_loaded = loaded_p(x)
            self.assertTrue(torch.equal(z_orig, z_loaded))

    def test_semi_orthogonality(self):
        """Verify row semi-orthogonality: P @ P.T = I_6."""
        p = self.projector.projection_matrix
        identity_6 = torch.eye(self.out_dim)
        product = p @ p.t()
        self.assertTrue(torch.allclose(product, identity_6, atol=1e-6))

    def test_normalization_modes(self):
        """Verify standard, minmax, and angle normalization ranges."""
        p_std = FixedPatchProjector(norm_type="standard").fit(self.train_data)
        p_mm = FixedPatchProjector(norm_type="minmax").fit(self.train_data)
        p_ang = FixedPatchProjector(norm_type="angle").fit(self.train_data)

        z_std = p_std(self.train_data)
        self.assertTrue(torch.allclose(z_std.mean(0), torch.zeros(self.out_dim), atol=1e-4))
        self.assertTrue(torch.allclose(z_std.std(0, unbiased=False), torch.ones(self.out_dim), atol=1e-4))

        z_mm = p_mm(self.train_data)
        self.assertGreaterEqual(z_mm.min().item(), -1.0 - 1e-5)
        self.assertLessEqual(z_mm.max().item(), 1.0 + 1e-5)

        z_ang = p_ang(self.train_data)
        self.assertGreaterEqual(z_ang.min().item(), -math.pi - 1e-5)
        self.assertLessEqual(z_ang.max().item(), math.pi + 1e-5)

    # -------------------------------------------------------------
    # 6. METHODOLOGICAL LOSSINESS & SUBSPACE RECONSTRUCTION TESTS
    # -------------------------------------------------------------
    def test_arbitrary_64d_reconstruction_is_lossy_not_exact(self):
        """Explicitly verify that 64-D -> 6-D projection is lossy and arbitrary 64-D patches
        cannot be exactly reconstructed."""
        p_raw = FixedPatchProjector(norm_type="none")
        x_arbitrary = torch.randn(64, self.in_dim)  # Full rank 64-D vectors

        # Project and back-project
        z = p_raw.project(x_arbitrary)
        x_recon = p_raw.reconstruct(z)

        # Reconstruction error must be strictly non-zero and substantial
        recon_mse = torch.mean((x_arbitrary - x_recon) ** 2).item()
        self.assertGreater(
            recon_mse,
            0.5,
            f"Expected substantial reconstruction error due to lossy 64->6 compression, got {recon_mse}",
        )

        # However, re-projecting x_recon must recover the exact same 6-D vector z (idempotency)
        z_from_recon = p_raw.project(x_recon)
        self.assertTrue(
            torch.allclose(z, z_from_recon, atol=1e-5),
            "Re-projecting reconstructed patch must yield identical 6-D coordinates (idempotency of P^T P)",
        )

    def test_subspace_vectors_reconstruct_faithfully(self):
        """Verify that vectors lying strictly within the 6-D projection subspace
        are reconstructed faithfully with near-zero error."""
        p_raw = FixedPatchProjector(norm_type="none")
        v = torch.randn(20, self.out_dim)
        # Construct vectors lying strictly in the row span of P
        x_subspace = v @ p_raw.projection_matrix

        z = p_raw.project(x_subspace)
        x_rec = p_raw.reconstruct(z)
        self.assertTrue(
            torch.allclose(x_subspace, x_rec, atol=1e-5),
            "Vectors in the 6-D projection subspace must be reconstructed faithfully",
        )


if __name__ == "__main__":
    unittest.main()
