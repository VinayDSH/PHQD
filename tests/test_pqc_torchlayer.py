"""
Reproducible test for 6-qubit PennyLane TorchLayer gradient experiment.

Architecture:
- Wires: 6 qubits
- State Encoding: Angle encoding via RY rotations (inputs)
- Variational Layers: 2 layers, each consisting of:
    - Trainable RY rotations on all 6 qubits
    - Nearest-neighbor entangling CNOT gates (0->1, 1->2, 2->3, 3->4, 4->5)
- Measurement: Pauli-Z expectation values on all 6 qubits
- Integration: PennyLane TorchLayer for seamless PyTorch autograd compatibility
"""

import unittest
import torch
import pennylane as qml


def create_pqc_torchlayer(n_qubits: int = 6, n_layers: int = 2) -> qml.qnn.TorchLayer:
    """Instantiate a 6-qubit PQC wrapped in a PennyLane TorchLayer."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Angle/Y encoding
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)

        # Variational ansatz
        for l in range(n_layers):
            # Parametric single-qubit RY rotations
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            # Nearest-neighbor CNOT entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Pauli-Z expectation value measurements across all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class TestPQCTorchLayer(unittest.TestCase):
    """Test suite verifying dimension correctness and gradient flow of the PQC TorchLayer."""

    def setUp(self):
        # Ensure deterministic execution
        torch.manual_seed(42)
        self.n_qubits = 6
        self.n_layers = 2
        self.layer = create_pqc_torchlayer(self.n_qubits, self.n_layers)

    def test_output_dimension_batched(self):
        """Verify output dimension matches (batch_size, n_qubits) for batched input."""
        batch_size = 4
        x = torch.randn(batch_size, self.n_qubits)
        out = self.layer(x)

        expected_shape = torch.Size([batch_size, self.n_qubits])
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Expected output shape {expected_shape}, but got {out.shape}",
        )

    def test_output_dimension_single_sample(self):
        """Verify output dimension matches (n_qubits,) for 1D single-sample input."""
        x = torch.randn(self.n_qubits)
        out = self.layer(x)

        expected_shape = torch.Size([self.n_qubits])
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Expected output shape {expected_shape}, but got {out.shape}",
        )

    def test_backpropagation_gradients(self):
        """Verify that backpropagation produces non-None, non-zero, finite gradients."""
        batch_size = 4
        x = torch.randn(batch_size, self.n_qubits, requires_grad=True)

        out = self.layer(x)
        loss = out.sum()
        loss.backward()

        # Check circuit trainable weights gradient
        self.assertIsNotNone(
            self.layer.weights.grad,
            "Layer weights gradient is None after backward pass",
        )
        self.assertFalse(
            torch.isnan(self.layer.weights.grad).any(),
            "Layer weights gradient contains NaN values",
        )
        weights_grad_norm = self.layer.weights.grad.norm().item()
        self.assertGreater(
            weights_grad_norm,
            0.0,
            f"Expected non-zero weight gradient norm, but got {weights_grad_norm}",
        )

        # Check input tensor gradient
        self.assertIsNotNone(
            x.grad,
            "Input tensor gradient is None after backward pass",
        )
        self.assertFalse(
            torch.isnan(x.grad).any(),
            "Input tensor gradient contains NaN values",
        )
        input_grad_norm = x.grad.norm().item()
        self.assertGreater(
            input_grad_norm,
            0.0,
            f"Expected non-zero input gradient norm, but got {input_grad_norm}",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_execution_forward_backward(self):
        """Regression test verifying PQCDenoiser forward and backward pass on CUDA."""
        from src.models.pqc import PQCDenoiser

        device = torch.device("cuda:0")
        model = PQCDenoiser(n_qubits=self.n_qubits, n_layers=self.n_layers).to(device)

        batch_size = 4
        x = torch.randn(batch_size, self.n_qubits, device=device, requires_grad=True)

        out = model(x)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, torch.Size([batch_size, self.n_qubits]))
        self.assertTrue(torch.isfinite(out).all())

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(model.torch_layer.weights.grad)
        self.assertEqual(model.torch_layer.weights.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(model.torch_layer.weights.grad).all())
        self.assertGreater(model.torch_layer.weights.grad.norm().item(), 0.0)

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.device.type, "cuda")
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_optimization_step(self):
        """Regression test verifying an optimization step on CUDA updates weights with finite loss."""
        from src.models.pqc import PQCDenoiser

        device = torch.device("cuda:0")
        model = PQCDenoiser(n_qubits=self.n_qubits, n_layers=self.n_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = torch.nn.MSELoss()

        x = torch.randn(8, self.n_qubits, device=device)
        target = torch.randn(8, self.n_qubits, device=device)

        initial_loss = criterion(model(x), target).item()
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        final_loss = criterion(model(x), target).item()

        self.assertTrue(torch.isfinite(torch.tensor(final_loss)))
        self.assertNotEqual(initial_loss, final_loss)


if __name__ == "__main__":
    unittest.main(verbosity=2)
