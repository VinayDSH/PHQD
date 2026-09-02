"""
Parameterized Quantum Circuit (PQC) models for PHQD Module 2.

Provides:
- create_pqc_circuit: QNode building angle encoding and hardware-efficient ansatz.
- PQCDenoiser: TorchLayer-wrapped PQC module for vector denoising.
- ClassicalToyBaseline: Minimal classical MLP baseline for parameter-scale comparison.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import pennylane as qml


def create_pqc_circuit(
    n_qubits: int = 6,
    n_layers: int = 2,
    dev: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Tuple[int, ...]]]:
    """Create a PennyLane QNode implementing angle encoding and a hardware-efficient ansatz.

    Architecture:
    - Angle/Y encoding on all qubits: RY(inputs[..., i]) on wire i.
    - n_layers variational layers:
        - Parametric RY rotations on all qubits: RY(weights[l, i]) on wire i.
        - Nearest-neighbor CNOT entanglement: CNOT(wire i -> wire i+1).
    - Measurement: Pauli-Z expectation values on all qubits: <Z_i> in [-1, 1].

    Args:
        n_qubits: Number of qubits (wires). Defaults to 6.
        n_layers: Circuit depth (number of variational layers). Defaults to 2.
        dev: Optional PennyLane device. Defaults to 'default.qubit'.

    Returns:
        Tuple of (qnode, weight_shapes_dict).
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Angle/Y encoding
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)

        # Hardware-efficient variational ansatz
        for l in range(n_layers):
            # Parametric single-qubit RY rotations
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            # Nearest-neighbor CNOT entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Pauli-Z expectation measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes


class PQCDenoiser(nn.Module):
    """PQC Denoiser module integrating a variational quantum circuit with PyTorch autograd.

    Interface:
        noisy_vector [B, n_qubits]
            -> optional input scaling
            -> angle encoding (RY)
            -> hardware-efficient variational ansatz
            -> Pauli-Z expectation measurements
            -> denoised_vector [B, n_qubits] with values in [-1, 1]
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        input_scale: float = 1.0,
        dev: Optional[Any] = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_scale = input_scale

        qnode, weight_shapes = create_pqc_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            dev=dev,
        )
        self.torch_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable quantum parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise input vector x using the parameterized quantum circuit.

        Args:
            x: Noisy input tensor of shape [B, n_qubits] or [n_qubits].

        Returns:
            Denoised vector of shape [B, n_qubits] or [n_qubits] with values in [-1, 1].
        """
        scaled_inputs = x * self.input_scale if self.input_scale != 1.0 else x
        return self.torch_layer(scaled_inputs)


class ClassicalToyBaseline(nn.Module):
    """Minimal classical MLP baseline at a comparable parameter scale for comparison.

    For in_dim=6, hidden_dim=1 without bias, total parameters = 6 + 6 = 12,
    matching the exact parameter count of a 6-qubit 2-layer PQC.
    """

    def __init__(
        self,
        dim: int = 6,
        hidden_dim: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Tanh(),  # Bounds output to [-1, 1] matching Pauli-Z expectation range
        )

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable classical parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise input vector x using classical MLP.

        Args:
            x: Noisy input tensor of shape [B, dim] or [dim].

        Returns:
            Denoised vector of shape [B, dim] or [dim].
        """
        return self.net(x)
