"""
Module 5: Entangled Patch Quantum Diffusion (EPQD) Models & Circuits.

Implements:
- create_epqd_single_circuit: 6-qubit PQC for independent patch processing (k=1).
- create_epqd_pair_circuit: 12-qubit PQC with boundary IsingZZ bridge gate (k=2).
- create_epqd_quad_circuit: 16-qubit PQC with 4 IsingZZ bridge gates for 2x2 patch quads (k=4, nq=4).
- consensus_combine_quads: Arithmetic mean consensus combination for overlapping quads.
- EPQDDenoiser: Unified neural module for k in {1, 2, 4} with exact parameter locking.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from src.models.hqcld_denoiser import sinusoidal_timestep_embedding


# ---------------------------------------------------------------------------
# Quantum Circuit Constructors
# ---------------------------------------------------------------------------

def create_epqd_single_circuit(
    n_qubits: int = 6,
    n_layers: int = 3,
    dev: Optional[Any] = None,
) -> Any:
    """Create a 6-qubit single-patch PQC (k=1)."""

    if dev is None:
        dev = qml.device("lightning.gpu", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):

        # Angle encoding
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)

        # Variational layers
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)

            # Nearest-neighbor CNOT chain
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return [
            qml.expval(qml.PauliZ(i))
            for i in range(n_qubits)
        ]

    return circuit


def create_epqd_pair_circuit(
    n_qubits: int = 6,
    n_layers: int = 3,
    dev: Optional[Any] = None,
) -> Any:
    """Create a 12-qubit joint pair PQC (k=2).

    Register A: wires {0..5}
    Register B: wires {6..11}

    Locked bridge:
        Exactly 1 IsingZZ gate per layer on boundary wires (5, 6)
    """

    total_qubits = n_qubits * 2  # 12

    if dev is None:
        dev = qml.device("lightning.gpu", wires=total_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, bridge_weights):

        # 1. Angle encoding on all 12 qubits
        for i in range(total_qubits):
            qml.RY(inputs[..., i], wires=i)

        # 2. Variational layers
        for l in range(n_layers):

            # Local RY rotations
            for i in range(total_qubits):
                qml.RY(weights[l, i], wires=i)

            # Nearest-neighbor CNOT within Register A
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Nearest-neighbor CNOT within Register B
            for i in range(n_qubits, total_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Locked IsingZZ bridge gate
            qml.IsingZZ(
                bridge_weights[l],
                wires=[n_qubits - 1, n_qubits],
            )

        return [
            qml.expval(qml.PauliZ(i))
            for i in range(total_qubits)
        ]

    return circuit


def create_epqd_quad_circuit(
    n_qubits: int = 4,
    n_layers: int = 3,
    dev: Optional[Any] = None,
) -> Any:
    """Create a 16-qubit joint quad PQC (k=4, nq=4).

    Registers:
      R0 (Top-Left):     wires {0..3}
      R1 (Top-Right):    wires {4..7}
      R2 (Bottom-Left):  wires {8..11}
      R3 (Bottom-Right): wires {12..15}

    Locked bridge gates:
      Bridge 1 (R0 <-> R1): wires (3, 4)
      Bridge 2 (R0 <-> R2): wires (2, 8)
      Bridge 3 (R2 <-> R3): wires (11, 12)
      Bridge 4 (R1 <-> R3): wires (6, 14)
    """

    total_qubits = n_qubits * 4  # 16

    if dev is None:
        dev = qml.device("lightning.gpu", wires=total_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, bridge_weights):

        # 1. Angle encoding on all 16 qubits
        for i in range(total_qubits):
            qml.RY(inputs[..., i], wires=i)

        # 2. Variational layers
        for l in range(n_layers):

            # Local RY rotations
            for i in range(total_qubits):
                qml.RY(weights[l, i], wires=i)

            # Intra-register CNOTs
            for r in range(4):
                start = r * n_qubits

                for i in range(
                    start,
                    start + n_qubits - 1
                ):
                    qml.CNOT(wires=[i, i + 1])

            # Four locked IsingZZ bridge gates
            qml.IsingZZ(
                bridge_weights[l, 0],
                wires=[3, 4],
            )

            qml.IsingZZ(
                bridge_weights[l, 1],
                wires=[2, 8],
            )

            qml.IsingZZ(
                bridge_weights[l, 2],
                wires=[11, 12],
            )

            qml.IsingZZ(
                bridge_weights[l, 3],
                wires=[6, 14],
            )

        return [
            qml.expval(qml.PauliZ(i))
            for i in range(total_qubits)
        ]

    return circuit


# ---------------------------------------------------------------------------
# Consensus Combining for k=4
# ---------------------------------------------------------------------------

def consensus_combine_quads(
    quad_predictions: torch.Tensor,
) -> torch.Tensor:
    """Combine 4 overlapping quad predictions into a 3x3 patch grid.

    Args:
        quad_predictions:
            Tensor of shape [B, 4, 4, patch_dim]

        Index 0:
            QTL = (P0, P1, P3, P4)

        Index 1:
            QTR = (P1, P2, P4, P5)

        Index 2:
            QBL = (P3, P4, P6, P7)

        Index 3:
            QBR = (P4, P5, P7, P8)

    Returns:
        Tensor of shape [B, 9, patch_dim]
        with arithmetic mean consensus.
    """

    b, num_quads, patches_per_quad, patch_dim = (
        quad_predictions.shape
    )

    device = quad_predictions.device
    dtype = quad_predictions.dtype

    pred_accum = torch.zeros(
        b,
        9,
        patch_dim,
        device=device,
        dtype=dtype,
    )

    counts = torch.zeros(
        9,
        device=device,
        dtype=dtype,
    )

    # QTL = (P0, P1, P3, P4)
    pred_accum[:, 0] += quad_predictions[:, 0, 0]
    pred_accum[:, 1] += quad_predictions[:, 0, 1]
    pred_accum[:, 3] += quad_predictions[:, 0, 2]
    pred_accum[:, 4] += quad_predictions[:, 0, 3]

    counts[[0, 1, 3, 4]] += 1.0

    # QTR = (P1, P2, P4, P5)
    pred_accum[:, 1] += quad_predictions[:, 1, 0]
    pred_accum[:, 2] += quad_predictions[:, 1, 1]
    pred_accum[:, 4] += quad_predictions[:, 1, 2]
    pred_accum[:, 5] += quad_predictions[:, 1, 3]

    counts[[1, 2, 4, 5]] += 1.0

    # QBL = (P3, P4, P6, P7)
    pred_accum[:, 3] += quad_predictions[:, 2, 0]
    pred_accum[:, 4] += quad_predictions[:, 2, 1]
    pred_accum[:, 6] += quad_predictions[:, 2, 2]
    pred_accum[:, 7] += quad_predictions[:, 2, 3]

    counts[[3, 4, 6, 7]] += 1.0

    # QBR = (P4, P5, P7, P8)
    pred_accum[:, 4] += quad_predictions[:, 3, 0]
    pred_accum[:, 5] += quad_predictions[:, 3, 1]
    pred_accum[:, 7] += quad_predictions[:, 3, 2]
    pred_accum[:, 8] += quad_predictions[:, 3, 3]

    counts[[4, 5, 7, 8]] += 1.0

    return pred_accum / counts.unsqueeze(0).unsqueeze(-1)


# ---------------------------------------------------------------------------
# EPQD Denoiser Module
# ---------------------------------------------------------------------------

class EPQDDenoiser(nn.Module):
    """Unified Entangled Patch Quantum Diffusion (EPQD) Denoiser.

    Supports:

        k=1:
            9 independent patches on 6-qubit PQC
            786 trainable parameters.

        k=2:
            4 pairs on 12-qubit PQC
            + 1 single remainder on 6-qubit PQC
            807 trainable parameters.

        k=4:
            4 overlapping 2x2 quads on 16-qubit PQC
            nq=4
            572 trainable parameters.
    """

    def __init__(
        self,
        k: int = 2,
        n_qubits: Optional[int] = None,
        n_layers: int = 3,
        patch_dim: int = 64,
        num_patches: int = 9,
        dev: Optional[Any] = None,
    ):
        super().__init__()

        if k not in (1, 2, 4):
            raise ValueError(
                f"Unsupported grouping scale k={k}. "
                f"Allowed values are {{1, 2, 4}}."
            )

        self.k = int(k)
        self.n_layers = int(n_layers)
        self.patch_dim = int(patch_dim)
        self.num_patches = int(num_patches)

        # Default number of qubits per patch
        if n_qubits is None:
            self.n_qubits = 4 if self.k == 4 else 6
        else:
            self.n_qubits = int(n_qubits)

        # k=4 empirical experiment is locked to 16 qubits
        if self.k == 4 and self.n_qubits != 4:
            raise ValueError(
                "k=4 empirical execution strictly requires "
                "n_qubits=4 (16 qubits total). "
                f"Got n_qubits={self.n_qubits}."
            )

        # k=1 and k=2 use 6 qubits per patch
        if self.k in (1, 2) and self.n_qubits != 6:
            raise ValueError(
                f"k={self.k} execution requires n_qubits=6. "
                f"Got n_qubits={self.n_qubits}."
            )

        # Shared classical linear adapters
        self.encoder = nn.Linear(
            self.patch_dim,
            self.n_qubits,
            bias=False,
        )

        self.decoder = nn.Linear(
            self.n_qubits,
            self.patch_dim,
            bias=False,
        )

        # ---------------------------------------------------------------
        # Initialize PQC core parameters and circuits
        # ---------------------------------------------------------------

        if self.k == 1:

            self.pqc_circuit = create_epqd_single_circuit(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                dev=dev,
            )

            # 6 * 3 = 18 parameters
            self.weights = nn.Parameter(
                torch.empty(
                    self.n_layers,
                    self.n_qubits,
                )
            )

            self.bridge_weights = None

            nn.init.uniform_(
                self.weights,
                -math.pi,
                math.pi,
            )

        elif self.k == 2:

            self.pair_circuit = create_epqd_pair_circuit(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                dev=dev,
            )

            self.single_circuit = create_epqd_single_circuit(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                dev=dev,
            )

            # Local parameters:
            # 12 qubits * 3 layers = 36
            self.weights = nn.Parameter(
                torch.empty(
                    self.n_layers,
                    self.n_qubits * 2,
                )
            )

            # Bridge parameters:
            # 1 IsingZZ per layer = 3
            self.bridge_weights = nn.Parameter(
                torch.empty(self.n_layers)
            )

            nn.init.uniform_(
                self.weights,
                -math.pi,
                math.pi,
            )

            nn.init.uniform_(
                self.bridge_weights,
                -math.pi,
                math.pi,
            )

        elif self.k == 4:

            self.quad_circuit = create_epqd_quad_circuit(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                dev=dev,
            )

            # Local parameters:
            # 16 qubits * 3 layers = 48
            self.weights = nn.Parameter(
                torch.empty(
                    self.n_layers,
                    self.n_qubits * 4,
                )
            )

            # Bridge parameters:
            # 4 IsingZZ gates * 3 layers = 12
            self.bridge_weights = nn.Parameter(
                torch.empty(
                    self.n_layers,
                    4,
                )
            )

            nn.init.uniform_(
                self.weights,
                -math.pi,
                math.pi,
            )

            nn.init.uniform_(
                self.bridge_weights,
                -math.pi,
                math.pi,
            )

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    @property
    def num_pqc_parameters(self) -> int:
        """Total trainable parameters in the PQC module."""

        count = self.weights.numel()

        if self.bridge_weights is not None:
            count += self.bridge_weights.numel()

        return count

    @property
    def num_adapter_parameters(self) -> int:
        """Total trainable parameters in classical adapters."""

        return sum(
            p.numel()
            for p in [
                self.encoder.weight,
                self.decoder.weight,
            ]
        )

    def _encode_to_angles(
        self,
        x_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Project patch representation into [-pi, pi]."""

        z = self.encoder(x_flat)

        return math.pi * torch.tanh(z)

    def _decode_from_expectations(
        self,
        z_denoised: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct clean patch representation."""

        return self.decoder(z_denoised)

    def forward(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Forward denoiser pass.

        Input:
            [B, 9, 64]

        Output:
            [B, 9, 64]
        """

        b = x.size(0)
        device = x.device

        # ---------------------------------------------------------------
        # 1. Format timesteps
        # ---------------------------------------------------------------

        if isinstance(t, int):

            t_tensor = torch.full(
                (b,),
                t,
                dtype=torch.long,
                device=device,
            )

        elif isinstance(t, torch.Tensor):

            if t.dim() == 0:
                t_tensor = t.expand(b).to(
                    device=device,
                    dtype=torch.long,
                )
            else:
                t_tensor = t.to(
                    device=device,
                    dtype=torch.long,
                )

        else:
            raise TypeError(
                f"Timestep must be int or torch.Tensor, "
                f"got {type(t).__name__}"
            )

        # ---------------------------------------------------------------
        # 2. Diffusion timestep conditioning
        # ---------------------------------------------------------------

        t_emb = sinusoidal_timestep_embedding(
            t_tensor,
            dim=self.patch_dim,
        )

        x_cond = x + t_emb.unsqueeze(1)

        # ---------------------------------------------------------------
        # 3. Mode-specific quantum processing
        # ---------------------------------------------------------------

        if self.k == 1:
            return self._forward_k1(
                x_cond,
                b,
                device,
            )

        elif self.k == 2:
            return self._forward_k2(
                x_cond,
                b,
                device,
            )

        elif self.k == 4:
            return self._forward_k4(
                x_cond,
                b,
                device,
            )

        raise RuntimeError("Invalid k value.")

    def _forward_k1(
        self,
        x_cond: torch.Tensor,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        """k=1: independent processing of 9 patches."""

        x_flat = x_cond.reshape(
            b * 9,
            self.patch_dim,
        )

        angles = self._encode_to_angles(
            x_flat
        )

        # Evaluate 6-qubit PQC
        raw_res = self.pqc_circuit(
            angles,
            self.weights,
        )

        z_denoised = torch.stack(
            raw_res,
            dim=-1,
        ).to(
            device=device,
            dtype=x_cond.dtype,
        )

        x_pred_flat = self._decode_from_expectations(
            z_denoised
        )

        return x_pred_flat.reshape(
            b,
            9,
            self.patch_dim,
        )

    def _forward_k2(
        self,
        x_cond: torch.Tensor,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        """k=2: 4 joint pairs + 1 single remainder."""

        # Encode all patches
        angles = self._encode_to_angles(
            x_cond.reshape(
                b * 9,
                self.patch_dim,
            )
        ).reshape(
            b,
            9,
            self.n_qubits,
        )

        # ---------------------------------------------------------------
        # Assemble pairs
        # ---------------------------------------------------------------

        # Pair 0 = (P0, P1)
        pair_0 = torch.cat(
            [
                angles[:, 0],
                angles[:, 1],
            ],
            dim=-1,
        )

        # Pair 1 = (P3, P4)
        pair_1 = torch.cat(
            [
                angles[:, 3],
                angles[:, 4],
            ],
            dim=-1,
        )

        # Pair 2 = (P6, P7)
        pair_2 = torch.cat(
            [
                angles[:, 6],
                angles[:, 7],
            ],
            dim=-1,
        )

        # Pair 3 = (P2, P5)
        pair_3 = torch.cat(
            [
                angles[:, 2],
                angles[:, 5],
            ],
            dim=-1,
        )

        pairs_input = torch.stack(
            [
                pair_0,
                pair_1,
                pair_2,
                pair_3,
            ],
            dim=1,
        ).reshape(
            4 * b,
            12,
        )

        # ---------------------------------------------------------------
        # 12-qubit pair simulation
        # ---------------------------------------------------------------

        pair_res = self.pair_circuit(
            pairs_input,
            self.weights,
            self.bridge_weights,
        )

        pair_z = torch.stack(
            pair_res,
            dim=-1,
        ).to(
            device=device,
            dtype=x_cond.dtype,
        ).reshape(
            b,
            4,
            12,
        )

        # ---------------------------------------------------------------
        # Remainder patch P8
        # ---------------------------------------------------------------

        p8_input = angles[:, 8]

        p8_res = self.single_circuit(
            p8_input,
            self.weights[:, :6],
        )

        p8_z = torch.stack(
            p8_res,
            dim=-1,
        ).to(
            device=device,
            dtype=x_cond.dtype,
        )

        # ---------------------------------------------------------------
        # Map pair outputs back to 9 patches
        # ---------------------------------------------------------------

        z_out = torch.zeros(
            b,
            9,
            self.n_qubits,
            device=device,
            dtype=x_cond.dtype,
        )

        # Pair 0 -> P0, P1
        z_out[:, 0] = pair_z[:, 0, :6]
        z_out[:, 1] = pair_z[:, 0, 6:]

        # Pair 1 -> P3, P4
        z_out[:, 3] = pair_z[:, 1, :6]
        z_out[:, 4] = pair_z[:, 1, 6:]

        # Pair 2 -> P6, P7
        z_out[:, 6] = pair_z[:, 2, :6]
        z_out[:, 7] = pair_z[:, 2, 6:]

        # Pair 3 -> P2, P5
        z_out[:, 2] = pair_z[:, 3, :6]
        z_out[:, 5] = pair_z[:, 3, 6:]

        # Remainder -> P8
        z_out[:, 8] = p8_z

        # ---------------------------------------------------------------
        # Decode
        # ---------------------------------------------------------------

        x_pred_flat = self._decode_from_expectations(
            z_out.reshape(
                b * 9,
                self.n_qubits,
            )
        )

        return x_pred_flat.reshape(
            b,
            9,
            self.patch_dim,
        )

    def _forward_k4(
        self,
        x_cond: torch.Tensor,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        """k=4: 4 overlapping 2x2 quads on 16-qubit PQC."""

        # ---------------------------------------------------------------
        # Encode all patches
        # ---------------------------------------------------------------

        angles = self._encode_to_angles(
            x_cond.reshape(
                b * 9,
                self.patch_dim,
            )
        ).reshape(
            b,
            9,
            self.n_qubits,
        )

        # ---------------------------------------------------------------
        # Assemble 4 overlapping quads
        # ---------------------------------------------------------------

        # QTL = (P0, P1, P3, P4)
        q_tl = torch.cat(
            [
                angles[:, 0],
                angles[:, 1],
                angles[:, 3],
                angles[:, 4],
            ],
            dim=-1,
        )

        # QTR = (P1, P2, P4, P5)
        q_tr = torch.cat(
            [
                angles[:, 1],
                angles[:, 2],
                angles[:, 4],
                angles[:, 5],
            ],
            dim=-1,
        )

        # QBL = (P3, P4, P6, P7)
        q_bl = torch.cat(
            [
                angles[:, 3],
                angles[:, 4],
                angles[:, 6],
                angles[:, 7],
            ],
            dim=-1,
        )

        # QBR = (P4, P5, P7, P8)
        q_br = torch.cat(
            [
                angles[:, 4],
                angles[:, 5],
                angles[:, 7],
                angles[:, 8],
            ],
            dim=-1,
        )

        quads_input = torch.stack(
            [
                q_tl,
                q_tr,
                q_bl,
                q_br,
            ],
            dim=1,
        ).reshape(
            4 * b,
            16,
        )

        # ---------------------------------------------------------------
        # 16-qubit GPU quantum simulation
        # ---------------------------------------------------------------

        quad_res = self.quad_circuit(
            quads_input,
            self.weights,
            self.bridge_weights,
        )

        quad_z = torch.stack(
            quad_res,
            dim=-1,
        ).to(
            device=device,
            dtype=x_cond.dtype,
        ).reshape(
            b,
            4,
            4,
            self.n_qubits,
        )

        # ---------------------------------------------------------------
        # Decode each quad's 4 patches
        # ---------------------------------------------------------------

        quad_pred_flat = self._decode_from_expectations(
            quad_z.reshape(
                b * 4 * 4,
                self.n_qubits,
            )
        )

        quad_predictions = quad_pred_flat.reshape(
            b,
            4,
            4,
            self.patch_dim,
        )

        # ---------------------------------------------------------------
        # Arithmetic mean consensus
        # ---------------------------------------------------------------

        return consensus_combine_quads(
            quad_predictions
        )