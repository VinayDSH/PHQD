"""
Module 5 EPQD Diagnostics.

Implements:
1. compute_von_neumann_entropy: Entanglement entropy S(rho_A) = -Tr(rho_A ln rho_A)
   across boundary registers to measure cross-patch quantum information transfer.
   Uses Schmidt/SVD decomposition on the pure statevector to avoid excessive density matrix allocation.
2. evaluate_bridge_zeroing_ablation: Clamps RZZ bridge weights to zero to measure
   functional contribution Delta L_bridge = L_val(phi=0) - L_val(phi*).
3. compute_seam_disagreement: Spatial boundary discord D_seam across overlapping patch margins.
"""

from typing import Dict, Any, Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from src.losses.seam_consistency import SeamConsistencyLoss
from src.models.hqcld_denoiser import sinusoidal_timestep_embedding


def _entropy_from_pure_statevector(state: np.ndarray, dim_a: int, dim_b: int) -> float:
    """Compute S(rho_A) = -Tr(rho_A ln rho_A) via Schmidt singular value decomposition.
    
    For pure state |psi> = sum_ij M_ij |i>_A |j>_B, the eigenvalues of reduced density matrix
    rho_A = M M^dagger are sigma_i^2 where sigma_i are singular values of M.
    This avoids forming the massive 2^N x 2^N full density matrix.
    """
    M = state.reshape(dim_a, dim_b)
    singular_values = np.linalg.svd(M, compute_uv=False)
    eigenvalues = singular_values ** 2
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def compute_von_neumann_entropy(
    model: nn.Module,
    x: torch.Tensor,
    t: Union[int, torch.Tensor],
    dev: Optional[Any] = None,
) -> float:
    """Compute mean Von Neumann entanglement entropy across adjacent patch registers.
    
    For k=2: Computes S(rho_A) across wires 0-5 (dim_A = 64) and wires 6-11 (dim_B = 64).
    For k=4: Computes S(rho_TL) across wires 0-3 (dim_A = 16) and wires 4-15 (dim_B = 4096).
    For k=1: Returns 0.0 (independent factorized registers).
    """
    if model.k == 1 or model.bridge_weights is None:
        return 0.0

    b = x.size(0)
    device = x.device

    if isinstance(t, int):
        t_tensor = torch.full((b,), t, dtype=torch.long, device=device)
    elif isinstance(t, torch.Tensor):
        if t.dim() == 0:
            t_tensor = t.expand(b).to(device=device, dtype=torch.long)
        else:
            t_tensor = t.to(device=device, dtype=torch.long)
    else:
        raise TypeError(f"Timestep must be int or Tensor, got {type(t)}")

    t_emb = sinusoidal_timestep_embedding(t_tensor, dim=model.patch_dim)
    x_cond = x + t_emb.unsqueeze(1)
    angles = model._encode_to_angles(x_cond.reshape(b * 9, model.patch_dim)).reshape(b, 9, model.n_qubits)

    if model.k == 2:
        total_qubits = 12
        if dev is None:
            dev = qml.device("default.qubit", wires=total_qubits)

        @qml.qnode(dev)
        def state_circuit(inp, w_loc, w_br):
            for i in range(total_qubits):
                qml.RY(inp[i], wires=i)
            for l in range(model.n_layers):
                for i in range(total_qubits):
                    qml.RY(w_loc[l, i], wires=i)
                for i in range(5):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(6, 11):
                    qml.CNOT(wires=[i, i + 1])
                qml.IsingZZ(w_br[l], wires=[5, 6])
            return qml.state()

        inp = torch.cat([angles[0, 0], angles[0, 1]], dim=-1).detach().cpu().numpy()
        w_loc = model.weights.detach().cpu().numpy()
        w_br = model.bridge_weights.detach().cpu().numpy()
        psi = state_circuit(inp, w_loc, w_br)
        return _entropy_from_pure_statevector(psi, dim_a=2**6, dim_b=2**6)

    elif model.k == 4:
        total_qubits = 16
        if dev is None:
            dev = qml.device("default.qubit", wires=total_qubits)

        @qml.qnode(dev)
        def state_circuit_quad(inp, w_loc, w_br):
            for i in range(total_qubits):
                qml.RY(inp[i], wires=i)
            for l in range(model.n_layers):
                for i in range(total_qubits):
                    qml.RY(w_loc[l, i], wires=i)
                for r in range(4):
                    start = r * model.n_qubits
                    for i in range(start, start + model.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                qml.IsingZZ(w_br[l, 0], wires=[3, 4])
                qml.IsingZZ(w_br[l, 1], wires=[2, 8])
                qml.IsingZZ(w_br[l, 2], wires=[11, 12])
                qml.IsingZZ(w_br[l, 3], wires=[6, 14])
            return qml.state()

        inp = torch.cat([angles[0, 0], angles[0, 1], angles[0, 3], angles[0, 4]], dim=-1).detach().cpu().numpy()
        w_loc = model.weights.detach().cpu().numpy()
        w_br = model.bridge_weights.detach().cpu().numpy()
        psi = state_circuit_quad(inp, w_loc, w_br)
        return _entropy_from_pure_statevector(psi, dim_a=2**4, dim_b=2**12)

    return 0.0


def evaluate_bridge_zeroing_ablation(
    model: nn.Module,
    x: torch.Tensor,
    t: Union[int, torch.Tensor],
    target_clean: Optional[torch.Tensor] = None,
) -> float:
    """Compute Delta L_bridge = L_val(phi=0) - L_val(phi*)."""
    if model.k == 1 or model.bridge_weights is None:
        return 0.0

    target = target_clean if target_clean is not None else x

    with torch.no_grad():
        # 1. Nominal prediction with trained bridge weights phi*
        pred_nominal = model(x, t)
        loss_nominal = F.mse_loss(pred_nominal, target).item()

        # 2. Prediction with bridge weights clamped to 0
        original_bridge = model.bridge_weights.clone()
        model.bridge_weights.zero_()
        pred_zeroed = model(x, t)
        loss_zeroed = F.mse_loss(pred_zeroed, target).item()

        # Restore original bridge weights
        model.bridge_weights.copy_(original_bridge)

    delta_loss = loss_zeroed - loss_nominal
    return float(delta_loss)


def compute_seam_disagreement(patches: torch.Tensor) -> float:
    """Compute mean spatial discord across overlapping boundaries of adjacent patches."""
    if patches.dim() == 3:
        patches_reshaped = patches.reshape(-1, 9, 4, 4, 4)
    elif patches.dim() == 5:
        patches_reshaped = patches
    else:
        raise ValueError(f"Expected 3D or 5D patches tensor, got {patches.dim()}D")

    loss_fn = SeamConsistencyLoss(num_patches=9, channels=4, patch_size=4, overlap=2)
    return float(loss_fn(patches_reshaped).item())
