"""
Experiment script for Module 2 Stage 1: Toy Gaussian Denoising with PQC.

Trains a PennyLane TorchLayer-wrapped PQC on a synthetic 6D Gaussian denoising regression task.
Optionally trains a comparable-parameter classical MLP baseline for debugging comparison.
Records training loss, validation loss, wall-clock time, parameter count, and logs results.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pqc import (
    PQCDenoiser,
    ClassicalToyBaseline,
    ParameterMatchedClassicalBaseline,
)
from src.data.toy_denoising import get_toy_denoising_dataloaders


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on a DataLoader using MSE criterion."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            total_loss += loss.item() * noisy.size(0)
            total_samples += noisy.size(0)

    return total_loss / max(total_samples, 1)


def train_toy_denoiser(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    lr: float = 0.05,
    device: Optional[torch.device] = None,
    model_name: str = "PQC",
) -> Dict[str, Any]:
    """Train a toy denoiser model with MSE loss and record metrics per epoch."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_time": [],
    }

    # Initial baseline evaluation (epoch 0)
    initial_val_loss = evaluate_model(model, val_loader, criterion, device)
    print(f"[{model_name}] Initial Val MSE (untrained): {initial_val_loss:.6f}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        sample_count = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)
            sample_count += noisy.size(0)

        epoch_train_loss = running_loss / max(sample_count, 1)
        epoch_val_loss = evaluate_model(model, val_loader, criterion, device)
        epoch_dur = time.time() - epoch_start

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["epoch_time"].append(epoch_dur)

        print(
            f"[{model_name}] Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train MSE: {epoch_train_loss:.6f} | "
            f"Val MSE: {epoch_val_loss:.6f} | "
            f"Time: {epoch_dur:.2f}s"
        )

    total_wall_time = time.time() - start_time
    final_val_loss = history["val_loss"][-1]

    return {
        "model_name": model_name,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": min(history["val_loss"]),
        "total_wall_time": total_wall_time,
        "history": history,
    }


def run_toy_denoising_experiment(
    qubits: int = 6,
    layers: int = 2,
    noise_std: float = 0.2,
    seed: int = 42,
    epochs: int = 10,
    lr: float = 0.05,
    batch_size: int = 32,
    num_train: int = 400,
    num_val: int = 100,
    compare_classical: bool = True,
    output_dir: str = "results/logs",
    device_name: Optional[str] = None,
    pqc_seed: Optional[int] = None,
    classical_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Orchestrate the toy denoising experiment, appending results to output_dir."""
    pqc_init_seed = pqc_seed if pqc_seed is not None else seed
    classical_init_seed = classical_seed if classical_seed is not None else (seed + 100)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        # PennyLane default.qubit statevector simulator operates natively on CPU
        device = torch.device("cpu")

    print("=" * 70)
    print("PHQD Module 2 Stage 1: Toy Gaussian Denoising Experiment")
    print(f"Device: {device} | Dataset Seed: {seed} | Vector Dim / Qubits: {qubits} | Depth: {layers}")
    print(f"PQC Init Seed: {pqc_init_seed} | Classical Init Seed: {classical_init_seed}")
    print(f"Noise Std: {noise_std} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"Samples: Train={num_train}, Val={num_val}")
    print("=" * 70)

    # 1. Prepare data loaders with controlled dataset seed
    set_seed(seed)
    train_loader, val_loader = get_toy_denoising_dataloaders(
        num_train=num_train,
        num_val=num_val,
        dim=qubits,
        noise_std=noise_std,
        batch_size=batch_size,
        seed=seed,
    )

    # 2. Train PQC Denoiser with independently controlled initialization seed
    set_seed(pqc_init_seed)
    pqc_model = PQCDenoiser(n_qubits=qubits, n_layers=layers)
    print(f"\n--- Training PQC Denoiser ({pqc_model.num_parameters} quantum parameters) ---")
    pqc_results = train_toy_denoiser(
        model=pqc_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        model_name="PQC",
    )

    results_summary: Dict[str, Any] = {
        "experiment": "toy_gaussian_denoising",
        "qubits": qubits,
        "qubit_count": qubits,
        "layers": layers,
        "circuit_depth": layers,
        "vector_dim": qubits,
        "noise_std": noise_std,
        "seed": seed,
        "pqc_init_seed": pqc_init_seed,
        "classical_init_seed": classical_init_seed if compare_classical else None,
        "model_initialization": {
            "pqc_init_seed": pqc_init_seed,
            "classical_init_seed": classical_init_seed if compare_classical else None,
        },
        "device": str(device),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "num_train": num_train,
        "num_val": num_val,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "pqc_results": pqc_results,
    }

    # 3. Optional Classical Baseline Comparison with independently controlled initialization seed
    if compare_classical:
        set_seed(classical_init_seed)
        classical_model = ParameterMatchedClassicalBaseline(dim=qubits, layers=layers)
        print(f"\n--- Training Parameter-Matched Classical Baseline ({classical_model.num_parameters} classical parameters) ---")
        classical_results = train_toy_denoiser(
            model=classical_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            model_name="Classical_Circulant_MLP",
        )
        results_summary["classical_results"] = classical_results

    # 4. Save results log, preserving previous runs instead of overwriting
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "toy_pqc_denoising_results.json"

    runs = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    runs = existing_data
                elif isinstance(existing_data, dict):
                    # Backward compatibility: preserve existing single-run dictionary
                    runs = [existing_data]
        except (json.JSONDecodeError, OSError):
            runs = []

    run_id = len(runs) + 1
    results_summary["run_id"] = run_id
    runs.append(results_summary)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    print("\n" + "=" * 70)
    print("Experiment Summary")
    print(f"Run ID:                  {run_id} (Total runs preserved: {len(runs)})")
    print(f"PQC Parameters:          {pqc_results['parameter_count']}")
    print(f"PQC Initial Val MSE:     {pqc_results['initial_val_loss']:.6f}")
    print(f"PQC Final Val MSE:       {pqc_results['final_val_loss']:.6f}")
    print(f"PQC Total Wall-Clock:    {pqc_results['total_wall_time']:.2f}s")
    if compare_classical:
        c_res = results_summary["classical_results"]
        print(f"Classical Parameters:    {c_res['parameter_count']}")
        print(f"Classical Final Val MSE: {c_res['final_val_loss']:.6f}")
        print(f"Classical Wall-Clock:    {c_res['total_wall_time']:.2f}s")
    print(f"Saved Results to:        {log_file}")
    print("=" * 70)

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy PQC Gaussian Denoising Experiment")
    parser.add_argument("--qubits", type=int, default=6, help="Number of qubits and vector dimension")
    parser.add_argument("--layers", type=int, default=2, help="Circuit depth (variational layers)")
    parser.add_argument("--noise-std", type=float, default=0.2, help="Gaussian noise std dev")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--train-samples", type=int, default=400, help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--no-classical", action="store_true", help="Skip classical baseline comparison")
    parser.add_argument("--output-dir", default="results/logs", help="Directory to save JSON results")
    parser.add_argument("--device", default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--pqc-seed", type=int, default=None, help="Independent seed for PQC initialization")
    parser.add_argument("--classical-seed", type=int, default=None, help="Independent seed for classical initialization")

    args = parser.parse_args()

    run_toy_denoising_experiment(
        qubits=args.qubits,
        layers=args.layers,
        noise_std=args.noise_std,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_train=args.train_samples,
        num_val=args.val_samples,
        compare_classical=not args.no_classical,
        output_dir=args.output_dir,
        device_name=args.device,
        pqc_seed=args.pqc_seed,
        classical_seed=args.classical_seed,
    )
