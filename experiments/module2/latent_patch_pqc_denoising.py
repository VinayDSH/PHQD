"""
Experiment script for Module 2 Stage 2: Latent-Patch Denoising with PQC and Circulant Baseline.

Applies the fixed 64->6 semi-orthogonal projection and standard normalization pipeline
(fitted strictly on training patches) to Module 1 latent patches.
Trains:
1. PennyLane TorchLayer-wrapped PQC Denoiser (6 qubits, 2 variational layers, 12 quantum params).
2. Parameter-Matched Circulant Classical Baseline (dim 6, 2 circulant layers, 12 classical params).

Evaluation Protocol:
- Primary metric: 6-D projected-space denoising MSE.
- Secondary/diagnostic metric: 64-D pseudo-inverse reconstruction MSE.

Reproducibility & Fair Comparison Guarantees:
- Both models train on identical projected noisy/clean pairs.
- Dedicated seeded torch.Generator ensures identical batch ordering across epochs for both models.
- Independent initialization seeds are recorded separately for PQC and classical baseline.
- Both models use identical optimizer, learning rate, batch size, and epoch schedules.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pqc import (
    PQCDenoiser,
    ParameterMatchedClassicalBaseline,
)
from src.models.projection import FixedPatchProjector
from src.data.latent_patch_denoising import (
    LatentPatchDenoisingDataset,
    prepare_latent_patch_denoising_datasets,
    load_raw_patches,
    flatten_latent_patches,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_reproducible_train_loader(
    dataset: LatentPatchDenoisingDataset,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with a dedicated seeded Generator for reproducible shuffling."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=num_workers,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on a DataLoader using MSE criterion in 6-D projected space."""
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


def compute_secondary_64d_reconstruction_mse(
    model: nn.Module,
    dataloader: DataLoader,
    projector: FixedPatchProjector,
    device: torch.device,
) -> float:
    """Compute secondary/diagnostic 64-D pseudo-inverse reconstruction MSE.

    Maps denoised 6-D predictions and clean 6-D targets back to the 6-D projection subspace
    in 64-D via pseudo_inverse_back_project, and evaluates their MSE.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred_6d = model(noisy)

            # Reconstruct to 64-D subspace via pseudo-inverse
            pred_64d = projector.pseudo_inverse_back_project(pred_6d.cpu())
            clean_64d = projector.pseudo_inverse_back_project(clean.cpu())

            loss = F.mse_loss(pred_64d, clean_64d)
            total_loss += loss.item() * noisy.size(0)
            total_samples += noisy.size(0)

    return total_loss / max(total_samples, 1)


def train_denoiser(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.05,
    device: Optional[torch.device] = None,
    model_name: str = "PQC",
) -> Dict[str, Any]:
    """Train a denoiser model on 6-D pairs with MSE loss and record metrics per epoch."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_time": [],
    }

    initial_val_loss = evaluate_model(model, val_loader, criterion, device)
    print(f"[{model_name}] Initial Val 6-D MSE (untrained): {initial_val_loss:.6f}")

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
            f"Train 6-D MSE: {epoch_train_loss:.6f} | "
            f"Val 6-D MSE: {epoch_val_loss:.6f} | "
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


def run_latent_patch_denoising_experiment(
    train_patches: Optional[Union[torch.Tensor, str, Path]] = None,
    val_patches: Optional[Union[torch.Tensor, str, Path]] = None,
    train_patches_path: str = "data/latent_patches/train_patches.pt",
    val_patches_path: str = "data/latent_patches/val_patches.pt",
    qubits: int = 6,
    layers: int = 2,
    noise_std: float = 0.2,
    projector_seed: int = 42,
    dataset_seed: int = 42,
    dataloader_seed: int = 42,
    pqc_seed: Optional[int] = None,
    classical_seed: Optional[int] = None,
    epochs: int = 10,
    lr: float = 0.05,
    batch_size: int = 32,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    compare_classical: bool = True,
    output_dir: str = "results/logs",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate the Stage 2 latent-patch denoising experiment."""
    # 1. Resolve inputs
    if train_patches is None:
        p = Path(train_patches_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Training latent patches not found at: {p}. Export them first via "
                f"'python scripts/export_latent_patches.py' or provide train_patches tensor."
            )
        train_patches = p

    if val_patches is None:
        p = Path(val_patches_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Validation latent patches not found at: {p}. Export them first via "
                f"'python scripts/export_latent_patches.py' or provide val_patches tensor."
            )
        val_patches = p

    train_patches_tensor = load_raw_patches(train_patches)
    if max_train_samples is not None:
        train_patches_tensor = train_patches_tensor[:max_train_samples]

    val_patches_tensor = load_raw_patches(val_patches)
    if max_val_samples is not None:
        val_patches_tensor = val_patches_tensor[:max_val_samples]

    pqc_init_seed = pqc_seed if pqc_seed is not None else dataset_seed
    classical_init_seed = classical_seed if classical_seed is not None else (dataset_seed + 100)

    device = torch.device(device_name if device_name is not None else "cpu")

    # 2. Build Datasets with frozen projector fitted only on training data
    train_dataset, val_dataset, projector = prepare_latent_patch_denoising_datasets(
        train_patches=train_patches_tensor,
        val_patches=val_patches_tensor,
        noise_std=noise_std,
        projector_seed=projector_seed,
        dataset_seed=dataset_seed,
    )

    print("=" * 70)
    print("PHQD Module 2 Stage 2: Latent-Patch PQC Denoising Experiment")
    print(f"Device: {device} | Projector Seed: {projector_seed} | Dataset Seed: {dataset_seed}")
    print(f"DataLoader Seed: {dataloader_seed}")
    print(f"PQC Init Seed: {pqc_init_seed} | Classical Init Seed: {classical_init_seed}")
    print(f"Max Train Samples: {max_train_samples} | Max Val Samples: {max_val_samples}")
    print(f"Train Images: {train_patches_tensor.size(0)} ({len(train_dataset)} patches) | "
          f"Val Images: {val_patches_tensor.size(0)} ({len(val_dataset)} patches)")
    print(f"Noise Std: {noise_std} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 70)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Train PQC Denoiser with dedicated seeded DataLoader
    pqc_train_loader = create_reproducible_train_loader(
        train_dataset,
        batch_size=batch_size,
        seed=dataloader_seed,
    )
    set_seed(pqc_init_seed)
    pqc_model = PQCDenoiser(n_qubits=qubits, n_layers=layers)
    print(f"\n--- Training PQC Denoiser ({pqc_model.num_parameters} quantum parameters) ---")
    pqc_results = train_denoiser(
        model=pqc_model,
        train_loader=pqc_train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        model_name="PQC",
    )
    # Compute secondary 64-D reconstruction MSE
    pqc_val_64d = compute_secondary_64d_reconstruction_mse(
        pqc_model, val_loader, projector, device
    )
    pqc_results["secondary_metrics"] = {"val_recon_mse_64d": pqc_val_64d}

    results_summary: Dict[str, Any] = {
        "experiment": "latent_patch_pqc_denoising",
        "stage": "Module 2 Stage 2",
        "primary_metric": "6d_projected_denoising_mse",
        "secondary_metric": "64d_pseudo_inverse_reconstruction_mse",
        "qubits": qubits,
        "layers": layers,
        "noise_std": noise_std,
        "projector_seed": projector_seed,
        "dataset_seed": dataset_seed,
        "dataloader_seed": dataloader_seed,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "num_train_images": train_patches_tensor.size(0),
        "num_val_images": val_patches_tensor.size(0),
        "num_train_patches": len(train_dataset),
        "num_val_patches": len(val_dataset),
        "model_initialization": {
            "pqc_init_seed": pqc_init_seed,
            "classical_init_seed": classical_init_seed if compare_classical else None,
        },
        "device": str(device),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "pqc_results": pqc_results,
    }

    # 4. Optional Classical Baseline Comparison with IDENTICALLY seeded DataLoader
    if compare_classical:
        classical_train_loader = create_reproducible_train_loader(
            train_dataset,
            batch_size=batch_size,
            seed=dataloader_seed,
        )
        set_seed(classical_init_seed)
        classical_model = ParameterMatchedClassicalBaseline(dim=qubits, layers=layers)
        print(f"\n--- Training Parameter-Matched Classical Baseline ({classical_model.num_parameters} classical parameters) ---")
        classical_results = train_denoiser(
            model=classical_model,
            train_loader=classical_train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            model_name="Classical_Circulant_MLP",
        )
        classical_val_64d = compute_secondary_64d_reconstruction_mse(
            classical_model, val_loader, projector, device
        )
        classical_results["secondary_metrics"] = {"val_recon_mse_64d": classical_val_64d}
        results_summary["classical_results"] = classical_results

    # 5. Non-destructive results logging
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "latent_patch_pqc_denoising_results.json"

    runs = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    runs = existing_data
                elif isinstance(existing_data, dict):
                    runs = [existing_data]
        except (json.JSONDecodeError, OSError):
            runs = []

    run_id = len(runs) + 1
    results_summary["run_id"] = run_id
    runs.append(results_summary)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    print("\n" + "=" * 70)
    print("Experiment Summary (Stage 2)")
    print(f"Run ID:                  {run_id} (Total runs preserved: {len(runs)})")
    print(f"PQC Parameters:          {pqc_results['parameter_count']}")
    print(f"PQC Initial Val 6-D MSE: {pqc_results['initial_val_loss']:.6f}")
    print(f"PQC Final Val 6-D MSE:   {pqc_results['final_val_loss']:.6f}")
    print(f"PQC Val 64-D Recon MSE:  {pqc_val_64d:.6f}")
    print(f"PQC Wall-Clock:          {pqc_results['total_wall_time']:.2f}s")
    if compare_classical:
        c_res = results_summary["classical_results"]
        print(f"Classical Parameters:    {c_res['parameter_count']}")
        print(f"Classical Final Val MSE: {c_res['final_val_loss']:.6f}")
        print(f"Classical 64-D Recon:    {classical_val_64d:.6f}")
        print(f"Classical Wall-Clock:    {c_res['total_wall_time']:.2f}s")
    print(f"Saved Results to:        {log_file}")
    print("=" * 70)

    return results_summary


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for Stage 2 latent-patch denoising experiment."""
    parser = argparse.ArgumentParser(description="Stage 2 Latent-Patch PQC Denoising Experiment")
    parser.add_argument("--train-patches", default="data/latent_patches/train_patches.pt", help="Path to train patches .pt")
    parser.add_argument("--val-patches", default="data/latent_patches/val_patches.pt", help="Path to val patches .pt")
    parser.add_argument("--qubits", type=int, default=6, help="Number of qubits")
    parser.add_argument("--layers", type=int, default=2, help="Circuit depth")
    parser.add_argument("--noise-std", type=float, default=0.2, help="Gaussian noise std dev")
    parser.add_argument("--projector-seed", type=int, default=42, help="Seed for fixed projection")
    parser.add_argument("--dataset-seed", type=int, default=42, help="Seed for dataset noise generation")
    parser.add_argument("--dataloader-seed", type=int, default=42, help="Seed for DataLoader batch shuffling")
    parser.add_argument("--pqc-seed", type=int, default=None, help="Seed for PQC parameter initialization")
    parser.add_argument("--classical-seed", type=int, default=None, help="Seed for classical parameter initialization")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum number of training images to use (default: None, use full dataset)",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Maximum number of validation images to use (default: None, use full dataset)",
    )
    parser.add_argument("--no-classical", action="store_true", help="Skip classical baseline comparison")
    parser.add_argument("--output-dir", default="results/logs", help="Output directory for results JSON")
    parser.add_argument("--device", default=None, help="Device ('cpu' or 'cuda')")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    run_latent_patch_denoising_experiment(
        train_patches_path=args.train_patches,
        val_patches_path=args.val_patches,
        qubits=args.qubits,
        layers=args.layers,
        noise_std=args.noise_std,
        projector_seed=args.projector_seed,
        dataset_seed=args.dataset_seed,
        dataloader_seed=args.dataloader_seed,
        pqc_seed=args.pqc_seed,
        classical_seed=args.classical_seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        compare_classical=not args.no_classical,
        output_dir=args.output_dir,
        device_name=args.device,
    )
