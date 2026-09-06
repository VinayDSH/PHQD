"""
Module 3 Milestone 2B: Toy Diffusion Denoising Learning Experiment.

Objective:
Verify that a single shared 6-qubit, depth-3 Parameterized Quantum Circuit (PQC)
can learn a controlled denoising mapping at a fixed diffusion timestep (t=500)
over 6D harmonic patch vectors.

Pipeline:
Clean toy projected patches [B, 9, 6]
    -> LinearDiffusionSchedule.q_sample(t=500)
    -> Noisy projected patches [B, 9, 6]
    -> Shared PQCDenoiser (18 trainable parameters)
    -> Predicted clean patches [B, 9, 6]
    -> MSE(predicted x0, clean x0)

Key Properties:
1. One Shared PQC: A single 6-qubit, depth-3 PQC (18 parameters) denoises all 9 patch positions.
2. Direct 6D Representation: Synthetic toy data is generated directly in 6D projected space;
   no 64->6 projection is applied.
3. Clean Target: Model is trained to predict clean x0, NOT noise epsilon.
4. Fixed Timestep: Diffusion is evaluated strictly at t=500 for all samples.
5. Zero Leakage & Reproducibility: Independent train (seed=42) and val (seed=1042) data generation
   with dedicated seeded DataLoader generators and reproducible PQC initialization (seed=10042).
"""

import sys
import os
import time
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pqc import PQCDenoiser
from src.training.diffusion_schedule import LinearDiffusionSchedule


def parse_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file with pure-Python fallback parser."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        config: Dict[str, Any] = {}
        current_section: Optional[str] = None

        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())

                if indent == 0 and stripped.endswith(":"):
                    current_section = stripped[:-1].strip()
                    config[current_section] = {}
                elif indent > 0 and current_section is not None:
                    if ":" in stripped:
                        k, v = stripped.split(":", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if v == "":
                            val: Any = {}
                        elif v.isdigit():
                            val = int(v)
                        else:
                            try:
                                val = float(v)
                            except ValueError:
                                if v.lower() == "true":
                                    val = True
                                elif v.lower() == "false":
                                    val = False
                                else:
                                    val = v
                        config[current_section][k] = val
        return config


def generate_toy_diffusion_patches(
    num_samples: int = 512,
    num_patches: int = 9,
    dim: int = 6,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic clean 6D patch vectors directly in projected space.

    Formula per sample i, patch p, and coordinate j:
        x0[i, p, j] = A_i * sin(2 * pi * f_i * j / dim + phi_p)
    where:
        - j = 0, ..., dim-1 (6D coordinate)
        - p = 0, ..., num_patches-1 (9 patch locations)
        - A_i in [0.3, 0.7] (per-sample amplitude)
        - f_i in {1, 2} (per-sample spatial frequency)
        - phi_p = 2 * pi * p / num_patches (deterministic phase per patch position)

    Clean target shape: [num_samples, num_patches, dim] with values in [-0.7, 0.7].

    Args:
        num_samples: Number of samples to generate. Defaults to 512.
        num_patches: Number of spatial patch positions. Defaults to 9.
        dim: Projected dimension (equals qubit count). Defaults to 6.
        seed: Random seed for reproducible generation. Defaults to 42.

    Returns:
        torch.Tensor of shape [num_samples, num_patches, dim] with dtype float32.
    """
    gen = torch.Generator().manual_seed(seed)

    # Per-sample amplitude A in [0.3, 0.7] -> shape [num_samples, 1, 1]
    amplitudes = 0.3 + 0.4 * torch.rand(num_samples, 1, 1, generator=gen)

    # Per-sample frequency f in {1, 2} -> shape [num_samples, 1, 1]
    freqs = torch.randint(1, 3, (num_samples, 1, 1), generator=gen).float()

    # Deterministic phase phi_p associated with patch position p -> shape [1, num_patches, 1]
    phi_p = (2.0 * math.pi * torch.arange(num_patches, dtype=torch.float32) / float(num_patches)).view(1, num_patches, 1)

    # Coordinate index j in {0, ..., dim-1} -> shape [1, 1, dim]
    j_indices = torch.arange(dim, dtype=torch.float32).view(1, 1, dim)

    # Compute clean harmonic coordinates
    angles = (2.0 * math.pi * freqs * j_indices / float(dim)) + phi_p
    clean_patches = amplitudes * torch.sin(angles)

    return clean_patches.to(torch.float32)


class ToyDiffusionPatchDataset(Dataset):
    """PyTorch Dataset serving noisy projected inputs xt and clean targets x0 at fixed t=500.

    Pipeline:
    1. Clean projected vectors x0 [N, 9, 6] generated from deterministic harmonic formula.
    2. Forward diffusion q_sample applied at fixed timestep t=500:
       xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
    3. Yields (xt[idx], x0[idx]) both of shape [9, 6].
    """

    def __init__(
        self,
        num_samples: int = 512,
        num_patches: int = 9,
        dim: int = 6,
        schedule: Optional[LinearDiffusionSchedule] = None,
        timestep: int = 500,
        seed: int = 42,
        noise_seed: Optional[int] = None,
        clean_patches: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_samples = int(num_samples)
        self.num_patches = int(num_patches)
        self.dim = int(dim)
        self.timestep = int(timestep)
        self.seed = int(seed)

        if schedule is None:
            schedule = LinearDiffusionSchedule(timesteps=1000, beta_start=0.0001, beta_end=0.02)
        self.schedule = schedule

        if clean_patches is not None:
            self.clean = clean_patches.to(torch.float32)
        else:
            self.clean = generate_toy_diffusion_patches(
                num_samples=self.num_samples,
                num_patches=self.num_patches,
                dim=self.dim,
                seed=self.seed,
            )

        # Precompute deterministic forward diffusion noisy inputs at fixed timestep t=500
        eff_noise_seed = noise_seed if noise_seed is not None else self.seed
        self.noisy = self.schedule.q_sample(
            x_0=self.clean,
            t=self.timestep,
            seed=eff_noise_seed,
        )

    def __len__(self) -> int:
        return self.clean.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


def get_toy_diffusion_dataloaders(
    train_samples: int = 512,
    val_samples: int = 128,
    num_patches: int = 9,
    dim: int = 6,
    batch_size: int = 16,
    timestep: int = 500,
    train_seed: int = 42,
    val_seed: int = 1042,
    schedule: Optional[LinearDiffusionSchedule] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Construct deterministic train and validation DataLoaders for toy diffusion denoising."""
    if schedule is None:
        schedule = LinearDiffusionSchedule(timesteps=1000, beta_start=0.0001, beta_end=0.02)

    train_dataset = ToyDiffusionPatchDataset(
        num_samples=train_samples,
        num_patches=num_patches,
        dim=dim,
        schedule=schedule,
        timestep=timestep,
        seed=train_seed,
    )
    val_dataset = ToyDiffusionPatchDataset(
        num_samples=val_samples,
        num_patches=num_patches,
        dim=dim,
        schedule=schedule,
        timestep=timestep,
        seed=val_seed,
    )

    train_gen = torch.Generator(device="cpu").manual_seed(train_seed)
    val_gen = torch.Generator(device="cpu").manual_seed(val_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=val_gen,
    )

    return train_loader, val_loader


class SharedPatchPQCDenoiser(nn.Module):
    """Wraps a single 6-qubit, depth-3 PQCDenoiser shared across all 9 spatial patch locations.

    Interface:
        Input: Noisy projected patches [B, 9, 6]
            -> Reshape to [B * 9, 6]
            -> Evaluated through ONE shared PQCDenoiser (18 trainable parameters)
            -> Reshape back to [B, 9, 6]
        Output: Predicted clean patches x_hat_0 [B, 9, 6] in [-1, 1].
    """

    def __init__(
        self,
        n_qubits: int = 6,
        depth: int = 3,
        dev: Optional[Any] = None,
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.pqc = PQCDenoiser(
            n_qubits=self.n_qubits,
            n_layers=self.depth,
            dev=dev,
        )

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the single shared PQC (strictly 6 * depth = 18)."""
        return self.pqc.num_parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise input patch tensor with the shared PQC.

        Args:
            x: Noisy projected patches of shape [B, 9, 6].

        Returns:
            Predicted clean patches of shape [B, 9, 6] with values in [-1, 1].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
        if x.dim() != 3 or x.shape[1:] != (9, self.n_qubits):
            raise ValueError(f"Expected input shape [B, 9, {self.n_qubits}], got {list(x.shape)}")

        b = x.size(0)
        # Reshape [B, 9, 6] -> [B * 9, 6] to share the single PQC across all 9 positions
        x_flat = x.reshape(b * 9, self.n_qubits)
        out_flat = self.pqc(x_flat)
        # Reshape back to [B, 9, 6]
        return out_flat.reshape(b, 9, self.n_qubits)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch over the dataloader and return the mean MSE loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        predicted_x0 = model(noisy)
        loss = criterion(predicted_x0, clean)
        loss.backward()
        optimizer.step()

        batch_size = noisy.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model over dataloader in eval mode and return the mean MSE loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            predicted_x0 = model(noisy)
            loss = criterion(predicted_x0, clean)
            batch_size = noisy.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def run_toy_diffusion_experiment(
    config_or_path: Union[str, Path, Dict[str, Any]] = "configs/module3_toy_diffusion.yaml",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the Module 3 Milestone 2B toy diffusion denoising learning experiment.

    Returns:
        Dictionary containing experiment parameters, training history, and convergence metrics.
    """
    if isinstance(config_or_path, (str, Path)):
        config = parse_yaml_config(config_path=config_or_path)
    else:
        config = config_or_path

    # Extract configuration parameters with default fallbacks
    ds_cfg = config.get("dataset", {})
    train_samples = int(ds_cfg.get("train_samples", 512))
    val_samples = int(ds_cfg.get("val_samples", 128))
    num_patches = int(ds_cfg.get("num_patches", 9))
    dim = int(ds_cfg.get("dim", 6))
    train_seed = int(ds_cfg.get("train_seed", 42))
    val_seed = int(ds_cfg.get("val_seed", 1042))

    diff_cfg = config.get("diffusion", {})
    timesteps = int(diff_cfg.get("timesteps", 1000))
    beta_start = float(diff_cfg.get("beta_start", 0.0001))
    beta_end = float(diff_cfg.get("beta_end", 0.02))
    fixed_timestep = int(diff_cfg.get("fixed_timestep", 500))

    model_cfg = config.get("model", {})
    n_qubits = int(model_cfg.get("n_qubits", 6))
    pqc_depth = int(model_cfg.get("pqc_depth", 3))
    pqc_seed = int(model_cfg.get("pqc_seed", 10042))

    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 15))
    batch_size = int(train_cfg.get("batch_size", 16))
    learning_rate = float(train_cfg.get("learning_rate", 0.05))

    out_cfg = config.get("output", {})
    log_dir = Path(out_cfg.get("log_dir", "results/logs"))
    log_filename = out_cfg.get("log_filename", "toy_diffusion_pqc_denoising_results.json")

    # Select execution device
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Schedule and DataLoaders
    schedule = LinearDiffusionSchedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    train_loader, val_loader = get_toy_diffusion_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        num_patches=num_patches,
        dim=dim,
        batch_size=batch_size,
        timestep=fixed_timestep,
        train_seed=train_seed,
        val_seed=val_seed,
        schedule=schedule,
    )

    # 2. Instantiate Model with fixed initialization seed
    torch.manual_seed(pqc_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pqc_seed)

    model = SharedPatchPQCDenoiser(n_qubits=n_qubits, depth=pqc_depth).to(device)
    param_count = model.num_parameters

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("=" * 72)
    print("PHQD Module 3 Milestone 2B: Toy Diffusion Denoising Learning Experiment")
    print(f"Device: {device} | Fixed Timestep: t={fixed_timestep}/{timesteps}")
    print(f"Model: Shared PQC ({n_qubits} qubits, depth {pqc_depth}) -> {param_count} trainable params")
    print(f"Train samples: {train_samples} | Val samples: {val_samples} | Batch size: {batch_size}")
    print(f"Epochs: {epochs} | Learning rate: {learning_rate}")
    print(f"Seeds: train={train_seed}, val={val_seed}, pqc_init={pqc_seed}")
    print("=" * 72)

    # 3. Evaluate initial untrained performance
    initial_train_mse = evaluate(model, train_loader, criterion, device)
    initial_val_mse = evaluate(model, val_loader, criterion, device)
    print(f"Initial (Untrained) -> Train MSE: {initial_train_mse:.6f} | Val MSE: {initial_val_mse:.6f}\n")

    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "epoch_time": [],
    }

    start_total_time = time.time()

    # 4. Training Loop
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_mse = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse = evaluate(model, val_loader, criterion, device)
        epoch_dur = time.time() - t0

        history["epoch"].append(ep)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["epoch_time"].append(epoch_dur)

        print(f"Epoch {ep:2d}/{epochs:2d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | Time: {epoch_dur:.2f}s")

    total_runtime = time.time() - start_total_time

    best_val_mse = min(history["val_mse"])
    best_epoch = history["val_mse"].index(best_val_mse) + 1
    final_train_mse = history["train_mse"][-1]
    final_val_mse = history["val_mse"][-1]

    # Check acceptance criteria:
    train_improved = final_train_mse < initial_train_mse
    val_improved = best_val_mse < initial_val_mse

    print("\n" + "=" * 72)
    print("EXPERIMENT SUMMARY & ACCEPTANCE CRITERIA")
    print(f"Total Runtime: {total_runtime:.2f}s")
    print(f"Train MSE: Initial {initial_train_mse:.6f} -> Final {final_train_mse:.6f} (Improved: {train_improved})")
    print(f"Val MSE:   Initial {initial_val_mse:.6f} -> Best {best_val_mse:.6f} (Epoch {best_epoch}) (Improved: {val_improved})")
    print("=" * 72)

    results: Dict[str, Any] = {
        "experiment": "toy_diffusion_pqc_denoising",
        "milestone": "Module 3 Milestone 2B",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_count": param_count,
        "fixed_timestep": fixed_timestep,
        "n_qubits": n_qubits,
        "pqc_depth": pqc_depth,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "pqc_seed": pqc_seed,
        "initial_train_mse": initial_train_mse,
        "initial_val_mse": initial_val_mse,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "final_train_mse": final_train_mse,
        "final_val_mse": final_val_mse,
        "train_improved": train_improved,
        "val_improved": val_improved,
        "total_runtime": total_runtime,
        "history": history,
    }

    # 5. Non-destructive JSON Logging
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename
    existing_entries = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing_entries = content
                elif isinstance(content, dict):
                    existing_entries = [content]
        except Exception as e:
            print(f"Warning: could not read existing log file: {e}")

    existing_entries.append(results)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(existing_entries, f, indent=2)
    print(f"Logged results to: {log_file}")

    return results


def main() -> None:
    """CLI entry point for running the Milestone 2B toy diffusion experiment."""
    parser = argparse.ArgumentParser(description="PHQD Module 3 Milestone 2B: Toy Diffusion PQC Denoising Experiment")
    parser.add_argument("--config", type=str, default="configs/module3_toy_diffusion.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    args = parser.parse_args()

    config = parse_yaml_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    run_toy_diffusion_experiment(config_or_path=config, device_name=args.device)


if __name__ == "__main__":
    main()
