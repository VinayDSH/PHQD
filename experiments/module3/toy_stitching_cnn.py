"""
Module 3 Milestone 3: Classical Stitching CNN Experiment.

Objective:
Verify that a classical 4 -> 32 -> 32 -> 4 residual CNN refines the deterministic 3x3
overlap-averaged continuous latent grid [B, 4, 8, 8] reconstructed from 9 spatial patches
[B, 9, 4, 4, 4].

Pipeline:
Synthetic Clean Latent Grid z0 [B, 4, 8, 8]
    -> extract_latent_patches(patch_size=4, stride=2) [B, 9, 4, 4, 4]
    -> independent patch-wise noise perturbation (noise_std=0.25)
    -> deterministic OverlapAverageStitcher [B, 4, 8, 8] (initial overlap MSE)
    -> ResidualRefinementCNN (4 -> 32 -> 32 -> 4 residual block)
    -> Refined Latent Grid [B, 4, 8, 8]
    -> MSE(refined z, clean z0)

Scientific Integrity Distinctions:
1. Exact Overlap Reconstruction Identity:
   Clean patches extracted from a clean continuous latent grid reconstruct the latent exactly
   via deterministic overlap averaging (zero difference).
2. Toy Patch-Noise Refinement:
   Independently perturbed patches are used to test whether the classical CNN improves
   reconstruction MSE beyond simple overlap averaging.
   (Note: This is toy patch-noise refinement; seam-consistency loss remains Milestone 4).

Acceptance Criterion:
    final_cnn_mse < initial_overlap_average_mse
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

from src.data.patch_extractor import extract_latent_patches
from src.models.stitching_cnn import ClassicalStitchingCNN, OverlapAverageStitcher


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


def generate_synthetic_latents(
    num_samples: int = 512,
    channels: int = 4,
    height: int = 8,
    width: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic continuous 4x8x8 latent grids with smooth 2D harmonic variations.

    Formula for sample i, channel c, coordinate (y, x):
        z[i, c, y, x] = A_{i,c} * sin(2*pi*f_y*y + phi_y) * cos(2*pi*f_x*x + phi_x)

    Args:
        num_samples: Number of latent grids to generate. Defaults to 512.
        channels: Channel depth. Defaults to 4.
        height: Spatial height. Defaults to 8.
        width: Spatial width. Defaults to 8.
        seed: Random seed. Defaults to 42.

    Returns:
        Tensor of shape [num_samples, channels, height, width] with dtype float32.
    """
    gen = torch.Generator().manual_seed(seed)

    y_coords = torch.linspace(0, 1, height).view(1, 1, height, 1)
    x_coords = torch.linspace(0, 1, width).view(1, 1, 1, width)

    amps = 0.5 + 0.5 * torch.rand(num_samples, channels, 1, 1, generator=gen)
    fy = torch.randint(1, 3, (num_samples, channels, 1, 1), generator=gen).float()
    fx = torch.randint(1, 3, (num_samples, channels, 1, 1), generator=gen).float()
    py = torch.rand(num_samples, channels, 1, 1, generator=gen) * 2.0 * math.pi
    px = torch.rand(num_samples, channels, 1, 1, generator=gen) * 2.0 * math.pi

    latents = amps * torch.sin(2.0 * math.pi * fy * y_coords + py) * torch.cos(2.0 * math.pi * fx * x_coords + px)
    return latents.to(torch.float32)


class ToyStitchingDataset(Dataset):
    """PyTorch Dataset yielding independently perturbed 9-patch inputs and clean 4x8x8 targets.

    Pipeline:
    1. Clean continuous latents z0 [N, 4, 8, 8].
    2. Overlapping patches extracted with stride 2: [N, 9, 4, 4, 4] via extract_latent_patches.
    3. Independent patch noise added: noisy_patches = patches + N(0, noise_std^2).
    4. Yields (noisy_patches[idx], clean_latents[idx]).
    """

    def __init__(
        self,
        num_samples: int = 512,
        channels: int = 4,
        height: int = 8,
        width: int = 8,
        patch_size: int = 4,
        stride: int = 2,
        noise_std: float = 0.25,
        seed: int = 42,
        clean_latents: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_samples = int(num_samples)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

        if clean_latents is not None:
            self.clean_latents = clean_latents.to(torch.float32)
        else:
            self.clean_latents = generate_synthetic_latents(
                num_samples=self.num_samples,
                channels=self.channels,
                height=self.height,
                width=self.width,
                seed=self.seed,
            )

        # Extract 9 overlapping patches using existing Module 1 patch extraction routine
        self.clean_patches = extract_latent_patches(
            self.clean_latents,
            patch_size=self.patch_size,
            stride=self.stride,
        )

        # Apply independent patch-wise perturbation (simulating independent patch denoiser errors)
        noise_gen = torch.Generator().manual_seed(self.seed + 1000)
        noise = self.noise_std * torch.randn(self.clean_patches.shape, generator=noise_gen)
        self.noisy_patches = (self.clean_patches + noise).to(torch.float32)

    def __len__(self) -> int:
        return self.clean_latents.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noisy_patches[idx], self.clean_latents[idx]


def get_toy_stitching_dataloaders(
    train_samples: int = 512,
    val_samples: int = 128,
    channels: int = 4,
    height: int = 8,
    width: int = 8,
    patch_size: int = 4,
    stride: int = 2,
    noise_std: float = 0.25,
    batch_size: int = 32,
    train_seed: int = 42,
    val_seed: int = 1042,
) -> Tuple[DataLoader, DataLoader]:
    """Construct deterministic train and validation DataLoaders."""
    train_dataset = ToyStitchingDataset(
        num_samples=train_samples,
        channels=channels,
        height=height,
        width=width,
        patch_size=patch_size,
        stride=stride,
        noise_std=noise_std,
        seed=train_seed,
    )
    val_dataset = ToyStitchingDataset(
        num_samples=val_samples,
        channels=channels,
        height=height,
        width=width,
        patch_size=patch_size,
        stride=stride,
        noise_std=noise_std,
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


def evaluate_overlap_average(
    model: ClassicalStitchingCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate MSE of deterministic overlap-average stitching alone against clean targets."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy_patches, clean_latents in dataloader:
            noisy_patches = noisy_patches.to(device)
            clean_latents = clean_latents.to(device)

            z_overlap = model.stitch(noisy_patches)
            loss = criterion(z_overlap, clean_latents)

            batch_size = noisy_patches.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


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

    for noisy_patches, clean_latents in dataloader:
        noisy_patches = noisy_patches.to(device)
        clean_latents = clean_latents.to(device)

        optimizer.zero_grad()
        refined = model(noisy_patches)
        loss = criterion(refined, clean_latents)
        loss.backward()
        optimizer.step()

        batch_size = noisy_patches.size(0)
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
        for noisy_patches, clean_latents in dataloader:
            noisy_patches = noisy_patches.to(device)
            clean_latents = clean_latents.to(device)

            refined = model(noisy_patches)
            loss = criterion(refined, clean_latents)

            batch_size = noisy_patches.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def run_toy_stitching_experiment(
    config_or_path: Union[str, Path, Dict[str, Any]] = "configs/module3_toy_stitching.yaml",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the Module 3 Milestone 3 toy stitching CNN experiment.

    Returns:
        Dictionary containing experiment parameters, training history, and convergence metrics.
    """
    if isinstance(config_or_path, (str, Path)):
        config = parse_yaml_config(config_path=config_or_path)
    else:
        config = config_or_path

    ds_cfg = config.get("dataset", {})
    train_samples = int(ds_cfg.get("train_samples", 512))
    val_samples = int(ds_cfg.get("val_samples", 128))
    channels = int(ds_cfg.get("channels", 4))
    height = int(ds_cfg.get("height", 8))
    width = int(ds_cfg.get("width", 8))
    patch_size = int(ds_cfg.get("patch_size", 4))
    stride = int(ds_cfg.get("stride", 2))
    noise_std = float(ds_cfg.get("noise_std", 0.25))
    train_seed = int(ds_cfg.get("train_seed", 42))
    val_seed = int(ds_cfg.get("val_seed", 1042))

    model_cfg = config.get("model", {})
    hidden_channels = int(model_cfg.get("hidden_channels", 32))
    kernel_size = int(model_cfg.get("kernel_size", 3))

    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 15))
    batch_size = int(train_cfg.get("batch_size", 32))
    learning_rate = float(train_cfg.get("learning_rate", 0.001))

    out_cfg = config.get("output", {})
    log_dir = Path(out_cfg.get("log_dir", "results/logs"))
    log_filename = out_cfg.get("log_filename", "toy_stitching_cnn_results.json")

    # Select execution device
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup DataLoaders
    train_loader, val_loader = get_toy_stitching_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        channels=channels,
        height=height,
        width=width,
        patch_size=patch_size,
        stride=stride,
        noise_std=noise_std,
        batch_size=batch_size,
        train_seed=train_seed,
        val_seed=val_seed,
    )

    # 2. Instantiate Model
    torch.manual_seed(42)
    model = ClassicalStitchingCNN(
        grid_size=height,
        patch_size=patch_size,
        stride=stride,
        channels=channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
    ).to(device)

    param_count = model.num_trainable_parameters
    stitcher_params = model.num_stitcher_parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("=" * 72)
    print("PHQD Module 3 Milestone 3: Classical Stitching CNN Experiment")
    print(f"Device: {device} | Model Parameters: {param_count} trainable (Stitcher: {stitcher_params})")
    print(f"Latent Grid: [{channels}, {height}, {width}] | Patches: 9 x [{channels}, {patch_size}, {patch_size}]")
    print(f"Noise std: {noise_std} (independent patch perturbation)")
    print(f"Train samples: {train_samples} | Val samples: {val_samples} | Batch size: {batch_size}")
    print(f"Epochs: {epochs} | Learning rate: {learning_rate}")
    print(f"Seeds: train={train_seed}, val={val_seed}")
    print("=" * 72)

    # 3. Evaluate Baseline Overlap-Average MSE (no CNN)
    initial_train_overlap_mse = evaluate_overlap_average(model, train_loader, criterion, device)
    initial_val_overlap_mse = evaluate_overlap_average(model, val_loader, criterion, device)
    print(f"Initial Overlap-Average Baseline MSE -> Train: {initial_train_overlap_mse:.6f} | Val: {initial_val_overlap_mse:.6f}\n")

    # Initial untrained CNN MSE
    initial_train_cnn_mse = evaluate(model, train_loader, criterion, device)
    initial_val_cnn_mse = evaluate(model, val_loader, criterion, device)
    print(f"Initial (Untrained CNN) MSE -> Train: {initial_train_cnn_mse:.6f} | Val: {initial_val_cnn_mse:.6f}\n")

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

    final_cnn_train_mse = history["train_mse"][-1]
    final_cnn_val_mse = history["val_mse"][-1]
    best_val_mse = min(history["val_mse"])
    best_epoch = history["val_mse"].index(best_val_mse) + 1

    # Acceptance criterion: final_cnn_mse < initial_overlap_average_mse
    train_improved = final_cnn_train_mse < initial_train_overlap_mse
    val_improved = final_cnn_val_mse < initial_val_overlap_mse
    val_improvement_pct = ((initial_val_overlap_mse - final_cnn_val_mse) / initial_val_overlap_mse) * 100.0

    print("\n" + "=" * 72)
    print("EXPERIMENT SUMMARY & ACCEPTANCE CRITERIA")
    print(f"Total Runtime: {total_runtime:.2f}s")
    print(f"Initial Overlap-Average Val MSE: {initial_val_overlap_mse:.6f}")
    print(f"Final CNN Val MSE:               {final_cnn_val_mse:.6f}")
    print(f"Best Val MSE:                    {best_val_mse:.6f} (Epoch {best_epoch})")
    print(f"Validation Improvement:          {val_improvement_pct:.2f}%")
    print(f"Acceptance Criterion (final_cnn_mse < initial_overlap_average_mse): {val_improved}")
    print("=" * 72)

    results: Dict[str, Any] = {
        "experiment": "toy_stitching_cnn",
        "milestone": "Module 3 Milestone 3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_count": param_count,
        "stitcher_parameters": stitcher_params,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "noise_std": noise_std,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "initial_train_overlap_mse": initial_train_overlap_mse,
        "initial_val_overlap_mse": initial_val_overlap_mse,
        "final_cnn_train_mse": final_cnn_train_mse,
        "final_cnn_val_mse": final_cnn_val_mse,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "val_improvement_pct": val_improvement_pct,
        "acceptance_criterion_met": val_improved,
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
    """CLI entry point for running the Milestone 3 toy stitching CNN experiment."""
    parser = argparse.ArgumentParser(description="PHQD Module 3 Milestone 3: Classical Stitching CNN Experiment")
    parser.add_argument("--config", type=str, default="configs/module3_toy_stitching.yaml", help="Path to config YAML")
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

    run_toy_stitching_experiment(config_or_path=config, device_name=args.device)


if __name__ == "__main__":
    main()
