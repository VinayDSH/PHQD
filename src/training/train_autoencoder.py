"""
Module 1 Training and Evaluation Pipeline for ConvAutoencoder.

Features:
- Config loading from configs/module1_fashion_mnist.yaml
- Reproducible deterministic seeding (seed=42)
- CUDA support with fallback to CPU
- MSE reconstruction loss with Adam optimizer
- Per-epoch validation tracking (MSE, PSNR, SSIM)
- Best checkpoint saving based on validation MSE to checkpoints/
- Reconstruction grid visualization saved to results/figures/
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import ConvAutoencoder
from src.data.datasets import get_fashion_mnist_dataloaders
from src.evaluation.metrics import evaluate_reconstruction, save_reconstruction_grid


def parse_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file using PyYAML if available or built-in fallback parser."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Robust pure-Python YAML parser for key-value and list structures
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
                    if stripped.startswith("- "):
                        item = stripped[2:].strip()
                        if not isinstance(config[current_section], list):
                            config[current_section] = []
                        config[current_section].append(item)
                    elif ":" in stripped:
                        k, v = stripped.split(":", 1)
                        k = k.strip()
                        v = v.strip()
                        if v == "":
                            val: Any = {}
                        elif v.isdigit():
                            val = int(v)
                        else:
                            try:
                                val = float(v)
                            except ValueError:
                                val = v
                        if isinstance(config[current_section], dict):
                            config[current_section][k] = val

        return config


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, PyTorch, and CUDA."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch of training and return average training loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()

        reconstructions = model(images)
        loss = criterion(reconstructions, images)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_autoencoder(
    config_path: Union[str, Path] = "configs/module1_fashion_mnist.yaml",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    figure_dir: Union[str, Path] = "results/figures",
    num_epochs: Optional[int] = None,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train ConvAutoencoder using configuration specifications and log metrics.

    Args:
        config_path: Path to YAML configuration file. Defaults to configs/module1_fashion_mnist.yaml.
        checkpoint_dir: Directory to save model checkpoints. Defaults to checkpoints/.
        figure_dir: Directory to save reconstruction visualization figures. Defaults to results/figures/.
        num_epochs: Optional override for number of training epochs (useful for quick tests).
        device_name: Optional override for compute device ('cuda' or 'cpu').

    Returns:
        Dictionary summarizing best validation metrics, history, and training time.
    """
    config = parse_yaml_config(config_path)

    # Extract configuration fields
    training_cfg = config.get("training", {})
    autoencoder_cfg = config.get("autoencoder", {})
    seed = training_cfg.get("seed", 42)
    batch_size = training_cfg.get("batch_size", 128)
    learning_rate = training_cfg.get("learning_rate", 0.001)
    epochs = num_epochs if num_epochs is not None else training_cfg.get("epochs", 30)
    in_channels = config.get("dataset", {}).get("channels", 1)
    latent_channels = autoencoder_cfg.get("latent_channels", 4)

    # Setup directories
    checkpoint_path_dir = Path(checkpoint_dir)
    checkpoint_path_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_path_dir / "best_autoencoder.pt"

    figure_path_dir = Path(figure_dir)
    figure_path_dir.mkdir(parents=True, exist_ok=True)
    figure_output_path = figure_path_dir / "reconstructions_best.png"

    # Enforce reproducibility
    set_seed(seed)

    # Device selection
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = f"{device.type} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"

    print("=" * 70)
    print("PHQD Module 1: ConvAutoencoder Training")
    print(f"Device: {device_info}")
    print(f"Seed: {seed} | Batch Size: {batch_size} | LR: {learning_rate} | Epochs: {epochs}")
    print(f"Architecture: {in_channels}x28x28 -> {latent_channels}x8x8 -> {in_channels}x28x28")
    print("=" * 70)

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        root="data/raw",
        batch_size=batch_size,
        val_size=10000,
        seed=seed,
        download=True,
    )

    # Instantiate model, optimizer, criterion
    model = ConvAutoencoder(in_channels=in_channels, latent_channels=latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    best_val_loss = float("inf")
    best_psnr = 0.0
    best_ssim = 0.0
    best_epoch = -1

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_psnr, val_ssim = evaluate_reconstruction(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_epoch = epoch

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_psnr": best_psnr,
                    "val_ssim": best_ssim,
                    "config": config,
                    "seed": seed,
                    "device": str(device),
                },
                best_checkpoint_path,
            )

        marker = " [*BEST*]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"Time: {epoch_time:.2f}s{marker}"
        )

    total_wall_time = time.time() - start_time

    # Load best weights for visualization and test evaluation
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Generate reconstruction visualization grid with best model
    save_reconstruction_grid(model, val_loader, figure_output_path, num_images=8, device=device)

    print("=" * 70)
    print("Training Complete")
    print(f"Total Wall-Clock Time: {total_wall_time:.2f} seconds ({total_wall_time/60:.2f} minutes)")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation MSE: {best_val_loss:.6f}")
    print(f"Corresponding PSNR:  {best_psnr:.2f} dB")
    print(f"Corresponding SSIM:  {best_ssim:.4f}")
    print(f"Best Checkpoint:     {best_checkpoint_path}")
    print(f"Reconstruction Grid: {figure_output_path}")
    print("=" * 70)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "total_wall_time": total_wall_time,
        "device": device_info,
        "checkpoint_path": str(best_checkpoint_path),
        "figure_path": str(figure_output_path),
        "history": history,
    }


if __name__ == "__main__":
    train_autoencoder()
