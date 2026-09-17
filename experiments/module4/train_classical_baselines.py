"""
Module 4.2: Classical Baseline Training Harness.

Trains all 5 models under strictly controlled experimental conditions:
1. Reference HQCLD (Hybrid 6-qubit PQC with classical adapters)
2. MLP Denoiser
3. Fourier-MLP Denoiser
4. Dequantized Surrogate Denoiser
5. Full Classical U-Net Denoiser

Experimental Controls:
- Identical train (seed=42) and validation (seed=1042) latent grids and patches.
- Identical diffusion schedule (Linear, T=1000, beta_start=1e-4, beta_end=0.02).
- Identical optimization parameters (Adam, lr=1e-3, batch_size=16).
- Logs train loss, validation loss, epoch runtime, total wall-clock, and best validation epoch.
- Saves best checkpoint for each model under checkpoints/baselines/.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patch_extractor import extract_latent_patches
from src.training.diffusion_schedule import LinearDiffusionSchedule
from src.models.hqcld_denoiser import HQCLDDenoiser
from src.models.baselines.mlp_denoiser import MLPDenoiser
from src.models.baselines.fourier_mlp_denoiser import FourierMLPDenoiser
from src.models.baselines.dequantized_surrogate import DequantizedSurrogateDenoiser
from src.models.baselines.unet_denoiser import UNetLatentDenoiser
from src.evaluation.computational_metrics import count_parameters


def generate_paired_diffusion_dataset(
    num_samples: int = 512,
    channels: int = 4,
    height: int = 8,
    width: int = 8,
    patch_size: int = 4,
    stride: int = 2,
    seed: int = 42,
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Pregenerate deterministic continuous latents and patches with diffusion noise."""
    gen = torch.Generator().manual_seed(seed)
    schedule = LinearDiffusionSchedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)

    # 1. Continuous clean latent grids z0 [N, 4, 8, 8]
    z0 = torch.randn(num_samples, channels, height, width, generator=gen, dtype=torch.float32)

    # 2. Clean patches x0 [N, 9, 4, 4, 4] -> flattened [N, 9, 64]
    patches_4d = extract_latent_patches(z0, patch_size=patch_size, stride=stride)
    x0_patches = patches_4d.view(num_samples, 9, 64)

    # 3. Random diffusion timesteps t in [0, T-1]
    t = torch.randint(0, timesteps, (num_samples,), generator=gen, dtype=torch.long)

    # 4. Noisy latent grids zt [N, 4, 8, 8] for U-Net
    zt = schedule.q_sample(z0, t=t, generator=gen)

    # 5. Noisy patches xt [N, 9, 64] for patch models
    xt_patches = schedule.q_sample(x0_patches, t=t, generator=gen)

    return {
        "z0": z0,
        "zt": zt,
        "x0_patches": x0_patches,
        "xt_patches": xt_patches,
        "t": t,
    }


def create_model_by_name(model_name: str, config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Instantiate a baseline or HQCLD model from configuration."""
    model_cfg = config.get("models", {}).get(model_name, {})

    if model_name == "hqcld":
        model = HQCLDDenoiser(
            patch_dim=model_cfg.get("patch_dim", 64),
            n_qubits=model_cfg.get("n_qubits", 6),
            pqc_depth=model_cfg.get("pqc_depth", 3),
            num_patches=model_cfg.get("num_patches", 9),
            adapter_mode=model_cfg.get("adapter_mode", "denoiser_matched"),
            bias=model_cfg.get("bias", False),
        )
    elif model_name == "mlp":
        model = MLPDenoiser(
            patch_dim=model_cfg.get("patch_dim", 64),
            num_patches=model_cfg.get("num_patches", 9),
            matching_tier=model_cfg.get("matching_tier", "denoiser_matched"),
            bias=model_cfg.get("bias", False),
        )
    elif model_name == "fourier_mlp":
        model = FourierMLPDenoiser(
            patch_dim=model_cfg.get("patch_dim", 64),
            num_patches=model_cfg.get("num_patches", 9),
            num_frequencies=model_cfg.get("num_frequencies", 6),
            sigma=model_cfg.get("sigma", 1.0),
            seed=model_cfg.get("seed", 42),
            matching_tier=model_cfg.get("matching_tier", "denoiser_matched"),
            bias=model_cfg.get("bias", False),
        )
    elif model_name == "dequantized_surrogate":
        model = DequantizedSurrogateDenoiser(
            patch_dim=model_cfg.get("patch_dim", 64),
            n_qubits=model_cfg.get("n_qubits", 6),
            order=model_cfg.get("order", 3),
            num_patches=model_cfg.get("num_patches", 9),
            matching_tier=model_cfg.get("matching_tier", "denoiser_matched"),
            bias=model_cfg.get("bias", False),
        )
    elif model_name == "unet":
        model = UNetLatentDenoiser(
            in_channels=model_cfg.get("in_channels", 4),
            out_channels=model_cfg.get("out_channels", 4),
            base_channels=model_cfg.get("base_channels", 16),
            time_emb_dim=model_cfg.get("time_emb_dim", 32),
        )
    else:
        raise ValueError(f"Unrecognized model_name: '{model_name}'")

    return model.to(device)


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    is_unet: bool,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a single model reporting progress and validation tracking."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    total_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Training phase
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            if is_unet:
                # (zt, z0, t)
                inputs, targets, t_batch = [b.to(device) for b in batch]
            else:
                # (xt, x0, t)
                inputs, targets, t_batch = [b.to(device) for b in batch]

            optimizer.zero_grad()
            preds = model(inputs, t_batch)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            train_batches += inputs.size(0)

        train_loss = running_train_loss / train_batches

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if is_unet:
                    inputs, targets, t_batch = [b.to(device) for b in batch]
                else:
                    inputs, targets, t_batch = [b.to(device) for b in batch]

                preds = model(inputs, t_batch)
                loss = criterion(preds, targets)
                running_val_loss += loss.item() * inputs.size(0)
                val_batches += inputs.size(0)

        val_loss = running_val_loss / val_batches
        epoch_duration = time.perf_counter() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if save_checkpoint_path is not None:
                save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_name": model_name,
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "parameters": count_parameters(model),
                    },
                    save_checkpoint_path,
                )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_duration_seconds": epoch_duration,
        })

    total_duration = time.perf_counter() - total_start_time

    return {
        "model_name": model_name,
        "trainable_parameters": count_parameters(model),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_wallclock_seconds": total_duration,
        "history": history,
    }


def parse_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Parse YAML config with pyyaml or simple fallback."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Module 4.2 Classical Baseline Training Harness")
    parser.add_argument("--config", type=str, default="configs/module4_baselines.yaml", help="Path to config")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch with small batch for testing")
    parser.add_argument("--models", nargs="+", default=["hqcld", "mlp", "fourier_mlp", "dequantized_surrogate", "unet"])
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = parse_yaml_config(config_path)

    epochs = 1 if args.smoke_test else config["training"]["epochs"]
    batch_size = 8 if args.smoke_test else config["training"]["batch_size"]
    train_samples = 32 if args.smoke_test else config["dataset"]["train_samples"]
    val_samples = 16 if args.smoke_test else config["dataset"]["val_samples"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate deterministic shared datasets
    train_data = generate_paired_diffusion_dataset(
        num_samples=train_samples,
        seed=config["dataset"]["train_seed"],
        timesteps=config["diffusion"]["timesteps"],
    )
    val_data = generate_paired_diffusion_dataset(
        num_samples=val_samples,
        seed=config["dataset"]["val_seed"],
        timesteps=config["diffusion"]["timesteps"],
    )

    # Patch dataloaders: (xt_patches, x0_patches, t)
    patch_train_ds = TensorDataset(train_data["xt_patches"], train_data["x0_patches"], train_data["t"])
    patch_val_ds = TensorDataset(val_data["xt_patches"], val_data["x0_patches"], val_data["t"])
    patch_train_loader = DataLoader(patch_train_ds, batch_size=batch_size, shuffle=True)
    patch_val_loader = DataLoader(patch_val_ds, batch_size=batch_size, shuffle=False)

    # Latent grid dataloaders: (zt, z0, t)
    latent_train_ds = TensorDataset(train_data["zt"], train_data["z0"], train_data["t"])
    latent_val_ds = TensorDataset(val_data["zt"], val_data["z0"], val_data["t"])
    latent_train_loader = DataLoader(latent_train_ds, batch_size=batch_size, shuffle=True)
    latent_val_loader = DataLoader(latent_val_ds, batch_size=batch_size, shuffle=False)

    ckpt_dir = PROJECT_ROOT / config["output"]["checkpoints_dir"]
    results = {}

    for name in args.models:
        print(f"--- Training {name.upper()} ({epochs} epochs) ---")
        model = create_model_by_name(name, config, device)
        is_unet = (name == "unet")
        tr_loader = latent_train_loader if is_unet else patch_train_loader
        va_loader = latent_val_loader if is_unet else patch_val_loader
        ckpt_path = ckpt_dir / f"best_{name}.pt"

        res = train_single_model(
            model_name=name,
            model=model,
            train_loader=tr_loader,
            val_loader=va_loader,
            is_unet=is_unet,
            epochs=epochs,
            lr=config["training"]["learning_rate"],
            device=device,
            save_checkpoint_path=ckpt_path,
        )
        results[name] = res
        print(f"Finished {name}: best val loss = {res['best_val_loss']:.6f} in {res['total_wallclock_seconds']:.2f}s")

    log_file = PROJECT_ROOT / config["output"]["logs_dir"] / "training_summary.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Training summary written to: {log_file}")


if __name__ == "__main__":
    main()
