"""
Module 3 Milestone 5: Combined Loss and lambda_seam Ablation Experiment.

Objective:
Evaluate the multi-objective loss:
    L_total = L_diffusion + lambda_seam * L_seam + lambda_recon * L_recon
with fixed lambda_recon = 1.0 across four lambda_seam values:
    lambda_seam in {0.0, 0.01, 0.05, 0.10}

Experimental Control:
- Strictly identical pre-generated training and validation tensors reused across all lambda runs:
  * Same clean continuous latent grids z0
  * Same clean spatial patches x0
  * Same Gaussian noise perturbation
  * Same noisy patch inputs xt
  * Same train/validation split
  * Same model initialization state dict
  * Same DataLoader batch ordering
- Only lambda_seam differs between the four runs.

Scientific Terminology Note:
L_diffusion is the toy patch denoising MSE objective for this milestone; it represents the
denoiser's patch-level fidelity, not the complete variable-timestep/reverse diffusion pipeline.
The lambda=0 setting serves as the baseline.
"""

import sys
import os
import time
import copy
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patch_extractor import extract_latent_patches
from src.losses.seam_consistency import SeamConsistencyLoss
from src.models.stitching_cnn import OverlapAverageStitcher
from src.losses.combined_loss import CombinedLoss


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
        current_list_key: Optional[str] = None

        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())

                if indent == 0 and stripped.endswith(":"):
                    current_section = stripped[:-1].strip()
                    config[current_section] = {}
                    current_list_key = None
                elif indent > 0 and current_section is not None:
                    if stripped.startswith("- "):
                        val_str = stripped[2:].strip()
                        try:
                            val: Any = float(val_str) if "." in val_str else int(val_str)
                        except ValueError:
                            val = val_str
                        if current_list_key is not None:
                            if not isinstance(config[current_section][current_list_key], list):
                                config[current_section][current_list_key] = []
                            config[current_section][current_list_key].append(val)
                        else:
                            if not isinstance(config[current_section], list):
                                config[current_section] = []
                            config[current_section].append(val)
                    elif ":" in stripped:
                        k, v = stripped.split(":", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if v == "":
                            current_list_key = k
                            config[current_section][k] = []
                        else:
                            current_list_key = None
                            if v.isdigit():
                                parsed_val: Any = int(v)
                            else:
                                try:
                                    parsed_val = float(v)
                                except ValueError:
                                    if v.lower() == "true":
                                        parsed_val = True
                                    elif v.lower() == "false":
                                        parsed_val = False
                                    else:
                                        parsed_val = v
                            if isinstance(config[current_section], dict):
                                config[current_section][k] = parsed_val
        return config


def generate_synthetic_latents(
    num_samples: int = 512,
    channels: int = 4,
    height: int = 8,
    width: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic continuous 4x8x8 latent grids with smooth 2D harmonic variations."""
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


def pregenerate_paired_datasets(
    train_samples: int = 512,
    val_samples: int = 128,
    channels: int = 4,
    height: int = 8,
    width: int = 8,
    patch_size: int = 4,
    stride: int = 2,
    noise_std: float = 0.25,
    train_seed: int = 42,
    val_seed: int = 1042,
) -> Dict[str, torch.Tensor]:
    """Pregenerate deterministic train and validation tensors once for paired ablation.

    Returns:
        Dict with keys:
        - "train_z0": clean train latents [train_samples, 4, 8, 8]
        - "train_x0": clean train patches [train_samples, 9, 4, 4, 4]
        - "train_xt": noisy train patches [train_samples, 9, 4, 4, 4]
        - "val_z0": clean val latents [val_samples, 4, 8, 8]
        - "val_x0": clean val patches [val_samples, 9, 4, 4, 4]
        - "val_xt": noisy val patches [val_samples, 9, 4, 4, 4]
    """
    # 1. Train Split
    train_z0 = generate_synthetic_latents(num_samples=train_samples, channels=channels, height=height, width=width, seed=train_seed)
    train_x0 = extract_latent_patches(train_z0, patch_size=patch_size, stride=stride)
    train_noise_gen = torch.Generator().manual_seed(train_seed + 5000)
    train_noise = noise_std * torch.randn(train_x0.shape, generator=train_noise_gen)
    train_xt = (train_x0 + train_noise).to(torch.float32)

    # 2. Validation Split
    val_z0 = generate_synthetic_latents(num_samples=val_samples, channels=channels, height=height, width=width, seed=val_seed)
    val_x0 = extract_latent_patches(val_z0, patch_size=patch_size, stride=stride)
    val_noise_gen = torch.Generator().manual_seed(val_seed + 5000)
    val_noise = noise_std * torch.randn(val_x0.shape, generator=val_noise_gen)
    val_xt = (val_x0 + val_noise).to(torch.float32)

    return {
        "train_z0": train_z0,
        "train_x0": train_x0,
        "train_xt": train_xt,
        "val_z0": val_z0,
        "val_x0": val_x0,
        "val_xt": val_xt,
    }


class ToyPatchDenoiser(nn.Module):
    """Small classical patch denoiser operating on [B, 9, 4, 4, 4] -> [B, 9, 4, 4, 4]."""

    def __init__(self, in_channels: int = 36, hidden_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c, ph, pw = x.shape
        x_flat = x.view(b, n * c, ph, pw)
        residual = self.conv(x_flat)
        out = x_flat + 0.1 * residual
        return out.view(b, n, c, ph, pw)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate all loss components across a DataLoader in eval mode."""
    model.eval()
    totals = {
        "total": 0.0,
        "diffusion": 0.0,
        "seam": 0.0,
        "horizontal_seam": 0.0,
        "vertical_seam": 0.0,
        "recon": 0.0,
    }
    total_samples = 0

    with torch.no_grad():
        for bx_xt, bx_x0, bx_z0 in dataloader:
            bx_xt = bx_xt.to(device)
            bx_x0 = bx_x0.to(device)
            bx_z0 = bx_z0.to(device)

            pred = model(bx_xt)
            comp = loss_fn.forward_with_components(pred, bx_x0, bx_z0)

            b = bx_xt.size(0)
            for k in totals:
                totals[k] += comp[k].item() * b
            total_samples += b

    return {k: v / total_samples for k, v in totals.items()}


def run_single_ablation(
    lambda_seam: float,
    lambda_recon: float,
    paired_data: Dict[str, torch.Tensor],
    initial_state_dict: Dict[str, Any],
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Train a single paired run under a specific lambda_seam setting."""
    # 1. Setup Loss Function
    loss_fn = CombinedLoss(
        lambda_seam=lambda_seam,
        lambda_recon=lambda_recon,
    ).to(device)

    # 2. Instantiate Model and restore identical initial weights
    model = ToyPatchDenoiser().to(device)
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. Create DataLoaders reusing exact pregenerated tensors
    train_ds = TensorDataset(paired_data["train_xt"], paired_data["train_x0"], paired_data["train_z0"])
    val_ds = TensorDataset(paired_data["val_xt"], paired_data["val_x0"], paired_data["val_z0"])

    train_gen = torch.Generator(device="cpu").manual_seed(999)
    val_gen = torch.Generator(device="cpu").manual_seed(999)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=val_gen)

    # Initial evaluation before training
    init_train_eval = evaluate_model(model, train_loader, loss_fn, device)
    init_val_eval = evaluate_model(model, val_loader, loss_fn, device)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_total": [],
        "val_total": [],
        "val_diffusion": [],
        "val_seam": [],
        "val_horizontal_seam": [],
        "val_vertical_seam": [],
        "val_recon": [],
        "epoch_time": [],
    }

    start_time = time.time()

    # 4. Training Loop
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_train_loss = 0.0
        train_samples = 0

        for bx_xt, bx_x0, bx_z0 in train_loader:
            bx_xt = bx_xt.to(device)
            bx_x0 = bx_x0.to(device)
            bx_z0 = bx_z0.to(device)

            optimizer.zero_grad()
            pred = model(bx_xt)
            loss = loss_fn(pred, bx_x0, bx_z0)
            loss.backward()
            optimizer.step()

            b = bx_xt.size(0)
            epoch_train_loss += loss.item() * b
            train_samples += b

        epoch_train_loss /= train_samples
        val_eval = evaluate_model(model, val_loader, loss_fn, device)
        epoch_time = time.time() - t0

        history["epoch"].append(ep)
        history["train_total"].append(epoch_train_loss)
        history["val_total"].append(val_eval["total"])
        history["val_diffusion"].append(val_eval["diffusion"])
        history["val_seam"].append(val_eval["seam"])
        history["val_horizontal_seam"].append(val_eval["horizontal_seam"])
        history["val_vertical_seam"].append(val_eval["vertical_seam"])
        history["val_recon"].append(val_eval["recon"])
        history["epoch_time"].append(epoch_time)

    runtime = time.time() - start_time

    final_train_eval = evaluate_model(model, train_loader, loss_fn, device)
    final_val_eval = evaluate_model(model, val_loader, loss_fn, device)
    best_val_total = min(history["val_total"])

    return {
        "lambda_seam": lambda_seam,
        "lambda_recon": lambda_recon,
        "parameter_count": model.num_parameters,
        "runtime": runtime,
        "initial_train_total": init_train_eval["total"],
        "initial_val_total": init_val_eval["total"],
        "final_train_total": final_train_eval["total"],
        "final_val_total": final_val_eval["total"],
        "best_val_total": best_val_total,
        "final_diffusion_mse": final_val_eval["diffusion"],
        "final_seam_loss": final_val_eval["seam"],
        "final_horizontal_seam_loss": final_val_eval["horizontal_seam"],
        "final_vertical_seam_loss": final_val_eval["vertical_seam"],
        "final_recon_mse": final_val_eval["recon"],
        "history": history,
    }


def run_toy_combined_loss_ablation(
    config_or_path: Union[str, Path, Dict[str, Any]] = "configs/module3_combined_loss.yaml",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete 4-value lambda_seam ablation experiment."""
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

    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 15))
    batch_size = int(train_cfg.get("batch_size", 32))
    learning_rate = float(train_cfg.get("learning_rate", 0.001))

    ablation_cfg = config.get("ablation", {})
    lambda_recon = float(ablation_cfg.get("lambda_recon", 1.0))
    lambda_seam_values: List[float] = [float(v) for v in ablation_cfg.get("lambda_seam_values", [0.0, 0.01, 0.05, 0.10])]

    out_cfg = config.get("output", {})
    log_dir = Path(out_cfg.get("log_dir", "results/logs"))
    log_filename = out_cfg.get("log_filename", "toy_combined_loss_ablation_results.json")

    # Select execution device
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("PHQD Module 3 Milestone 5: Combined Loss & lambda_seam Ablation Experiment")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"Ablation Values lambda_seam: {lambda_seam_values} | Fixed lambda_recon: {lambda_recon}")
    print("=" * 78)

    # 1. Pre-generate complete deterministic dataset ONCE before the lambda loop
    print("Pre-generating shared deterministic dataset tensors...")
    paired_data = pregenerate_paired_datasets(
        train_samples=train_samples,
        val_samples=val_samples,
        channels=channels,
        height=height,
        width=width,
        patch_size=patch_size,
        stride=stride,
        noise_std=noise_std,
        train_seed=train_seed,
        val_seed=val_seed,
    )

    # 2. Create canonical initial model state dict ONCE for identical starting weights
    torch.manual_seed(42)
    canonical_model = ToyPatchDenoiser()
    initial_state_dict = copy.deepcopy(canonical_model.state_dict())
    param_count = canonical_model.num_parameters
    print(f"Canonical model initialized with {param_count} trainable parameters.")
    print("Beginning paired ablation across lambda_seam values...\n")

    ablation_results: List[Dict[str, Any]] = []

    # 3. Execute Paired Ablation Loop
    for l_seam in lambda_seam_values:
        print(f"--- Running Ablation lambda_seam = {l_seam:.2f} (lambda_recon = {lambda_recon:.1f}) ---")
        run_res = run_single_ablation(
            lambda_seam=l_seam,
            lambda_recon=lambda_recon,
            paired_data=paired_data,
            initial_state_dict=initial_state_dict,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            device=device,
        )
        ablation_results.append(run_res)
        print(
            f"lambda_seam={l_seam:4.2f} | Init Val Total: {run_res['initial_val_total']:.6f} -> "
            f"Final Val Total: {run_res['final_val_total']:.6f} (Best: {run_res['best_val_total']:.6f}) | "
            f"Seam: {run_res['final_seam_loss']:.6f} (H: {run_res['final_horizontal_seam_loss']:.6f}, V: {run_res['final_vertical_seam_loss']:.6f}) | "
            f"Rec: {run_res['final_recon_mse']:.6f} | Diff: {run_res['final_diffusion_mse']:.6f} | Runtime: {run_res['runtime']:.2f}s\n"
        )

    # 4. Comparative Analysis Against Baseline (lambda_seam = 0.0)
    baseline_run = next(r for r in ablation_results if r["lambda_seam"] == 0.0)
    b_seam = baseline_run["final_seam_loss"]
    b_rec = baseline_run["final_recon_mse"]
    b_diff = baseline_run["final_diffusion_mse"]

    print("=" * 78)
    print("ABLATION SUMMARY & RELATIVE COMPARISON (Baseline: lambda_seam = 0.0)")
    print(f"{'lambda_seam':>12} | {'Val Total':>10} | {'Seam Loss':>10} | {'delta Seam %':>12} | {'Rec MSE':>10} | {'Diff MSE':>10} | {'Runtime':>8}")
    print("-" * 78)

    for r in ablation_results:
        l_val = r["lambda_seam"]
        v_total = r["final_val_total"]
        v_seam = r["final_seam_loss"]
        v_rec = r["final_recon_mse"]
        v_diff = r["final_diffusion_mse"]
        rt = r["runtime"]

        seam_delta_pct = ((v_seam - b_seam) / b_seam) * 100.0 if b_seam > 0 else 0.0
        r["seam_delta_vs_baseline_pct"] = seam_delta_pct
        r["rec_delta_vs_baseline_pct"] = ((v_rec - b_rec) / b_rec) * 100.0 if b_rec > 0 else 0.0
        r["diff_delta_vs_baseline_pct"] = ((v_diff - b_diff) / b_diff) * 100.0 if b_diff > 0 else 0.0

        print(f"{l_val:12.2f} | {v_total:10.6f} | {v_seam:10.6f} | {seam_delta_pct:+11.2f}% | {v_rec:10.6f} | {v_diff:10.6f} | {rt:7.2f}s")
    print("=" * 78)

    # 5. Non-destructive JSON Logging
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename

    summary_entry = {
        "experiment": "toy_combined_loss_ablation",
        "milestone": "Module 3 Milestone 5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_count": param_count,
        "lambda_recon": lambda_recon,
        "lambda_seam_values": lambda_seam_values,
        "runs": ablation_results,
    }

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

    existing_entries.append(summary_entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(existing_entries, f, indent=2)
    print(f"Logged ablation results to: {log_file}")

    return summary_entry


def main() -> None:
    """CLI entry point for running the Milestone 5 ablation experiment."""
    parser = argparse.ArgumentParser(description="PHQD Module 3 Milestone 5: Combined Loss & lambda_seam Ablation")
    parser.add_argument("--config", type=str, default="configs/module3_combined_loss.yaml", help="Path to config YAML")
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

    run_toy_combined_loss_ablation(config_or_path=config, device_name=args.device)


if __name__ == "__main__":
    main()
