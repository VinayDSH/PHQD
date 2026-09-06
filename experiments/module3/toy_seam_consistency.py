"""
Module 3 Milestone 4: Seam-Consistency Loss Experiment.

Objective:
Verify that a small classical correction network reduces cross-patch boundary mismatch
(seam-consistency loss) across adjacent spatial patches in a 3x3 patch grid.

Pipeline:
Synthetic Clean Latent Grid z0 [B, 4, 8, 8]
    -> extract_latent_patches(patch_size=4, stride=2) [B, 9, 4, 4, 4]
    -> independent patch-wise noise perturbation (noise_std=0.25)
    -> toy classical PatchSeamCorrectionNet [B, 9, 4, 4, 4]
    -> SeamConsistencyLoss (horizontal and vertical overlap comparisons)
    -> Secondary diagnostic: reconstruction MSE against clean latent z0 via OverlapAverageStitcher

Important Note:
The PatchSeamCorrectionNet is strictly a toy Milestone 4 experimental component used to
validate that SeamConsistencyLoss is effective and differentiable. It is NOT part of the
final PHQD image-generation architecture.

Acceptance Criteria:
    final_train_seam_loss < initial_train_seam_loss
    best_val_seam_loss < initial_val_seam_loss
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patch_extractor import extract_latent_patches
from src.losses.seam_consistency import SeamConsistencyLoss, HORIZONTAL_PAIRS, VERTICAL_PAIRS
from src.models.stitching_cnn import OverlapAverageStitcher


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


class ToySeamDataset(Dataset):
    """PyTorch Dataset providing independently perturbed 9-patch inputs and clean 4x8x8 targets."""

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

        # Apply independent patch-wise perturbation (inducing seam disagreement)
        noise_gen = torch.Generator().manual_seed(self.seed + 2000)
        noise = self.noise_std * torch.randn(self.clean_patches.shape, generator=noise_gen)
        self.noisy_patches = (self.clean_patches + noise).to(torch.float32)

    def __len__(self) -> int:
        return self.clean_latents.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noisy_patches[idx], self.clean_latents[idx]


class PatchSeamCorrectionNet(nn.Module):
    """Toy classical correction network mapping [B, 9, 4, 4, 4] -> [B, 9, 4, 4, 4].

    Note: This is strictly a toy Milestone 4 experimental validation module to demonstrate
    seam loss optimization; it is NOT part of the final PHQD generation pipeline.
    """

    def __init__(
        self,
        num_patches: int = 9,
        channels: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_patches = int(num_patches)
        self.channels = int(channels)
        in_channels = self.num_patches * self.channels  # 36 channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the toy correction network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Correct patch overlaps with residual skip connection."""
        b, n, c, ph, pw = x.shape
        x_flat = x.view(b, n * c, ph, pw)
        residual = self.net(x_flat)
        out = x_flat + 0.1 * residual
        return out.view(b, n, c, ph, pw)


def get_toy_seam_dataloaders(
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
    train_dataset = ToySeamDataset(
        num_samples=train_samples,
        channels=channels,
        height=height,
        width=width,
        patch_size=patch_size,
        stride=stride,
        noise_std=noise_std,
        seed=train_seed,
    )
    val_dataset = ToySeamDataset(
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


def evaluate_seam_and_rec(
    model: nn.Module,
    dataloader: DataLoader,
    seam_loss_fn: SeamConsistencyLoss,
    stitcher: OverlapAverageStitcher,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate mean seam loss (total, horizontal, vertical) and reconstruction MSE."""
    model.eval()
    total_seam = 0.0
    total_h = 0.0
    total_v = 0.0
    total_rec = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy_patches, clean_latents in dataloader:
            noisy_patches = noisy_patches.to(device)
            clean_latents = clean_latents.to(device)

            corrected_patches = model(noisy_patches)
            components = seam_loss_fn.forward_with_components(corrected_patches)

            # Secondary diagnostic reconstruction MSE
            z_stitched = stitcher(corrected_patches)
            rec_mse = F.mse_loss(z_stitched, clean_latents)

            b = noisy_patches.size(0)
            total_seam += components["total"].item() * b
            total_h += components["horizontal"].item() * b
            total_v += components["vertical"].item() * b
            total_rec += rec_mse.item() * b
            total_samples += b

    return {
        "seam_loss": total_seam / total_samples,
        "horizontal_seam_loss": total_h / total_samples,
        "vertical_seam_loss": total_v / total_samples,
        "rec_mse": total_rec / total_samples,
    }


def evaluate_baseline_seam_and_rec(
    dataloader: DataLoader,
    seam_loss_fn: SeamConsistencyLoss,
    stitcher: OverlapAverageStitcher,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate baseline uncorrected perturbed patches."""
    total_seam = 0.0
    total_h = 0.0
    total_v = 0.0
    total_rec = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy_patches, clean_latents in dataloader:
            noisy_patches = noisy_patches.to(device)
            clean_latents = clean_latents.to(device)

            components = seam_loss_fn.forward_with_components(noisy_patches)
            z_stitched = stitcher(noisy_patches)
            rec_mse = F.mse_loss(z_stitched, clean_latents)

            b = noisy_patches.size(0)
            total_seam += components["total"].item() * b
            total_h += components["horizontal"].item() * b
            total_v += components["vertical"].item() * b
            total_rec += rec_mse.item() * b
            total_samples += b

    return {
        "seam_loss": total_seam / total_samples,
        "horizontal_seam_loss": total_h / total_samples,
        "vertical_seam_loss": total_v / total_samples,
        "rec_mse": total_rec / total_samples,
    }


def save_seam_disagreement_visualization(
    before_patches: torch.Tensor,
    after_patches: torch.Tensor,
    clean_latents: torch.Tensor,
    stitcher: OverlapAverageStitcher,
    output_path: Union[str, Path],
    device: torch.device,
) -> Path:
    """Save before/after seam disagreement and reconstruction visualization.

    Note: Diagnostic toy demonstration only; not the final PHQD image generation result.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Select first 4 validation samples
        b_samples = min(4, before_patches.size(0))
        p_before = before_patches[:b_samples].to(device)
        p_after = after_patches[:b_samples].to(device)
        z_target = clean_latents[:b_samples].to(device)

        # Stitched grids: [B, 4, 8, 8]
        z_before = stitcher(p_before)
        z_after = stitcher(p_after)

        # Compute seam disagreement map across the 6 horizontal and 6 vertical pairs
        # Average channel magnitude to make 1x8x8 error map
        err_before = (z_before - z_target).abs().mean(dim=1, keepdim=True)  # [B, 1, 8, 8]
        err_after = (z_after - z_target).abs().mean(dim=1, keepdim=True)    # [B, 1, 8, 8]

        # Upsample 8x8 grids to 64x64 for clear visual inspection
        scale = 8
        err_before_up = F.interpolate(err_before, scale_factor=scale, mode="nearest")
        err_after_up = F.interpolate(err_after, scale_factor=scale, mode="nearest")

        # Combine into comparison grid: Row 1 = Before (perturbed), Row 2 = After (corrected)
        comparison = torch.cat([err_before_up, err_after_up], dim=0)

        # Save normalized visualization
        save_image(
            comparison,
            output_path,
            nrow=b_samples,
            normalize=True,
            padding=4,
            pad_value=1.0,
        )

    print(f"Saved seam disagreement visualization to: {output_path}")
    print("  Note: Diagnostic toy demonstration only; not the final PHQD image-generation result.")
    return output_path


def run_toy_seam_experiment(
    config_or_path: Union[str, Path, Dict[str, Any]] = "configs/module3_seam_loss.yaml",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the Module 3 Milestone 4 toy seam-consistency loss experiment."""
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
    hidden_dim = int(model_cfg.get("hidden_dim", 64))

    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 15))
    batch_size = int(train_cfg.get("batch_size", 32))
    learning_rate = float(train_cfg.get("learning_rate", 0.001))

    out_cfg = config.get("output", {})
    log_dir = Path(out_cfg.get("log_dir", "results/logs"))
    log_filename = out_cfg.get("log_filename", "toy_seam_consistency_results.json")
    fig_dir = Path(out_cfg.get("figure_dir", "results/figures"))
    fig_filename = out_cfg.get("figure_filename", "seam_disagreement_before_after.png")

    # Select execution device
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup DataLoaders
    train_loader, val_loader = get_toy_seam_dataloaders(
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

    # 2. Setup Loss, Evaluator, and Model
    seam_loss_fn = SeamConsistencyLoss(
        num_patches=9,
        channels=channels,
        patch_size=patch_size,
        overlap=stride,
    ).to(device)

    # OverlapAverageStitcher strictly for secondary diagnostic reconstruction MSE and visualization
    stitcher = OverlapAverageStitcher(
        grid_size=height,
        patch_size=patch_size,
        stride=stride,
        channels=channels,
    ).to(device)

    torch.manual_seed(42)
    model = PatchSeamCorrectionNet(
        num_patches=9,
        channels=channels,
        hidden_dim=hidden_dim,
    ).to(device)

    param_count = model.num_parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("=" * 72)
    print("PHQD Module 3 Milestone 4: Seam-Consistency Loss Experiment")
    print(f"Device: {device} | Toy Model Parameters: {param_count} (SeamLoss: 0 parameters)")
    print(f"Noise std: {noise_std} (independent patch perturbation)")
    print(f"Train samples: {train_samples} | Val samples: {val_samples} | Batch size: {batch_size}")
    print(f"Epochs: {epochs} | Learning rate: {learning_rate}")
    print(f"Seeds: train={train_seed}, val={val_seed}")
    print("=" * 72)

    # 3. Evaluate Initial Baseline Metrics (Before Correction)
    initial_train_metrics = evaluate_baseline_seam_and_rec(train_loader, seam_loss_fn, stitcher, device)
    initial_val_metrics = evaluate_baseline_seam_and_rec(val_loader, seam_loss_fn, stitcher, device)

    initial_train_seam = initial_train_metrics["seam_loss"]
    initial_val_seam = initial_val_metrics["seam_loss"]
    initial_train_rec = initial_train_metrics["rec_mse"]
    initial_val_rec = initial_val_metrics["rec_mse"]

    print(f"Initial Train Seam Loss: {initial_train_seam:.6f} | Rec MSE: {initial_train_rec:.6f}")
    print(f"Initial Val Seam Loss:   {initial_val_seam:.6f} (H: {initial_val_metrics['horizontal_seam_loss']:.6f}, V: {initial_val_metrics['vertical_seam_loss']:.6f}) | Rec MSE: {initial_val_rec:.6f}\n")

    history = {
        "epoch": [],
        "train_seam_loss": [],
        "val_seam_loss": [],
        "val_horizontal_seam_loss": [],
        "val_vertical_seam_loss": [],
        "train_rec_mse": [],
        "val_rec_mse": [],
        "epoch_time": [],
    }

    start_total_time = time.time()

    # 4. Training Loop
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_seam = 0.0
        total_train_samples = 0

        for (noisy_patches, _) in train_loader:
            noisy_patches = noisy_patches.to(device)

            optimizer.zero_grad()
            corrected = model(noisy_patches)
            loss = seam_loss_fn(corrected)
            loss.backward()
            optimizer.step()

            b = noisy_patches.size(0)
            total_train_seam += loss.item() * b
            total_train_samples += b

        train_seam_epoch = total_train_seam / total_train_samples

        # Evaluate on validation set
        val_eval = evaluate_seam_and_rec(model, val_loader, seam_loss_fn, stitcher, device)
        train_eval = evaluate_seam_and_rec(model, train_loader, seam_loss_fn, stitcher, device)
        epoch_dur = time.time() - t0

        history["epoch"].append(ep)
        history["train_seam_loss"].append(train_seam_epoch)
        history["val_seam_loss"].append(val_eval["seam_loss"])
        history["val_horizontal_seam_loss"].append(val_eval["horizontal_seam_loss"])
        history["val_vertical_seam_loss"].append(val_eval["vertical_seam_loss"])
        history["train_rec_mse"].append(train_eval["rec_mse"])
        history["val_rec_mse"].append(val_eval["rec_mse"])
        history["epoch_time"].append(epoch_dur)

        print(
            f"Epoch {ep:2d}/{epochs:2d} | Train Seam: {train_seam_epoch:.6f} | "
            f"Val Seam: {val_eval['seam_loss']:.6f} (H: {val_eval['horizontal_seam_loss']:.6f}, V: {val_eval['vertical_seam_loss']:.6f}) | "
            f"Val Rec MSE: {val_eval['rec_mse']:.6f} | Time: {epoch_dur:.2f}s"
        )

    total_runtime = time.time() - start_total_time

    final_train_seam = history["train_seam_loss"][-1]
    final_val_seam = history["val_seam_loss"][-1]
    final_val_h = history["val_horizontal_seam_loss"][-1]
    final_val_v = history["val_vertical_seam_loss"][-1]

    best_val_seam = min(history["val_seam_loss"])
    best_epoch = history["val_seam_loss"].index(best_val_seam) + 1

    final_train_rec = history["train_rec_mse"][-1]
    final_val_rec = history["val_rec_mse"][-1]
    best_val_rec = min(history["val_rec_mse"])

    # Acceptance criteria verification
    train_improved = final_train_seam < initial_train_seam
    val_improved = best_val_seam < initial_val_seam
    seam_improvement_pct = ((initial_val_seam - final_val_seam) / initial_val_seam) * 100.0

    print("\n" + "=" * 72)
    print("EXPERIMENT SUMMARY & ACCEPTANCE CRITERIA")
    print(f"Total Runtime: {total_runtime:.2f}s")
    print(f"Initial Val Seam Loss: {initial_val_seam:.6f}")
    print(f"Final Val Seam Loss:   {final_val_seam:.6f} (H: {final_val_h:.6f}, V: {final_val_v:.6f})")
    print(f"Best Val Seam Loss:    {best_val_seam:.6f} (Epoch {best_epoch})")
    print(f"Seam Loss Improvement: {seam_improvement_pct:.2f}%")
    print(f"Acceptance Criteria (final_train < init and best_val < init): {train_improved and val_improved}")
    print("-" * 72)
    print("SECONDARY DIAGNOSTIC METRICS (Reconstruction MSE against clean latent):")
    print(f"Initial Val Rec MSE:   {initial_val_rec:.6f}")
    print(f"Final Val Rec MSE:     {final_val_rec:.6f}")
    print(f"Best Val Rec MSE:      {best_val_rec:.6f}")
    print("=" * 72)

    # 5. Generate and Save Before/After Visualization
    fig_path = fig_dir / fig_filename
    val_dataset = val_loader.dataset
    noisy_val_patches = val_dataset.noisy_patches
    clean_val_latents = val_dataset.clean_latents

    model.eval()
    with torch.no_grad():
        corrected_val_patches = model(noisy_val_patches.to(device)).cpu()

    save_seam_disagreement_visualization(
        before_patches=noisy_val_patches,
        after_patches=corrected_val_patches,
        clean_latents=clean_val_latents,
        stitcher=stitcher,
        output_path=fig_path,
        device=device,
    )

    results: Dict[str, Any] = {
        "experiment": "toy_seam_consistency",
        "milestone": "Module 3 Milestone 4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_count": param_count,
        "loss_parameters": seam_loss_fn.num_parameters,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "noise_std": noise_std,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "initial_train_seam_loss": initial_train_seam,
        "initial_val_seam_loss": initial_val_seam,
        "initial_val_horizontal_seam_loss": initial_val_metrics["horizontal_seam_loss"],
        "initial_val_vertical_seam_loss": initial_val_metrics["vertical_seam_loss"],
        "final_train_seam_loss": final_train_seam,
        "final_val_seam_loss": final_val_seam,
        "final_val_horizontal_seam_loss": final_val_h,
        "final_val_vertical_seam_loss": final_val_v,
        "best_val_seam_loss": best_val_seam,
        "best_epoch": best_epoch,
        "seam_improvement_pct": seam_improvement_pct,
        "train_improved": train_improved,
        "val_improved": val_improved,
        "acceptance_criteria_met": train_improved and val_improved,
        "initial_train_rec_mse": initial_train_rec,
        "initial_val_rec_mse": initial_val_rec,
        "final_train_rec_mse": final_train_rec,
        "final_val_rec_mse": final_val_rec,
        "best_val_rec_mse": best_val_rec,
        "visualization_file": str(fig_path),
        "total_runtime": total_runtime,
        "history": history,
    }

    # 6. Non-destructive JSON Logging
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
    """CLI entry point for running the Milestone 4 toy seam consistency experiment."""
    parser = argparse.ArgumentParser(description="PHQD Module 3 Milestone 4: Seam-Consistency Loss Experiment")
    parser.add_argument("--config", type=str, default="configs/module3_seam_loss.yaml", help="Path to config YAML")
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

    run_toy_seam_experiment(config_or_path=config, device_name=args.device)


if __name__ == "__main__":
    main()
