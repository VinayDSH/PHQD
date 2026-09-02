"""
Export script for PHQD Module 1 Latent Patches.

Encodes Fashion-MNIST images into continuous 4x8x8 latents using a frozen,
trained ConvAutoencoder, extracts overlapping 4x4 spatial patches with stride 2
(preserving the channel dimension as [N, 9, 4, 4, 4]), and saves them along with
reproducibility metadata under data/latent_patches/.
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import get_fashion_mnist_dataloaders
from src.data.patch_extractor import LatentPatchExtractor, load_frozen_autoencoder
from src.training.train_autoencoder import parse_yaml_config, set_seed


def export_split_patches(
    extractor: LatentPatchExtractor,
    dataloader: torch.utils.data.DataLoader,
    split_name: str,
    output_dir: Path,
    metadata_base: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> Path:
    """Export latent patches for a single data split and save to disk."""
    print(f"\n--- Processing split: {split_name} ---")
    start_t = time.time()

    patches, labels = extractor.extract_from_loader(dataloader, max_samples=max_samples)
    elapsed = time.time() - start_t

    # Validate output tensor shape: [N, 9, 4, 4, 4]
    expected_patches = metadata_base["patches_per_image"]
    expected_channels = metadata_base["latent_channels"]
    expected_patch_size = metadata_base["patch_size"]

    expected_shape = (
        patches.size(0),
        expected_patches,
        expected_channels,
        expected_patch_size,
        expected_patch_size,
    )
    assert patches.shape == expected_shape, (
        f"Shape mismatch for {split_name}: expected {expected_shape}, got {patches.shape}"
    )

    split_metadata = {
        **metadata_base,
        "split": split_name,
        "num_samples": patches.size(0),
        "patch_tensor_shape": list(patches.shape),
        "labels_tensor_shape": list(labels.shape),
        "export_duration_seconds": round(elapsed, 2),
    }

    output_file = output_dir / f"{split_name}_patches.pt"
    payload = {
        "patches": patches,  # Shape: [N, 9, 4, 4, 4]
        "labels": labels,    # Shape: [N]
        "metadata": split_metadata,
    }

    torch.save(payload, output_file)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(
        f"Exported {split_name}: {patches.size(0)} images -> "
        f"tensor shape {list(patches.shape)} ({file_size_mb:.2f} MB) in {elapsed:.2f}s"
    )
    print(f"Saved to: {output_file}")
    return output_file


def run_patch_export(
    config_path: str = "configs/module1_fashion_mnist.yaml",
    checkpoint_path: str = "checkpoints/best_autoencoder.pt",
    output_dir: str = "data/latent_patches",
    splits: Optional[list] = None,
    max_samples: Optional[int] = None,
    device_name: Optional[str] = None,
) -> Dict[str, Path]:
    """Execute latent patch extraction pipeline across configured dataset splits."""
    config = parse_yaml_config(config_path)

    # Extract patch and training configuration
    patch_cfg = config.get("patches", {})
    patch_size = patch_cfg.get("patch_size", 4)
    stride = patch_cfg.get("stride", 2)

    training_cfg = config.get("training", {})
    seed = training_cfg.get("seed", 42)
    batch_size = training_cfg.get("batch_size", 128)

    autoencoder_cfg = config.get("autoencoder", {})
    latent_channels = autoencoder_cfg.get("latent_channels", 4)
    latent_h = autoencoder_cfg.get("latent_height", 8)
    latent_w = autoencoder_cfg.get("latent_width", 8)

    # Enforce reproducibility
    set_seed(seed)

    # Device selection
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen autoencoder checkpoint
    model, checkpoint_info = load_frozen_autoencoder(checkpoint_path, device=device)

    # Calculate expected number of patches per image
    n_h = (latent_h - patch_size) // stride + 1
    n_w = (latent_w - patch_size) // stride + 1
    patches_per_image = n_h * n_w

    print("=" * 70)
    print("PHQD Module 1: Latent-Patch Dataset Export")
    print(f"Device: {device.type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint Best Val Loss: {checkpoint_info.get('val_loss', 'N/A')}")
    print(f"Latent Grid: {latent_channels}x{latent_h}x{latent_w}")
    print(f"Patch Geometry: size={patch_size}x{patch_size}, stride={stride} -> {patches_per_image} patches/image")
    print(f"Single Patch Shape: [{latent_channels}, {patch_size}, {patch_size}] ({latent_channels*patch_size*patch_size} values)")
    print(f"Exported Image Shape: [9, {latent_channels}, {patch_size}, {patch_size}]")
    print("=" * 70)

    # Initialize patch extractor
    extractor = LatentPatchExtractor(
        model=model,
        patch_size=patch_size,
        stride=stride,
        device=device,
    )

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        root="data/raw",
        batch_size=batch_size,
        val_size=10000,
        seed=seed,
        download=True,
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    metadata_base = {
        "dataset": config.get("dataset", {}).get("name", "FashionMNIST"),
        "latent_shape": [latent_channels, latent_h, latent_w],
        "latent_channels": latent_channels,
        "patch_size": patch_size,
        "stride": stride,
        "patches_per_image": patches_per_image,
        "patch_shape": [latent_channels, patch_size, patch_size],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_info.get("epoch"),
        "checkpoint_val_loss": checkpoint_info.get("val_loss"),
        "checkpoint_val_psnr": checkpoint_info.get("val_psnr"),
        "checkpoint_val_ssim": checkpoint_info.get("val_ssim"),
        "seed": seed,
        "device": str(device),
    }

    target_splits = splits or ["train", "val", "test"]
    loader_map = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    exported_files = {}
    for split_name in target_splits:
        if split_name not in loader_map:
            raise ValueError(f"Unknown split: {split_name}. Choose from {list(loader_map.keys())}")
        out_file = export_split_patches(
            extractor=extractor,
            dataloader=loader_map[split_name],
            split_name=split_name,
            output_dir=output_dir_path,
            metadata_base=metadata_base,
            max_samples=max_samples,
        )
        exported_files[split_name] = out_file

    print("\n" + "=" * 70)
    print("Patch Export Finished Successfully")
    print(f"Saved artifacts in: {output_dir_path}")
    print("=" * 70)

    return exported_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Fashion-MNIST latent patches")
    parser.add_argument("--config", default="configs/module1_fashion_mnist.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default="checkpoints/best_autoencoder.pt", help="Path to autoencoder checkpoint")
    parser.add_argument("--output-dir", default="data/latent_patches", help="Directory to save exported patches")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all", help="Split to export")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap sample count (for quick tests)")
    parser.add_argument("--device", default=None, help="Device to use ('cuda' or 'cpu')")

    args = parser.parse_args()
    splits_to_run = None if args.split == "all" else [args.split]

    run_patch_export(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        splits=splits_to_run,
        max_samples=args.max_samples,
        device_name=args.device,
    )
