"""
Module 4.2: Classical Baseline & HQCLD Benchmark Evaluation.

Evaluates trained or initialized models on:
1. Image-Space Quality:
   - FID (Fréchet Inception Distance)
   - LPIPS (SqueezeNet v0.1 perceptual distance)
   - PSNR (Peak Signal-to-Noise Ratio in dB)
   - SSIM (Structural Similarity Index Measure)
2. Computational Efficiency:
   - Trainable parameter count
   - Wall-clock inference latency (mean, std, min, max)
   - Throughput (samples/second)
3. Compute Accounting (§3.6):
   - Classical FLOP count C_cls (forward+backward training step)
   - Compute-normalized cost C_train across configurable kappa values

Saves results to:
- results/tables/module4_baseline_benchmarks.csv
- results/logs/module4_baseline_benchmarks.json
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import ConvAutoencoder
from src.data.patch_extractor import load_frozen_autoencoder, extract_latent_patches
from src.models.stitching_cnn import ClassicalStitchingCNN
from src.evaluation.evaluator import GenerationEvaluator
from src.evaluation.computational_metrics import count_parameters, measure_inference_runtime
from src.evaluation.flop_counter import profile_model_flops
from src.evaluation.compute_cost import compute_training_cost, sweep_kappa
from experiments.module4.train_classical_baselines import create_model_by_name, parse_yaml_config


def assemble_generated_images(
    model_name: str,
    model: nn.Module,
    noisy_inputs: torch.Tensor,
    t: torch.Tensor,
    stitching_cnn: ClassicalStitchingCNN,
    autoencoder: ConvAutoencoder,
    device: torch.device,
) -> torch.Tensor:
    """Generate image batch [B, 1, 28, 28] through the model's respective pipeline."""
    model.eval()
    stitching_cnn.eval()
    autoencoder.eval()

    with torch.no_grad():
        noisy_inputs = noisy_inputs.to(device)
        t = t.to(device)

        if model_name == "unet":
            # U-Net: [B, 4, 8, 8] + t -> [B, 4, 8, 8] -> AE decode -> [B, 1, 28, 28]
            clean_latent = model(noisy_inputs, t)
            images = autoencoder.decode(clean_latent)
        else:
            # Patch models: [B, 9, 64] + t -> [B, 9, 64] -> reshape [B, 9, 4, 4, 4]
            # -> stitching CNN -> [B, 4, 8, 8] -> AE decode -> [B, 1, 28, 28]
            clean_patches_flat = model(noisy_inputs, t)
            b = clean_patches_flat.size(0)
            clean_patches_5d = clean_patches_flat.view(b, 9, 4, 4, 4)
            stitched_latent = stitching_cnn(clean_patches_5d)
            images = autoencoder.decode(stitched_latent)

    return torch.clamp(images, 0.0, 1.0)


def benchmark_all_models(
    config_path: Path,
    num_eval_samples: int = 64,
    kappa: float = 1.0,
    compute_fid_metric: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run comprehensive benchmark across HQCLD and 4 classical baselines."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = parse_yaml_config(config_path)
    ckpt_dir = PROJECT_ROOT / config["output"]["checkpoints_dir"]

    # Load frozen components
    ae_path = PROJECT_ROOT / "checkpoints/best_autoencoder.pt"
    if ae_path.exists():
        autoencoder, _ = load_frozen_autoencoder(ae_path, device=device)
    else:
        autoencoder = ConvAutoencoder(in_channels=1, latent_channels=4).to(device).eval()

    stitching_cnn = ClassicalStitchingCNN().to(device).eval()
    evaluator = GenerationEvaluator(device=device)

    # Deterministic evaluation latents
    gen = torch.Generator().manual_seed(9999)
    ref_latents = torch.randn(num_eval_samples, 4, 8, 8, generator=gen, device=device)
    with torch.no_grad():
        ref_images = torch.clamp(autoencoder.decode(ref_latents), 0.0, 1.0)

    # Patch inputs
    ref_patches = extract_latent_patches(ref_latents).view(num_eval_samples, 9, 64)
    eval_t = torch.full((num_eval_samples,), 500, dtype=torch.long, device=device)

    # Additive noise for evaluation
    noise_patches = 0.25 * torch.randn_like(ref_patches)
    noisy_patches = ref_patches + noise_patches

    noise_latents = 0.25 * torch.randn_like(ref_latents)
    noisy_latents = ref_latents + noise_latents

    models_to_bench = ["hqcld", "mlp", "fourier_mlp", "dequantized_surrogate", "unet"]
    benchmark_results = {}

    epochs = config["training"]["epochs"]
    n_batch = config["dataset"]["train_samples"] // config["training"]["batch_size"]

    for name in models_to_bench:
        model = create_model_by_name(name, config, device)
        ckpt_file = ckpt_dir / f"best_{name}.pt"
        if ckpt_file.exists():
            payload = torch.load(ckpt_file, map_location=device, weights_only=False)
            model.load_state_dict(payload["state_dict"])

        # 1. Parameter counts
        trainable_params = count_parameters(model)
        total_system_params = trainable_params + (0 if name == "unet" else stitching_cnn.num_trainable_parameters)

        # 2. FLOP profiling
        sample_in = noisy_latents[:2] if name == "unet" else noisy_patches[:2]
        flop_summary = profile_model_flops(model, sample_in, eval_t[:2])
        c_cls = float(flop_summary["forward_backward_flops"])

        # 3. Compute cost §3.6
        p_params = 18 if name == "hqcld" else 0
        cost_summary = compute_training_cost(
            n_epochs=epochs,
            n_batch=n_batch,
            c_cls=c_cls,
            p_params=p_params,
            n_qubits=6,
            depth=3,
            num_patches=9,
            kappa=kappa,
        )

        # 4. Generate images
        inp = noisy_latents if name == "unet" else noisy_patches
        gen_images = assemble_generated_images(name, model, inp, eval_t, stitching_cnn, autoencoder, device)

        # 5. Evaluation metrics
        eval_metrics = evaluator.evaluate(
            gen_images=gen_images,
            ref_images=ref_images,
            paired=True,
            compute_fid_metric=compute_fid_metric,
            compute_lpips_metric=True,
        )

        # 6. Inference runtime
        runtime = measure_inference_runtime(
            model,
            sample_in,
            eval_t[:2],
            num_warmup=2,
            num_runs=5,
            device=device,
            batch_size=2,
        )

        benchmark_results[name] = {
            "model_name": name,
            "trainable_parameters": trainable_params,
            "total_system_parameters": total_system_params,
            "c_cls_flops": c_cls,
            "c_train": cost_summary["c_train"],
            "kappa": kappa,
            "psnr": eval_metrics["image_metrics"].get("psnr", None),
            "ssim": eval_metrics["image_metrics"].get("ssim", None),
            "lpips": eval_metrics["image_metrics"].get("lpips", None),
            "fid": eval_metrics["image_metrics"].get("fid", None),
            "latency_mean_seconds": runtime["mean_time_seconds"],
            "samples_per_second": runtime.get("samples_per_second", None),
        }

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Module 4.2 Baseline Benchmarking")
    parser.add_argument("--config", type=str, default="configs/module4_baselines.yaml")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--fid", action="store_true", help="Compute Inception FID")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    results = benchmark_all_models(
        config_path=config_path,
        num_eval_samples=args.samples,
        kappa=args.kappa,
        compute_fid_metric=args.fid,
    )

    out_file = PROJECT_ROOT / "results/logs/module4_baseline_benchmarks.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark written to {out_file}")


if __name__ == "__main__":
    main()
