"""
Module 4.2: Full Multi-Seed Training, Benchmarking, and Ablation Experiment Runner.

Executes the complete experimental protocol for Module 4.2:
1. 5 Locked Models:
   - Reference HQCLD (denoiser_matched)
   - MLP Denoiser (denoiser_matched)
   - Fourier-MLP Denoiser (denoiser_matched)
   - Dequantized Surrogate Denoiser (denoiser_matched)
   - Full Classical U-Net
2. 3 Random Seeds: [42, 1042, 2042]
   - Fully independent dataset and parameter initialization per seed
   - Mean +/- standard deviation reporting (no declaring winner from single seed)
3. Mandatory Ablations:
   - Matching Tier Ablation: 'denoiser_matched' (~786 params) vs 'quantum_matched' (18 params)
   - Compute Accounting Sensitivity: kappa in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
4. Comprehensive Metric Suite:
   - Image quality: PSNR, SSIM, LPIPS, FID
   - Computational: Trainable parameters, system parameters, inference latency, throughput
   - Compute Accounting: C_cls FLOPs, C_sim, C_train(kappa)
"""

import sys
import os
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
from experiments.module4.train_classical_baselines import (
    generate_paired_diffusion_dataset,
    create_model_by_name,
    train_single_model,
    parse_yaml_config,
)
from experiments.module4.benchmark_baselines import assemble_generated_images
from src.models.hqcld_denoiser import HQCLDDenoiser
from src.models.baselines.mlp_denoiser import MLPDenoiser
from src.models.baselines.fourier_mlp_denoiser import FourierMLPDenoiser
from src.models.baselines.dequantized_surrogate import DequantizedSurrogateDenoiser


def run_preflight_checks(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Execute preflight checks and report exact parameter counts, FLOPs, and C_train."""
    print("================================================================================")
    print("MODULE 4.2 PREFLIGHT & CONFIGURATION REPORT")
    print("================================================================================")
    
    # 1. Models & Parameter counts
    models = {
        "HQCLD (denoiser_matched)": create_model_by_name("hqcld", config, device),
        "MLP (denoiser_matched)": create_model_by_name("mlp", config, device),
        "Fourier-MLP (denoiser_matched)": create_model_by_name("fourier_mlp", config, device),
        "Dequantized Surrogate (denoiser_matched)": create_model_by_name("dequantized_surrogate", config, device),
        "Full Classical U-Net": create_model_by_name("unet", config, device),
        # Quantum matched ablation tier
        "HQCLD (quantum_matched)": HQCLDDenoiser(adapter_mode="quantum_matched").to(device),
        "MLP (quantum_matched)": MLPDenoiser(matching_tier="quantum_matched").to(device),
        "Fourier-MLP (quantum_matched)": FourierMLPDenoiser(matching_tier="quantum_matched").to(device),
        "Dequantized Surrogate (quantum_matched)": DequantizedSurrogateDenoiser(matching_tier="quantum_matched").to(device),
    }

    stitching_cnn = ClassicalStitchingCNN().to(device)
    stitch_params = stitching_cnn.num_trainable_parameters

    batch_size = config["training"]["batch_size"]
    n_epochs = config["training"]["epochs"]
    n_batch = config["dataset"]["train_samples"] // batch_size
    kappa_default = config["compute_accounting"]["kappa"]

    dummy_patches = torch.randn(batch_size, 9, 64, device=device)
    dummy_latents = torch.randn(batch_size, 4, 8, 8, device=device)
    dummy_t = torch.full((batch_size,), 500, dtype=torch.long, device=device)

    preflight_info = {}
    print(f"Dataset Configuration:")
    print(f"  Train Samples: {config['dataset']['train_samples']} | Val Samples: {config['dataset']['val_samples']}")
    print(f"  Latent Shape: [{config['dataset']['channels']}, {config['dataset']['height']}, {config['dataset']['width']}]")
    print(f"  Diffusion Schedule: Linear (T={config['diffusion']['timesteps']}, beta_start={config['diffusion']['beta_start']}, beta_end={config['diffusion']['beta_end']})")
    print(f"  Training: {n_epochs} epochs, batch size {batch_size} ({n_batch} batches/epoch)")
    print(f"  Optimizer: {config['training']['optimizer']}, lr={config['training']['learning_rate']}")
    print(f"  Seeds: {config['training']['seeds']}")
    print(f"  Compute Accounting: kappa={kappa_default}, sweep={config['compute_accounting']['sweep_kappa']}")
    print("-" * 80)
    print(f"{'Model Name':<42} | {'Trainable Params':<16} | {'Total System':<12} | {'C_train (FLOP-equiv)':<20}")
    print("-" * 80)

    for name, model in models.items():
        p = count_parameters(model)
        is_unet = "U-Net" in name
        tot_p = p if is_unet else p + stitch_params
        inp = dummy_latents if is_unet else dummy_patches
        flops = profile_model_flops(model, inp, dummy_t)
        c_cls = float(flops["forward_backward_flops"])
        p_pqc = 18 if "HQCLD" in name else 0
        cost = compute_training_cost(
            n_epochs=n_epochs,
            n_batch=n_batch,
            c_cls=c_cls,
            p_params=p_pqc,
            kappa=kappa_default,
        )
        preflight_info[name] = {
            "trainable_params": p,
            "total_system_params": tot_p,
            "c_cls_flops": c_cls,
            "c_train": cost["c_train"],
        }
        print(f"{name:<42} | {p:<16,d} | {tot_p:<12,d} | {cost['c_train']:<20,.0f}")

    print("================================================================================\n")
    return preflight_info


def train_and_benchmark_multiseed(
    config_path: Path,
    smoke_test: bool = False,
    run_ablations: bool = True,
    compute_fid: bool = True,
    eval_samples: int = 64,
) -> Dict[str, Any]:
    """Train all models across all seeds and evaluate comprehensive benchmark metrics."""
    config = parse_yaml_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 1 if smoke_test else config["training"]["epochs"]
    batch_size = 8 if smoke_test else config["training"]["batch_size"]
    train_samples = 32 if smoke_test else config["dataset"]["train_samples"]
    val_samples = 16 if smoke_test else config["dataset"]["val_samples"]
    seeds = [42] if smoke_test else config["training"]["seeds"]
    n_eval = 8 if smoke_test else eval_samples

    ckpt_dir = PROJECT_ROOT / config["output"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / config["output"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = PROJECT_ROOT / config["output"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run and verify preflight checks
    preflight_info = run_preflight_checks(config, device)

    # Load frozen components
    ae_path = PROJECT_ROOT / "checkpoints/best_autoencoder.pt"
    if ae_path.exists():
        autoencoder, _ = load_frozen_autoencoder(ae_path, device=device)
    else:
        autoencoder = ConvAutoencoder(in_channels=1, latent_channels=4).to(device).eval()

    stitching_cnn = ClassicalStitchingCNN().to(device).eval()
    evaluator = GenerationEvaluator(device=device)

    # Reference evaluation set (shared across all models for fair paired comparison)
    gen_eval = torch.Generator().manual_seed(9999)
    ref_eval_latents = torch.randn(n_eval, 4, 8, 8, generator=gen_eval, device=device)
    with torch.no_grad():
        ref_eval_images = torch.clamp(autoencoder.decode(ref_eval_latents), 0.0, 1.0)
    ref_eval_patches = extract_latent_patches(ref_eval_latents).view(n_eval, 9, 64)
    eval_t = torch.full((n_eval,), 500, dtype=torch.long, device=device)

    eval_noise_patches = 0.25 * torch.randn_like(ref_eval_patches)
    noisy_eval_patches = ref_eval_patches + eval_noise_patches
    eval_noise_latents = 0.25 * torch.randn_like(ref_eval_latents)
    noisy_eval_latents = ref_eval_latents + eval_noise_latents

    core_models = ["hqcld", "mlp", "fourier_mlp", "dequantized_surrogate", "unet"]
    multiseed_results: Dict[str, Dict[int, Any]] = {name: {} for name in core_models}

    # 2. Main Multi-Seed Training Loop
    print(f"Starting Multi-Seed Training across {len(seeds)} seeds: {seeds}...")
    for seed in seeds:
        print(f"\n==================== SEED {seed} ====================")
        # Dedicated deterministic dataset per seed
        train_data = generate_paired_diffusion_dataset(
            num_samples=train_samples,
            seed=seed,
            timesteps=config["diffusion"]["timesteps"],
        )
        val_data = generate_paired_diffusion_dataset(
            num_samples=val_samples,
            seed=seed + 1000,
            timesteps=config["diffusion"]["timesteps"],
        )

        patch_tr_ds = TensorDataset(train_data["xt_patches"], train_data["x0_patches"], train_data["t"])
        patch_va_ds = TensorDataset(val_data["xt_patches"], val_data["x0_patches"], val_data["t"])
        patch_tr_loader = DataLoader(patch_tr_ds, batch_size=batch_size, shuffle=True)
        patch_va_loader = DataLoader(patch_va_ds, batch_size=batch_size, shuffle=False)

        latent_tr_ds = TensorDataset(train_data["zt"], train_data["z0"], train_data["t"])
        latent_va_ds = TensorDataset(val_data["zt"], val_data["z0"], val_data["t"])
        latent_tr_loader = DataLoader(latent_tr_ds, batch_size=batch_size, shuffle=True)
        latent_va_loader = DataLoader(latent_va_ds, batch_size=batch_size, shuffle=False)

        for name in core_models:
            print(f"--> Training {name.upper()} (Seed={seed}, Epochs={epochs})...")
            # Set seed for model initialization reproducibility
            torch.manual_seed(seed)
            model = create_model_by_name(name, config, device)
            is_unet = (name == "unet")
            tr_loader = latent_tr_loader if is_unet else patch_tr_loader
            va_loader = latent_va_loader if is_unet else patch_va_loader
            ckpt_path = ckpt_dir / f"best_{name}_seed{seed}.pt"

            train_res = train_single_model(
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

            # Benchmarking
            inp = noisy_eval_latents if is_unet else noisy_eval_patches
            gen_images = assemble_generated_images(name, model, inp, eval_t, stitching_cnn, autoencoder, device)

            eval_res = evaluator.evaluate(
                gen_images=gen_images,
                ref_images=ref_eval_images,
                paired=True,
                compute_fid_metric=compute_fid,
                compute_lpips_metric=True,
            )

            sample_in = inp[:2]
            runtime = measure_inference_runtime(
                model,
                sample_in,
                eval_t[:2],
                num_warmup=2,
                num_runs=5,
                device=device,
                batch_size=2,
            )

            # FLOP profiling & compute accounting
            flop_summary = profile_model_flops(model, sample_in, eval_t[:2])
            c_cls = float(flop_summary["forward_backward_flops"])
            n_batch = train_samples // batch_size
            cost_summary = compute_training_cost(
                n_epochs=epochs,
                n_batch=n_batch,
                c_cls=c_cls,
                p_params=18 if name == "hqcld" else 0,
                kappa=config["compute_accounting"]["kappa"],
            )

            record = {
                "seed": seed,
                "model_name": name,
                "best_epoch": train_res["best_epoch"],
                "best_val_loss": train_res["best_val_loss"],
                "train_wallclock_seconds": train_res["total_wallclock_seconds"],
                "trainable_parameters": train_res["trainable_parameters"],
                "total_system_parameters": train_res["trainable_parameters"] + (0 if is_unet else stitching_cnn.num_trainable_parameters),
                "psnr": eval_res["image_metrics"].get("psnr"),
                "ssim": eval_res["image_metrics"].get("ssim"),
                "lpips": eval_res["image_metrics"].get("lpips"),
                "fid": eval_res["image_metrics"].get("fid"),
                "inference_latency_mean_s": runtime["mean_time_seconds"],
                "inference_samples_per_s": runtime.get("samples_per_second"),
                "c_cls_flops": c_cls,
                "c_train": cost_summary["c_train"],
            }
            multiseed_results[name][seed] = record
            print(f"    Done {name}: ValLoss={record['best_val_loss']:.5f} | PSNR={record['psnr']:.2f}dB | SSIM={record['ssim']:.4f} | LPIPS={record['lpips']:.4f} | FID={record['fid']} | Time={record['train_wallclock_seconds']:.2f}s")

    # 3. Ablation 1: Parameter Matching Tier (Quantum-Matched vs Denoiser-Matched)
    ablation_tier_results = {}
    if run_ablations:
        print("\n==================== ABLATION: PARAMETER MATCHING TIERS ====================")
        qm_models = {
            "hqcld_qm": HQCLDDenoiser(adapter_mode="quantum_matched"),
            "mlp_qm": MLPDenoiser(matching_tier="quantum_matched"),
            "fourier_mlp_qm": FourierMLPDenoiser(matching_tier="quantum_matched"),
            "dequantized_surrogate_qm": DequantizedSurrogateDenoiser(matching_tier="quantum_matched"),
        }
        for qm_name, qm_model in qm_models.items():
            qm_model = qm_model.to(device)
            p = count_parameters(qm_model)
            print(f"--> Training Quantum-Matched Baseline {qm_name.upper()} (P={p})...")
            # Train on seed 42
            ckpt_path = ckpt_dir / f"best_{qm_name}.pt"
            tr_res = train_single_model(
                model_name=qm_name,
                model=qm_model,
                train_loader=patch_tr_loader,
                val_loader=patch_va_loader,
                is_unet=False,
                epochs=epochs,
                lr=config["training"]["learning_rate"],
                device=device,
                save_checkpoint_path=ckpt_path,
            )
            gen_img = assemble_generated_images(qm_name, qm_model, noisy_eval_patches, eval_t, stitching_cnn, autoencoder, device)
            ev_res = evaluator.evaluate(
                gen_images=gen_img,
                ref_images=ref_eval_images,
                paired=True,
                compute_fid_metric=compute_fid,
                compute_lpips_metric=True,
            )
            flops = profile_model_flops(qm_model, noisy_eval_patches[:2], eval_t[:2])
            c_cls = float(flops["forward_backward_flops"])
            n_batch = train_samples // batch_size
            cost = compute_training_cost(
                n_epochs=epochs,
                n_batch=n_batch,
                c_cls=c_cls,
                p_params=18 if "hqcld" in qm_name else 0,
                kappa=config["compute_accounting"]["kappa"],
            )
            ablation_tier_results[qm_name] = {
                "model_name": qm_name,
                "tier": "quantum_matched",
                "trainable_parameters": p,
                "best_val_loss": tr_res["best_val_loss"],
                "psnr": ev_res["image_metrics"].get("psnr"),
                "ssim": ev_res["image_metrics"].get("ssim"),
                "lpips": ev_res["image_metrics"].get("lpips"),
                "fid": ev_res["image_metrics"].get("fid"),
                "c_cls_flops": c_cls,
                "c_train": cost["c_train"],
            }
            print(f"    Done {qm_name}: Params={p} | ValLoss={tr_res['best_val_loss']:.5f} | PSNR={ev_res['image_metrics']['psnr']:.2f}dB | SSIM={ev_res['image_metrics']['ssim']:.4f}")

    # 4. Ablation 2: Kappa Sensitivity Sweep for Compute Accounting
    print("\n==================== ABLATION: KAPPA COMPUTE SENSITIVITY SWEEP ====================")
    kappa_sweep_values = config["compute_accounting"].get("sweep_kappa", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    kappa_sweep_results = {}
    n_batch = train_samples // batch_size
    for name in core_models:
        c_cls = multiseed_results[name][seeds[0]]["c_cls_flops"]
        p_pqc = 18 if name == "hqcld" else 0
        sweep_data = sweep_kappa(
            n_epochs=epochs,
            n_batch=n_batch,
            c_cls=c_cls,
            p_params=p_pqc,
            kappa_values=kappa_sweep_values,
        )
        kappa_sweep_results[name] = {
            k_val: next(item["c_train"] for item in sweep_data if item["kappa"] == k_val)
            for k_val in kappa_sweep_values
        }
        print(f"Model {name.upper()}:")
        for k_val in kappa_sweep_values:
            print(f"  kappa={k_val:4.1f} -> C_train={kappa_sweep_results[name][k_val]:,.0f}")

    # 5. Summary Statistics Calculation (Mean +/- Std across seeds)
    summary_stats = {}
    for name in core_models:
        seed_records = list(multiseed_results[name].values())
        def get_stat(key):
            vals = [r[key] for r in seed_records if r[key] is not None]
            if not vals:
                return {"mean": None, "std": None}
            t = torch.tensor(vals, dtype=torch.float64)
            return {
                "mean": float(t.mean().item()),
                "std": float(t.std(unbiased=True).item()) if len(vals) > 1 else 0.0,
            }

        summary_stats[name] = {
            "trainable_parameters": seed_records[0]["trainable_parameters"],
            "total_system_parameters": seed_records[0]["total_system_parameters"],
            "best_val_loss": get_stat("best_val_loss"),
            "train_wallclock_seconds": get_stat("train_wallclock_seconds"),
            "psnr": get_stat("psnr"),
            "ssim": get_stat("ssim"),
            "lpips": get_stat("lpips"),
            "fid": get_stat("fid"),
            "inference_latency_mean_s": get_stat("inference_latency_mean_s"),
            "inference_samples_per_s": get_stat("inference_samples_per_s"),
            "c_cls_flops": seed_records[0]["c_cls_flops"],
            "c_train": seed_records[0]["c_train"],
        }

    # 6. Save JSON & CSV Artifacts
    full_output = {
        "config": config,
        "preflight": preflight_info,
        "multiseed_runs": multiseed_results,
        "summary_statistics": summary_stats,
        "ablation_tiers": ablation_tier_results,
        "ablation_kappa_sweep": kappa_sweep_results,
    }

    log_path = logs_dir / "module4_multiseed_results.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved full multi-seed results to {log_path}")

    # Also update module4_baseline_benchmarks.json with the summary for backward compatibility
    bench_log = logs_dir / config["output"]["results_filename"]
    compat_summary = {}
    for name, s in summary_stats.items():
        compat_summary[name] = {
            "model_name": name,
            "trainable_parameters": s["trainable_parameters"],
            "total_system_parameters": s["total_system_parameters"],
            "c_cls_flops": s["c_cls_flops"],
            "c_train": s["c_train"],
            "kappa": config["compute_accounting"]["kappa"],
            "psnr": s["psnr"]["mean"],
            "psnr_std": s["psnr"]["std"],
            "ssim": s["ssim"]["mean"],
            "ssim_std": s["ssim"]["std"],
            "lpips": s["lpips"]["mean"],
            "lpips_std": s["lpips"]["std"],
            "fid": s["fid"]["mean"],
            "fid_std": s["fid"]["std"],
            "latency_mean_seconds": s["inference_latency_mean_s"]["mean"],
            "samples_per_second": s["inference_samples_per_s"]["mean"],
        }
    with open(bench_log, "w", encoding="utf-8") as f:
        json.dump(compat_summary, f, indent=2)
    print(f"Saved benchmark summary to {bench_log}")

    # CSV Table 1: Multi-Seed Summary
    csv_multiseed = tables_dir / "module4_multiseed_summary.csv"
    with open(csv_multiseed, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Trainable Params", "Total System Params",
            "Val Loss (Mean+/-Std)", "PSNR (dB)", "SSIM", "LPIPS", "FID",
            "Train Time (s)", "Latency (s)", "Throughput (samples/s)", "C_cls FLOPs", "C_train (kappa=1)"
        ])
        for name, s in summary_stats.items():
            writer.writerow([
                name.upper(),
                s["trainable_parameters"],
                s["total_system_parameters"],
                f"{s['best_val_loss']['mean']:.4f} +/- {s['best_val_loss']['std']:.4f}",
                f"{s['psnr']['mean']:.2f} +/- {s['psnr']['std']:.2f}",
                f"{s['ssim']['mean']:.4f} +/- {s['ssim']['std']:.4f}",
                f"{s['lpips']['mean']:.4f} +/- {s['lpips']['std']:.4f}",
                f"{s['fid']['mean']:.2f} +/- {s['fid']['std']:.2f}" if s['fid']['mean'] is not None else "N/A",
                f"{s['train_wallclock_seconds']['mean']:.2f} +/- {s['train_wallclock_seconds']['std']:.2f}",
                f"{s['inference_latency_mean_s']['mean']:.6f}",
                f"{s['inference_samples_per_s']['mean']:.1f}",
                f"{s['c_cls_flops']:.0f}",
                f"{s['c_train']:.0f}",
            ])
    print(f"Saved multi-seed summary CSV to {csv_multiseed}")

    # CSV Table 2: Kappa Sweep
    csv_kappa = tables_dir / "module4_kappa_sweep.csv"
    with open(csv_kappa, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + [f"kappa={k}" for k in kappa_sweep_values])
        for name in core_models:
            row = [name.upper()] + [f"{kappa_sweep_results[name][k]:.0f}" for k in kappa_sweep_values]
            writer.writerow(row)
    print(f"Saved kappa sweep CSV to {csv_kappa}")

    # CSV Table 3: Tier Ablation
    if run_ablations and ablation_tier_results:
        csv_tier = tables_dir / "module4_tier_ablation.csv"
        with open(csv_tier, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Tier", "Trainable Params", "Val Loss", "PSNR (dB)", "SSIM", "LPIPS", "FID", "C_train"])
            for name, s in summary_stats.items():
                if name != "unet":
                    writer.writerow([
                        name.upper(), "denoiser_matched", s["trainable_parameters"],
                        f"{s['best_val_loss']['mean']:.4f}", f"{s['psnr']['mean']:.2f}",
                        f"{s['ssim']['mean']:.4f}", f"{s['lpips']['mean']:.4f}",
                        f"{s['fid']['mean']:.2f}" if s['fid']['mean'] is not None else "N/A",
                        f"{s['c_train']:.0f}"
                    ])
            for qm_name, r in ablation_tier_results.items():
                writer.writerow([
                    qm_name.upper(), r["tier"], r["trainable_parameters"],
                    f"{r['best_val_loss']:.4f}", f"{r['psnr']:.2f}",
                    f"{r['ssim']:.4f}", f"{r['lpips']:.4f}",
                    f"{r['fid']:.2f}" if r['fid'] is not None else "N/A",
                    f"{r['c_train']:.0f}"
                ])
        print(f"Saved tier ablation CSV to {csv_tier}")

    print("\n================================================================================")
    print("MODULE 4.2 FULL EXPERIMENT COMPLETED SUCCESSFULLY")
    print("================================================================================")
    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4.2 Multi-Seed Training & Benchmark Runner")
    parser.add_argument("--config", type=str, default="configs/module4_baselines.yaml", help="Path to config")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1-epoch smoke test across models")
    parser.add_argument("--no-ablations", action="store_true", help="Skip tier ablations")
    parser.add_argument("--no-fid", action="store_true", help="Skip FID calculation")
    parser.add_argument("--eval-samples", type=int, default=64, help="Number of evaluation samples for image metrics")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    train_and_benchmark_multiseed(
        config_path=cfg_path,
        smoke_test=args.smoke_test,
        run_ablations=not args.no_ablations,
        compute_fid=not args.no_fid,
        eval_samples=args.eval_samples,
    )
