"""
Module 5: Entangled Patch Quantum Diffusion (EPQD) Multi-Seed Experiment Runner.

Executes the locked EPQD experimental protocol:
1. Three Model Configurations:
   - epqd_k1 (Reference single-patch baseline, nq=6, 786 params)
   - epqd_k2 (Entangled patch pairs, nq=6, 807 params)
   - epqd_k4 (Overlapping 2x2 quads, nq=4, 572 params)
2. Three Locked Seeds: [42, 1042, 2042]
   - Fully independent dataset and parameter initialization per seed.
3. Compute Accounting:
   - Nominal 15-epoch (480-step) budget:
     C_train(k=1) = 63,866,880
     C_train(k=2) = 1,900,431,360
     C_train(k=4, nq=4) = 45,698,088,960
     C_train(k=4, nq=6, theoretical) = 16,331,646,320,640
4. Diagnostics Suite:
   - Von Neumann entanglement entropy S(rho_A)
   - Bridge-zeroing ablation Delta L_bridge
   - Seam disagreement D_seam
5. Gate 2 Statistical Verification:
   - Two-sided paired Student's t-test matching exact formula:
     t = d_bar / (s_d / sqrt(3))
   - Power law fit: MSE(k) = MSE_inf + A / k^gamma
   - Compute Efficiency Index eta = Delta_PSNR / log10(C_train(k=2) / C_train(k=1))
6. Compute-Matched Experiment:
   - Evaluated separately (16 steps for k=2 matching k=1 budget) and not in denominator of eta.
"""

import sys
import os
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import ConvAutoencoder
from src.data.patch_extractor import load_frozen_autoencoder, extract_latent_patches
from src.models.stitching_cnn import ClassicalStitchingCNN
from src.evaluation.evaluator import GenerationEvaluator
from src.models.epqd import EPQDDenoiser
from src.evaluation.epqd_diagnostics import (
    compute_von_neumann_entropy,
    evaluate_bridge_zeroing_ablation,
    compute_seam_disagreement,
)
from src.evaluation.epqd_analysis import (
    calculate_epqd_c_train,
    compute_efficiency_index,
    compute_paired_t_test,
    fit_power_law,
    evaluate_gate2_criteria,
)
from experiments.module4.train_classical_baselines import (
    generate_paired_diffusion_dataset,
    parse_yaml_config,
)


def create_epqd_model(model_name: str, config: Dict[str, Any], device: torch.device) -> EPQDDenoiser:
    """Instantiate locked EPQD model configuration."""
    model_cfg = config["models"][model_name]
    model = EPQDDenoiser(
        k=model_cfg["k"],
        n_qubits=model_cfg.get("n_qubits"),
        n_layers=model_cfg.get("n_layers", 3),
        patch_dim=model_cfg.get("patch_dim", 64),
        num_patches=model_cfg.get("num_patches", 9),
    ).to(device)
    return model


def train_epqd_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 15,
    max_steps: Optional[int] = None,
) -> Tuple[List[float], List[float], float]:
    """Train single EPQD model and return train losses, val losses, and total train time."""
    model.train()
    train_loss_history = []
    val_loss_history = []
    t_start = time.time()
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for x_t_batch, x_0_batch, t_batch in train_loader:
            x_t_batch = x_t_batch.to(device)
            x_0_batch = x_0_batch.to(device)
            t_batch = t_batch.to(device)

            optimizer.zero_grad()
            x_pred = model(x_t_batch, t_batch)
            loss = F.mse_loss(x_pred, x_0_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1

            if max_steps is not None and total_steps >= max_steps:
                break

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_loss_history.append(avg_train_loss)

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx_t, vx_0, vt in val_loader:
                vx_t = vx_t.to(device)
                vx_0 = vx_0.to(device)
                vt = vt.to(device)
                v_pred = model(vx_t, vt)
                val_loss += F.mse_loss(v_pred, vx_0).item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1)
        val_loss_history.append(avg_val_loss)

        if max_steps is not None and total_steps >= max_steps:
            break

    total_time = time.time() - t_start
    return train_loss_history, val_loss_history, total_time


def evaluate_epqd_model(
    model: nn.Module,
    val_loader: DataLoader,
    stitching_cnn: ClassicalStitchingCNN,
    autoencoder: ConvAutoencoder,
    evaluator: GenerationEvaluator,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate image quality metrics and spatial diagnostics on validation dataset."""
    model.eval()
    stitching_cnn.eval()
    autoencoder.eval()

    all_pred_patches = []
    all_clean_patches = []
    all_noisy_patches = []
    all_t = []

    with torch.no_grad():
        for x_t, x_0, t in val_loader:
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            t = t.to(device)
            pred = model(x_t, t)
            all_pred_patches.append(pred)
            all_clean_patches.append(x_0)
            all_noisy_patches.append(x_t)
            all_t.append(t)

    pred_patches = torch.cat(all_pred_patches, dim=0)
    clean_patches = torch.cat(all_clean_patches, dim=0)
    noisy_patches = torch.cat(all_noisy_patches, dim=0)
    t_tensor = torch.cat(all_t, dim=0)

    # 1. Patch MSE Loss
    mse_val = float(F.mse_loss(pred_patches, clean_patches).item())

    # 2. Seam Disagreement D_seam on predicted patches
    d_seam = compute_seam_disagreement(pred_patches)

    # 3. Downstream Image Reconstruction
    b = pred_patches.size(0)
    pred_5d = pred_patches.view(b, 9, 4, 4, 4)
    clean_5d = clean_patches.view(b, 9, 4, 4, 4)

    stitched_pred = stitching_cnn(pred_5d)
    stitched_clean = stitching_cnn(clean_5d)

    recon_images = torch.clamp(autoencoder.decode(stitched_pred), 0.0, 1.0)
    target_images = torch.clamp(autoencoder.decode(stitched_clean), 0.0, 1.0)

    # 4. Evaluator Image Metrics
    eval_res = evaluator.evaluate(recon_images, target_images)
    img_metrics = eval_res.get("image_metrics", eval_res)

    # 5. Quantum Diagnostics
    entropy_val = compute_von_neumann_entropy(model, noisy_patches[:4], t_tensor[:4])
    delta_bridge = evaluate_bridge_zeroing_ablation(
        model, noisy_patches[:16], t_tensor[:16], target_clean=clean_patches[:16]
    )

    return {
        "val_mse": mse_val,
        "d_seam": d_seam,
        "psnr": float(img_metrics.get("psnr", 0.0)),
        "ssim": float(img_metrics.get("ssim", 0.0)),
        "lpips": float(img_metrics.get("lpips", 0.0)),
        "fid": float(img_metrics.get("fid", 0.0)),
        "von_neumann_entropy": entropy_val,
        "delta_loss_bridge": delta_bridge,
    }


def run_full_epqd_experiment(
    config_path: Path,
    device: Optional[torch.device] = None,
    seeds: Optional[List[int]] = None,
    models_to_run: Optional[List[str]] = None,
    run_compute_matched: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute complete Module 5 EPQD multi-seed experiment and statistical analysis."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = parse_yaml_config(config_path)
    if seeds is not None:
        config["training"]["seeds"] = seeds
    if models_to_run is None:
        models_to_run = ["epqd_k1", "epqd_k2", "epqd_k4"]

    epochs = 1 if dry_run else config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]

    output_cfg = config.get("output", {})
    ckpt_dir = PROJECT_ROOT / output_cfg.get("checkpoints_dir", "checkpoints/module5")
    logs_dir = PROJECT_ROOT / output_cfg.get("logs_dir", "results/logs")
    tables_dir = PROJECT_ROOT / output_cfg.get("tables_dir", "results/tables")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen downstream components
    ae_path = PROJECT_ROOT / "checkpoints/best_autoencoder.pt"
    if ae_path.exists():
        autoencoder, _ = load_frozen_autoencoder(ae_path, device=device)
    else:
        autoencoder = ConvAutoencoder(in_channels=1, latent_channels=4).to(device).eval()

    stitching_cnn = ClassicalStitchingCNN().to(device).eval()
    evaluator = GenerationEvaluator(device=device)

    c_train_dict = calculate_epqd_c_train(
        n_epochs=config["training"]["epochs"],
        n_batch=config["dataset"]["train_samples"] // batch_size,
        kappa=config.get("compute_accounting", {}).get("kappa", 1.0),
    )

    experiment_results: Dict[str, Any] = {
        "metadata": {
            "module": "Module 5: Entangled Patch Quantum Diffusion (EPQD)",
            "config_path": str(config_path),
            "seeds": config["training"]["seeds"],
            "models": models_to_run,
            "c_train_analytical": c_train_dict,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_seed_results": {},
        "summary": {},
        "diagnostics": {},
        "statistical_evaluation": {},
        "gate2_evaluation": {},
    }

    print("================================================================================")
    print("STARTING MODULE 5: EPQD MULTI-SEED EXPERIMENT")
    print(f"Models: {models_to_run} | Seeds: {config['training']['seeds']} | Epochs: {epochs}")
    print(f"Analytical C_train: k=1: {c_train_dict['k1']}, k=2: {c_train_dict['k2']}, k=4: {c_train_dict['k4_nq4']}")
    print("================================================================================")

    # 1. Main Nominal Multi-Seed Training Loop (Seed-outer for shared dataset per seed)
    for seed in config["training"]["seeds"]:
        print(f"\n==================== SEED {seed} ====================")
        # Dedicated deterministic dataset per seed (identical across all models)
        train_samples = config["dataset"]["train_samples"]
        val_samples = config["dataset"]["val_samples"]
        timesteps = config["diffusion"]["timesteps"]

        train_data = generate_paired_diffusion_dataset(
            num_samples=train_samples,
            seed=seed,
            timesteps=timesteps,
        )
        val_data = generate_paired_diffusion_dataset(
            num_samples=val_samples,
            seed=seed + 1000,
            timesteps=timesteps,
        )

        patch_tr_ds = TensorDataset(train_data["xt_patches"], train_data["x0_patches"], train_data["t"])
        patch_va_ds = TensorDataset(val_data["xt_patches"], val_data["x0_patches"], val_data["t"])

        if dry_run:
            train_loader = DataLoader(patch_tr_ds, batch_size=4, shuffle=False)
            val_loader = DataLoader(patch_va_ds, batch_size=4, shuffle=False)
            train_loader = [next(iter(train_loader))]
            val_loader = [next(iter(val_loader))]
        else:
            train_loader = DataLoader(patch_tr_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(patch_va_ds, batch_size=batch_size, shuffle=False)

        for model_name in models_to_run:
            print(f"  --> Training {model_name} (Seed={seed}, Epochs={epochs})...")
            if model_name not in experiment_results["per_seed_results"]:
                experiment_results["per_seed_results"][model_name] = {}

            torch.manual_seed(seed)
            model = create_epqd_model(model_name, config, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_losses, val_losses, train_time = train_epqd_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                epochs=epochs,
            )

            # Checkpoint saving
            ckpt_path = ckpt_dir / f"{model_name}_seed_{seed}.pt"
            torch.save({
                "model_name": model_name,
                "seed": seed,
                "state_dict": model.state_dict(),
                "pqc_params": model.num_pqc_parameters,
                "trainable_params": model.num_parameters,
            }, ckpt_path)

            # Evaluation
            eval_metrics = evaluate_epqd_model(
                model=model,
                val_loader=val_loader,
                stitching_cnn=stitching_cnn,
                autoencoder=autoencoder,
                evaluator=evaluator,
                device=device,
            )
            eval_metrics["train_time_sec"] = train_time
            eval_metrics["final_train_loss"] = train_losses[-1]
            eval_metrics["trainable_params"] = model.num_parameters
            eval_metrics["pqc_params"] = model.num_pqc_parameters

            experiment_results["per_seed_results"][model_name][str(seed)] = eval_metrics
            print(f"      Seed {seed} -> Val MSE: {eval_metrics['val_mse']:.5f}, PSNR: {eval_metrics['psnr']:.2f} dB, "
                  f"SSIM: {eval_metrics['ssim']:.4f}, LPIPS: {eval_metrics['lpips']:.4f}, D_seam: {eval_metrics['d_seam']:.5f}")

    # 2. Compute Summary Statistics (Mean +/- Std)
    metric_keys = ["val_mse", "psnr", "ssim", "lpips", "fid", "d_seam", "von_neumann_entropy", "delta_loss_bridge", "train_time_sec"]
    for model_name in models_to_run:
        summary_dict = {}
        for k in metric_keys:
            vals = [experiment_results["per_seed_results"][model_name][str(s)][k] for s in config["training"]["seeds"]]
            summary_dict[f"{k}_mean"] = float(np.mean(vals))
            summary_dict[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        experiment_results["summary"][model_name] = summary_dict

    # 3. Statistical Analysis: Paired t-test and Power Law Fit
    if "epqd_k1" in models_to_run and "epqd_k2" in models_to_run:
        psnr_k1 = [experiment_results["per_seed_results"]["epqd_k1"][str(s)]["psnr"] for s in config["training"]["seeds"]]
        psnr_k2 = [experiment_results["per_seed_results"]["epqd_k2"][str(s)]["psnr"] for s in config["training"]["seeds"]]
        ttest_res = compute_paired_t_test(psnr_k1, psnr_k2, alpha=0.05)
        experiment_results["statistical_evaluation"]["paired_t_test_k2_vs_k1"] = ttest_res

        # Power law fit across available k
        k_list = []
        mse_list = []
        for m_name, k_val in [("epqd_k1", 1.0), ("epqd_k2", 2.0), ("epqd_k4", 4.0)]:
            if m_name in experiment_results["summary"]:
                k_list.append(k_val)
                mse_list.append(experiment_results["summary"][m_name]["val_mse_mean"])
        if len(k_list) >= 2:
            power_law_res = fit_power_law(np.array(k_list), np.array(mse_list))
            experiment_results["statistical_evaluation"]["power_law_fit"] = power_law_res

        # Gate 2 Evaluation
        gate2_res = evaluate_gate2_criteria(
            psnr_k1_seeds=psnr_k1,
            psnr_k2_seeds=psnr_k2,
            lpips_k1=experiment_results["summary"]["epqd_k1"]["lpips_mean"],
            lpips_k2=experiment_results["summary"]["epqd_k2"]["lpips_mean"],
            fid_k1=experiment_results["summary"]["epqd_k1"]["fid_mean"],
            fid_k2=experiment_results["summary"]["epqd_k2"]["fid_mean"],
            d_seam_k1=experiment_results["summary"]["epqd_k1"]["d_seam_mean"],
            d_seam_k2=experiment_results["summary"]["epqd_k2"]["d_seam_mean"],
            entropy_k2=experiment_results["summary"]["epqd_k2"]["von_neumann_entropy_mean"],
            delta_loss_bridge=experiment_results["summary"]["epqd_k2"]["delta_loss_bridge_mean"],
            c_train_k1=c_train_dict["k1"],
            c_train_k2=c_train_dict["k2"],
        )
        experiment_results["gate2_evaluation"] = gate2_res

    # 4. Separate Compute-Matched Experiment (k=2 matched to k=1 budget)
    if run_compute_matched and "epqd_k2" in models_to_run:
        matched_steps = config.get("compute_matched", {}).get("k2_matched_steps", 16)
        print(f"\n--- Running Compute-Matched k=2 Experiment ({matched_steps} steps) ---")
        experiment_results["compute_matched_k2"] = {}
        for seed in config["training"]["seeds"]:
            train_data = generate_paired_diffusion_dataset(
                num_samples=train_samples,
                seed=seed,
                timesteps=timesteps,
            )
            val_data = generate_paired_diffusion_dataset(
                num_samples=val_samples,
                seed=seed + 1000,
                timesteps=timesteps,
            )

            patch_tr_ds = TensorDataset(train_data["xt_patches"], train_data["x0_patches"], train_data["t"])
            patch_va_ds = TensorDataset(val_data["xt_patches"], val_data["x0_patches"], val_data["t"])

            if dry_run:
                train_loader = DataLoader(patch_tr_ds, batch_size=4, shuffle=False)
                val_loader = DataLoader(patch_va_ds, batch_size=4, shuffle=False)
                train_loader = [next(iter(train_loader))]
                val_loader = [next(iter(val_loader))]
            else:
                train_loader = DataLoader(patch_tr_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(patch_va_ds, batch_size=batch_size, shuffle=False)

            torch.manual_seed(seed)
            model_matched = create_epqd_model("epqd_k2", config, device)
            opt_matched = torch.optim.Adam(model_matched.parameters(), lr=lr)

            _, _, m_time = train_epqd_single_model(
                model=model_matched,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=opt_matched,
                device=device,
                epochs=1,
                max_steps=matched_steps,
            )

            m_eval = evaluate_epqd_model(
                model=model_matched,
                val_loader=val_loader,
                stitching_cnn=stitching_cnn,
                autoencoder=autoencoder,
                evaluator=evaluator,
                device=device,
            )
            m_eval["train_time_sec"] = m_time
            m_eval["matched_steps"] = matched_steps
            experiment_results["compute_matched_k2"][str(seed)] = m_eval

    # 5. Save Artifacts (JSON & CSV)
    json_path = logs_dir / "module5_epqd_results.json"
    with open(json_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    csv_path = tables_dir / "module5_epqd_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "k", "Total Trainable Params", "PQC Core Params",
            "C_train (Nominal)", "Val MSE (Mean)", "Val MSE (Std)",
            "PSNR (Mean)", "PSNR (Std)", "SSIM (Mean)", "SSIM (Std)",
            "LPIPS (Mean)", "LPIPS (Std)", "D_seam (Mean)",
            "Von Neumann Entropy", "Bridge Delta L"
        ])
        for m_name in models_to_run:
            s = experiment_results["summary"].get(m_name, {})
            k_val = 1 if "k1" in m_name else (2 if "k2" in m_name else 4)
            pqc_p = 18 if k_val == 1 else (39 if k_val == 2 else 60)
            tot_p = 786 if k_val == 1 else (807 if k_val == 2 else 572)
            c_tr = c_train_dict["k1"] if k_val == 1 else (c_train_dict["k2"] if k_val == 2 else c_train_dict["k4_nq4"])
            writer.writerow([
                m_name, k_val, tot_p, pqc_p, c_tr,
                f"{s.get('val_mse_mean', 0.0):.5f}", f"{s.get('val_mse_std', 0.0):.5f}",
                f"{s.get('psnr_mean', 0.0):.2f}", f"{s.get('psnr_std', 0.0):.2f}",
                f"{s.get('ssim_mean', 0.0):.4f}", f"{s.get('ssim_std', 0.0):.4f}",
                f"{s.get('lpips_mean', 0.0):.4f}", f"{s.get('lpips_std', 0.0):.4f}",
                f"{s.get('d_seam_mean', 0.0):.5f}",
                f"{s.get('von_neumann_entropy_mean', 0.0):.4f}",
                f"{s.get('delta_loss_bridge_mean', 0.0):.4f}",
            ])

    print("\n================================================================================")
    print(f"MODULE 5 EXPERIMENT COMPLETE! Saved logs to {json_path} and CSV to {csv_path}")
    print("================================================================================")

    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 5 EPQD Experiment Runner")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/module5_epqd.yaml"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--models", nargs="+", type=str, default=None)
    parser.add_argument("--no-compute-matched", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dev = torch.device(args.device)
    cfg_p = Path(args.config)
    run_full_epqd_experiment(
        config_path=cfg_p,
        device=dev,
        seeds=args.seeds,
        models_to_run=args.models,
        run_compute_matched=not args.no_compute_matched,
        dry_run=args.dry_run,
    )
