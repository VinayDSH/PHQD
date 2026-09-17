"""
Module 5: EPQD Statistical Analysis, Compute Accounting, and Gate 2 Verification.

Implements:
1. calculate_epqd_c_train: Locked analytical compute accounting formulations.
2. compute_efficiency_index: Compute Efficiency Index eta.
3. compute_paired_t_test: Two-sided paired Student's t-test matching exact formula:
   t = d_bar / (s_d / sqrt(3)).
4. fit_power_law: Empirical scaling fit MSE(k) = MSE_inf + A / k^gamma.
5. evaluate_gate2_criteria: Strict Gate 2 verification against locked criteria.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def calculate_epqd_c_train(
    n_epochs: int = 15,
    n_batch: int = 32,
    kappa: float = 1.0,
) -> Dict[str, int]:
    """Calculate exact analytical training compute accounting C_train values.
    
    Formula:
        C_sim(N, L) = kappa * L * 2^N
        Total steps = n_epochs * n_batch = 480 (at 15 epochs, 32 batches)
        C_train = steps * [Quantum Cost/sample + C_cls]
    """
    steps = n_epochs * n_batch  # 480

    # Simulation gate-equivalents (L=3)
    c_sim_6 = int(kappa * 3 * (2 ** 6))        # 192
    c_sim_12 = int(kappa * 3 * (2 ** 12))      # 12,288
    c_sim_16 = int(kappa * 3 * (2 ** 16))      # 196,608
    c_sim_24 = int(kappa * 3 * (2 ** 24))      # 50,331,648

    # Classical adapter FLOPs per batch
    c_cls_64_6 = 69120   # for nq=6
    c_cls_64_4 = 46080   # for nq=4

    # 1. k=1: 9 independent patches, P=18 -> (2P+1) = 37 evaluations
    # Quantum Cost = 9 * 37 * 192 = 63,936
    cost_sample_k1 = 9 * (2 * 18 + 1) * c_sim_6
    c_train_k1 = steps * (cost_sample_k1 + c_cls_64_6)  # 480 * (63936 + 69120) = 63,866,880

    # 2. k=2: 4 pairs (P=39 -> 79 evals) + 1 single (P=18 -> 37 evals)
    # Quantum Cost = 4 * (79 * 12288) + 1 * (37 * 192) = 3,883,008 + 7,104 = 3,890,112
    cost_sample_k2 = 4 * ((2 * 39 + 1) * c_sim_12) + 1 * ((2 * 18 + 1) * c_sim_6)
    c_train_k2 = steps * (cost_sample_k2 + c_cls_64_6)  # 480 * (3890112 + 69120) = 1,900,431,360

    # 3. k=4 (nq=4, Empirical): 4 quads (P=60 -> 121 evals)
    # Quantum Cost = 4 * (121 * 196608) = 95,158,272
    cost_sample_k4_nq4 = 4 * ((2 * 60 + 1) * c_sim_16)
    c_train_k4_nq4 = steps * (cost_sample_k4_nq4 + c_cls_64_4)  # 480 * (95158272 + 46080) = 45,716,697,600

    # 4. k=4 (nq=6, Theoretical): 4 quads (P=84 -> 169 evals)
    # Quantum Cost = 4 * (169 * 50331648) = 34,024,194,048
    cost_sample_k4_nq6 = 4 * ((2 * 84 + 1) * c_sim_24)
    c_train_k4_nq6 = steps * (cost_sample_k4_nq6 + c_cls_64_6)  # 16,339,472,332,800

    return {
        "k1": c_train_k1,
        "k2": c_train_k2,
        "k4_nq4": c_train_k4_nq4,
        "k4_nq6_theoretical": c_train_k4_nq6,
        "cost_sample_k1": cost_sample_k1,
        "cost_sample_k2": cost_sample_k2,
        "cost_sample_k4_nq4": cost_sample_k4_nq4,
        "cost_sample_k4_nq6": cost_sample_k4_nq6,
    }


def compute_efficiency_index(delta_psnr: float, c_k1: float, c_k2: float) -> float:
    """Compute Efficiency Index eta.
    
    Clarification from Specification:
    Compute Efficiency Index eta is evaluated only under the nominal 15-epoch,
    batch-32 budgets using the analytically defined Ctrain values in Section 3.
    The compute-matched experiment is reported separately and is not used in
    the denominator of eta.
    
    Formula:
        eta = Delta_PSNR / log10(C_train(k=2) / C_train(k=1))
    """
    ratio = c_k2 / c_k1
    log_ratio = math.log10(ratio)
    return float(delta_psnr / log_ratio)


def compute_paired_t_test(
    psnr_k1: List[float],
    psnr_k2: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform exact paired two-sided Student's t-test across locked seed set.
    
    Formulation:
        d_i = PSNR_k=2(seed_i) - PSNR_k=1(seed_i)
        d_bar = mean(d_i)
        s_d = sample standard deviation of d_i (ddof=1)
        t = d_bar / (s_d / sqrt(3))
    """
    if len(psnr_k1) != len(psnr_k2):
        raise ValueError(f"Seed counts must match: {len(psnr_k1)} vs {len(psnr_k2)}")
    n = len(psnr_k1)
    if n < 2:
        raise ValueError(f"Need at least 2 seeds for t-test, got {n}")

    d = np.array(psnr_k2) - np.array(psnr_k1)
    d_bar = float(np.mean(d))
    s_d = float(np.std(d, ddof=1))

    if s_d == 0.0:
        t_stat = float("inf") if d_bar > 0 else (float("-inf") if d_bar < 0 else 0.0)
        p_val = 0.0 if d_bar != 0 else 1.0
    else:
        t_stat = float(d_bar / (s_d / math.sqrt(n)))
        # Two-sided p-value from Student's t distribution with df = n - 1
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))

    return {
        "d_differences": d.tolist(),
        "d_bar": d_bar,
        "s_d": s_d,
        "t_statistic": t_stat,
        "degrees_of_freedom": n - 1,
        "p_value": p_val,
        "alpha": alpha,
        "is_significant": bool(p_val < alpha),
        "n_seeds": n,
    }


def _power_law_func(k, mse_inf, a, gamma):
    return mse_inf + a / (k ** gamma)


def fit_power_law(
    k_values: np.ndarray,
    mse_values: np.ndarray,
) -> Dict[str, Any]:
    """Fit empirical power-law equation: MSE(k) = MSE_inf + A / k^gamma.
    
    with:
      A > 0: Positive scaling coefficient
      gamma > 0: Indicates improving empirical scaling with increasing grouping scale k
      gamma <= 0: Indicates no positive scaling benefit
    """
    k_arr = np.asarray(k_values, dtype=float)
    mse_arr = np.asarray(mse_values, dtype=float)

    try:
        # Initial guess: mse_inf = min(mse)/2, A = range(mse), gamma = 0.5
        p0 = [max(0.0, float(np.min(mse_arr)) * 0.5), float(np.ptp(mse_arr) + 1e-4), 0.5]
        bounds = ([0.0, 1e-6, -5.0], [np.inf, np.inf, 5.0])
        popt, _ = curve_fit(_power_law_func, k_arr, mse_arr, p0=p0, bounds=bounds, maxfev=5000)
        mse_inf, a, gamma = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        # Fallback estimation if curve_fit fails on sparse points
        mse_inf = float(np.min(mse_arr))
        gamma = float((np.log(mse_arr[0] - mse_inf + 1e-6) - np.log(mse_arr[-1] - mse_inf + 1e-6)) / (np.log(k_arr[-1]) - np.log(k_arr[0])))
        a = float((mse_arr[0] - mse_inf) * (k_arr[0] ** gamma))

    is_improving = bool(gamma > 0 and a > 0)
    return {
        "mse_inf": mse_inf,
        "A": a,
        "gamma": gamma,
        "is_improving": is_improving,
    }


def evaluate_gate2_criteria(
    psnr_k1_seeds: List[float],
    psnr_k2_seeds: List[float],
    lpips_k1: float,
    lpips_k2: float,
    fid_k1: float,
    fid_k2: float,
    d_seam_k1: float,
    d_seam_k2: float,
    entropy_k2: float,
    delta_loss_bridge: float,
    c_train_k1: Optional[float] = None,
    c_train_k2: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate all five locked Gate 2 acceptance criteria."""
    if c_train_k1 is None or c_train_k2 is None:
        c_dict = calculate_epqd_c_train()
        c_train_k1 = c_dict["k1"]
        c_train_k2 = c_dict["k2"]

    # Criterion 1: Reconstruction Gain & Statistical Significance
    ttest_res = compute_paired_t_test(psnr_k1_seeds, psnr_k2_seeds, alpha=0.05)
    mean_k1 = float(np.mean(psnr_k1_seeds))
    mean_k2 = float(np.mean(psnr_k2_seeds))
    delta_psnr = mean_k2 - mean_k1
    crit1_pass = bool(delta_psnr >= 0.30 and ttest_res["is_significant"])

    # Criterion 2: Seam Harmonization (D_seam decrease >= 20%)
    if d_seam_k1 > 0:
        seam_reduction_pct = float((d_seam_k1 - d_seam_k2) / d_seam_k1 * 100.0)
    else:
        seam_reduction_pct = 0.0
    crit2_pass = bool(seam_reduction_pct >= 20.0)

    # Criterion 3: Active Quantum Entanglement
    crit3_pass = bool(entropy_k2 > 0.10 and delta_loss_bridge > 0.01)

    # Criterion 4: Compute Efficiency Index eta > 0.15
    eta = compute_efficiency_index(delta_psnr, c_train_k1, c_train_k2)
    crit4_pass = bool(eta > 0.15)

    # Criterion 5: Metric Consistency (No degradation in LPIPS or FID)
    # LPIPS lower is better: lpips_k2 <= lpips_k1 + 0.01 tolerance
    # FID lower is better: fid_k2 <= fid_k1 + 1.0 tolerance
    crit5_pass = bool(lpips_k2 <= lpips_k1 + 0.01 and fid_k2 <= fid_k1 + 1.0)

    gate2_overall_pass = bool(crit1_pass and crit2_pass and crit3_pass and crit4_pass and crit5_pass)

    return {
        "gate2_passed": gate2_overall_pass,
        "criterion_1_psnr_gain": {
            "mean_psnr_k1": mean_k1,
            "mean_psnr_k2": mean_k2,
            "delta_psnr": delta_psnr,
            "t_statistic": ttest_res["t_statistic"],
            "p_value": ttest_res["p_value"],
            "passed": crit1_pass,
        },
        "criterion_2_seam_harmonization": {
            "d_seam_k1": d_seam_k1,
            "d_seam_k2": d_seam_k2,
            "seam_reduction_pct": seam_reduction_pct,
            "passed": crit2_pass,
        },
        "criterion_3_active_entanglement": {
            "von_neumann_entropy": entropy_k2,
            "delta_loss_bridge": delta_loss_bridge,
            "passed": crit3_pass,
        },
        "criterion_4_compute_efficiency": {
            "eta": eta,
            "c_train_k1": c_train_k1,
            "c_train_k2": c_train_k2,
            "passed": crit4_pass,
        },
        "criterion_5_metric_consistency": {
            "lpips_k1": lpips_k1,
            "lpips_k2": lpips_k2,
            "fid_k1": fid_k1,
            "fid_k2": fid_k2,
            "passed": crit5_pass,
        },
    }
