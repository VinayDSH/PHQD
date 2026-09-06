"""
Computational efficiency and complexity metrics for PHQD Module 4.

Provides:
- count_parameters: Counts trainable and total parameters in an nn.Module.
- parameter_summary: Detailed parameter count, byte storage, and memory footprint in MB.
- measure_inference_runtime: High-resolution wall-clock inference latency measurement with
  warmup runs, CUDA synchronization, and throughput calculation (samples/second).
"""

from typing import Dict, Any, Optional, Callable
import time
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a PyTorch module.

    Args:
        model: nn.Module to inspect.
        trainable_only: If True, counts only parameters with requires_grad=True.
                        If False, counts all parameters.

    Returns:
        Integer count of parameters.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected model to be an nn.Module, got {type(model).__name__}")

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_summary(model: nn.Module) -> Dict[str, Any]:
    """Generate detailed parameter count and memory storage summary for a model.

    Args:
        model: nn.Module to inspect.

    Returns:
        Dictionary containing:
        - 'trainable_parameters': Number of parameters with requires_grad=True.
        - 'non_trainable_parameters': Number of frozen parameters with requires_grad=False.
        - 'total_parameters': Total parameter count.
        - 'parameter_bytes': Estimated storage in bytes based on element sizes.
        - 'model_size_mb': Estimated parameter memory footprint in megabytes (MB).
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected model to be an nn.Module, got {type(model).__name__}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_bytes / (1024.0 * 1024.0)

    return {
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "total_parameters": total,
        "parameter_bytes": param_bytes,
        "model_size_mb": model_size_mb,
    }


def measure_inference_runtime(
    fn_or_model: Callable,
    *args,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Dict[str, float]:
    """Measure high-resolution wall-clock inference latency and throughput.

    Deterministic & Benchmarking Methodology:
    - Runs `num_warmup` executions prior to measurement to trigger kernel compilation,
      allocate GPU memory, and warm up caches.
    - Synchronizes CUDA before and after each timed execution (`torch.cuda.synchronize()`)
      to ensure asynchronous GPU kernel queues do not skew wall-clock measurements.
    - Uses `time.perf_counter()` for highest available resolution timing.
    - Optionally calculates throughput (`samples_per_second`) when `batch_size` is specified.

    Args:
        fn_or_model: Callable function or nn.Module to benchmark.
        *args: Positional arguments passed to fn_or_model.
        num_warmup: Number of untimed warmup iterations (default 3).
        num_runs: Number of timed benchmark iterations (default 10).
        device: PyTorch device (used to determine whether CUDA synchronization is required).
        batch_size: Batch size of input to compute samples/sec throughput.
        **kwargs: Keyword arguments passed to fn_or_model.

    Returns:
        Dictionary containing:
        - 'mean_time_seconds': Mean execution time per run.
        - 'std_time_seconds': Standard deviation of execution time.
        - 'min_time_seconds': Minimum execution time across runs.
        - 'max_time_seconds': Maximum execution time across runs.
        - 'total_time_seconds': Cumulative execution time across all timed runs.
        - 'num_runs': Number of timed runs completed.
        - 'samples_per_second': Throughput in samples per second (if batch_size provided).
    """
    if num_warmup < 0:
        raise ValueError(f"num_warmup must be non-negative, got {num_warmup}")
    if num_runs <= 0:
        raise ValueError(f"num_runs must be positive, got {num_runs}")

    # Determine if CUDA synchronization is needed
    is_cuda = False
    if device is not None:
        is_cuda = torch.device(device).type == "cuda"
    elif isinstance(fn_or_model, nn.Module):
        try:
            param = next(fn_or_model.parameters())
            is_cuda = param.is_cuda
        except StopIteration:
            is_cuda = torch.cuda.is_available()
    elif len(args) > 0 and isinstance(args[0], torch.Tensor):
        is_cuda = args[0].is_cuda
    else:
        is_cuda = torch.cuda.is_available()

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            if is_cuda:
                torch.cuda.synchronize()
            _ = fn_or_model(*args, **kwargs)
            if is_cuda:
                torch.cuda.synchronize()

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = fn_or_model(*args, **kwargs)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            timings.append(t1 - t0)

    timings_tensor = torch.tensor(timings, dtype=torch.float64)
    mean_time = float(timings_tensor.mean().item())
    std_time = float(timings_tensor.std().item()) if num_runs > 1 else 0.0
    min_time = float(timings_tensor.min().item())
    max_time = float(timings_tensor.max().item())
    total_time = float(timings_tensor.sum().item())

    results: Dict[str, float] = {
        "mean_time_seconds": mean_time,
        "std_time_seconds": std_time,
        "min_time_seconds": min_time,
        "max_time_seconds": max_time,
        "total_time_seconds": total_time,
        "num_runs": float(num_runs),
    }

    if batch_size is not None and batch_size > 0:
        total_samples = batch_size * num_runs
        samples_per_second = total_samples / total_time if total_time > 0 else 0.0
        results["samples_per_second"] = samples_per_second

    return results
