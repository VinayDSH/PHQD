"""
Toy Gaussian Denoising Dataset for PHQD Module 2.

Generates clean structured vectors with smooth harmonic coordinates and adds
zero-mean Gaussian noise with configurable standard deviation.
"""

from typing import Tuple, Optional
import math
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def generate_toy_denoising_data(
    num_samples: int = 1000,
    dim: int = 6,
    noise_std: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic clean and noisy vector pairs for toy denoising.

    Clean vectors are sampled from smooth harmonic curves:
        y_j = A * sin(2*pi*f*j/dim + phi)
    where A in [0.3, 0.7], f in {1, 2}, phi in [0, 2*pi).
    Clean values are bounded in [-0.7, 0.7] subset of [-1, 1].

    Noisy vectors:
        x_j = y_j + epsilon_j, where epsilon_j ~ N(0, noise_std^2).

    Args:
        num_samples: Number of vector pairs to generate. Defaults to 1000.
        dim: Dimensionality of each vector (equals qubit count). Defaults to 6.
        noise_std: Standard deviation of additive zero-mean Gaussian noise. Defaults to 0.2.
        seed: Random seed for reproducible generation. Defaults to 42.

    Returns:
        Tuple of (noisy_vectors, clean_targets), each of shape [num_samples, dim].
    """
    generator = torch.Generator().manual_seed(seed)

    # Random parameters for each sample
    # Amplitude A in [0.3, 0.7]
    amplitudes = 0.3 + 0.4 * torch.rand(num_samples, 1, generator=generator)
    # Frequency f in {1, 2}
    freqs = torch.randint(1, 3, (num_samples, 1), generator=generator).float()
    # Phase phi in [0, 2*pi)
    phases = 2.0 * math.pi * torch.rand(num_samples, 1, generator=generator)

    # Coordinate indices j = 0, 1, ..., dim-1
    j_indices = torch.arange(dim, dtype=torch.float32).unsqueeze(0)  # [1, dim]

    # Compute clean vectors: [num_samples, dim]
    angles = (2.0 * math.pi * freqs * j_indices / dim) + phases
    clean_vectors = amplitudes * torch.sin(angles)

    # Additive zero-mean Gaussian noise
    noise = torch.randn(num_samples, dim, generator=generator) * noise_std
    noisy_vectors = clean_vectors + noise

    return noisy_vectors.to(torch.float32), clean_vectors.to(torch.float32)


class ToyGaussianDenoisingDataset(Dataset):
    """PyTorch Dataset wrapping synthetic noisy-clean vector pairs."""

    def __init__(
        self,
        num_samples: int = 1000,
        dim: int = 6,
        noise_std: float = 0.2,
        seed: int = 42,
    ):
        self.noisy, self.clean = generate_toy_denoising_data(
            num_samples=num_samples,
            dim=dim,
            noise_std=noise_std,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.noisy.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


def get_toy_denoising_dataloaders(
    num_train: int = 800,
    num_val: int = 200,
    dim: int = 6,
    noise_std: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Construct deterministic train and validation DataLoaders for the toy denoising task.

    Args:
        num_train: Number of training samples. Defaults to 800.
        num_val: Number of validation samples. Defaults to 200.
        dim: Vector dimension. Defaults to 6.
        noise_std: Gaussian noise standard deviation. Defaults to 0.2.
        batch_size: Mini-batch size. Defaults to 32.
        seed: Random seed. Defaults to 42.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = ToyGaussianDenoisingDataset(
        num_samples=num_train,
        dim=dim,
        noise_std=noise_std,
        seed=seed,
    )
    # Use distinct seed for validation set to test generalization
    val_dataset = ToyGaussianDenoisingDataset(
        num_samples=num_val,
        dim=dim,
        noise_std=noise_std,
        seed=seed + 1000,
    )

    train_shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_shuffle_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
