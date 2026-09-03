"""
Latent-Patch Denoising Dataset and Preprocessing Pipeline for PHQD Module 2 Stage 2.

Provides:
- LatentPatchDenoisingDataset: PyTorch Dataset taking Module 1 latent patches [B, 9, 4, 4, 4],
  flattening each patch to 64-D, projecting to 6-D via frozen FixedPatchProjector with
  training-derived standard normalization, and generating deterministic Gaussian noise (std=0.2).
  Returns:
      input:  noisy projected 6-D vector
      target: clean projected 6-D vector
- prepare_latent_patch_denoising_datasets: Helper ensuring train/val/test splits share the exact
  same frozen projector whose normalization statistics are fitted strictly on training patches only.
- get_latent_patch_denoising_dataloaders: DataLoader builder for Stage 2 experiments.

Methodological Context:
- Intentionally Lossy Dimension Reduction: The 64-D -> 6-D projection maps a 64-dimensional
  latent patch space to a 6-dimensional subspace (compression ratio 10.67:1). This reduction
  is intentionally lossy and discards the 58-dimensional orthogonal complement (null space of P).
  Arbitrary 64-D patches cannot be exactly reconstructed.
- Pseudo-Inverse Back-Projection: Any back-projection from 6-D to 64-D is a pseudo-inverse
  reconstruction onto the fixed 6-D projection subspace, recovering vectors in the subspace faithfully
  while incurring expected non-zero reconstruction error for arbitrary 64-D vectors.
- Metric Priority for Stage 2 Experiment: 6-D projected-space denoising MSE serves as the primary
  evaluation metric. 64-D pseudo-inverse reconstruction MSE serves as a secondary/diagnostic metric.
"""

from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.projection import FixedPatchProjector


def load_raw_patches(patches_input: Union[torch.Tensor, str, Path]) -> torch.Tensor:
    """Load and format patch tensor from in-memory tensor or exported .pt file."""
    if isinstance(patches_input, (str, Path)):
        path = Path(patches_input)
        if not path.exists():
            raise FileNotFoundError(f"Latent patch file not found at: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "patches" in payload:
            patches = payload["patches"]
        elif isinstance(payload, torch.Tensor):
            patches = payload
        else:
            raise ValueError(f"Unrecognized payload structure in: {path}")
    elif isinstance(patches_input, torch.Tensor):
        patches = patches_input
    else:
        raise TypeError(f"Expected torch.Tensor, str, or Path, got {type(patches_input)}")
    return patches


def flatten_latent_patches(patches: torch.Tensor, patch_dim: int = 64) -> torch.Tensor:
    """Flatten [B, 9, 4, 4, 4] or [N, 4, 4, 4] patch tensors into [Total_patches, patch_dim]."""
    if patches.dim() == 5 and patches.shape[1:] == (9, 4, 4, 4):
        # [B, 9, 4, 4, 4] -> [B * 9, 64]
        b = patches.size(0)
        return patches.view(b * 9, patch_dim)
    elif patches.dim() == 4 and patches.shape[1:] == (4, 4, 4):
        # [N, 4, 4, 4] -> [N, 64]
        return patches.view(patches.size(0), patch_dim)
    elif patches.dim() == 2 and patches.size(1) == patch_dim:
        return patches
    elif patches.dim() == 1 and patches.size(0) == patch_dim:
        return patches.unsqueeze(0)
    else:
        # Generic reshape if total elements are a multiple of patch_dim
        if patches.numel() % patch_dim != 0:
            raise ValueError(
                f"Tensor elements {patches.numel()} not divisible by patch_dim {patch_dim}"
            )
        return patches.reshape(-1, patch_dim)


class LatentPatchDenoisingDataset(Dataset):
    """Dataset serving deterministic noisy 6-D vector inputs and clean 6-D vector targets.

    Pipeline:
    1. Input patches [B, 9, 4, 4, 4] flattened to 64-D [B*9, 64].
    2. Projected and standardized to 6-D clean vectors y in R^6 via frozen FixedPatchProjector.
    3. Deterministic zero-mean Gaussian noise epsilon ~ N(0, noise_std^2) added: x = y + epsilon.
    4. Yields (noisy_vector [6], clean_target [6]).
    """

    def __init__(
        self,
        patches: Union[torch.Tensor, str, Path],
        projector: FixedPatchProjector,
        noise_std: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.projector = projector

        # 1. Load and flatten patches to [Total_patches, 64]
        raw_patches = load_raw_patches(patches)
        patches_64 = flatten_latent_patches(raw_patches, patch_dim=projector.in_dim)
        patches_64 = patches_64.to(dtype=torch.float32, device=projector.projection_matrix.device)

        # 2. Project clean vectors to 6-D using frozen projector (eval mode, no grad)
        with torch.no_grad():
            self.clean = self.projector(patches_64).detach().cpu()  # [Total_patches, 6]

        # 3. Generate deterministic additive Gaussian noise
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.randn(self.clean.shape, generator=gen, dtype=torch.float32) * self.noise_std
        self.noisy = (self.clean + noise).detach().cpu()  # [Total_patches, 6]

    def __len__(self) -> int:
        return self.clean.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]

    @property
    def dim(self) -> int:
        return self.clean.size(1)


def prepare_latent_patch_denoising_datasets(
    train_patches: Union[torch.Tensor, str, Path],
    val_patches: Optional[Union[torch.Tensor, str, Path]] = None,
    test_patches: Optional[Union[torch.Tensor, str, Path]] = None,
    noise_std: float = 0.2,
    projector_seed: int = 42,
    dataset_seed: int = 42,
    projector: Optional[FixedPatchProjector] = None,
) -> Tuple[LatentPatchDenoisingDataset, ...]:
    """Build train, val, and test datasets with strictly training-data-only normalization statistics.

    Args:
        train_patches: Training patches [B_train, 9, 4, 4, 4] or .pt path.
        val_patches: Optional validation patches or .pt path.
        test_patches: Optional test patches or .pt path.
        noise_std: Gaussian noise standard deviation. Defaults to 0.2.
        projector_seed: Seed for 64->6 semi-orthogonal projection matrix. Defaults to 42.
        dataset_seed: Seed for training noise generation. Defaults to 42.
        projector: Optional pre-fitted FixedPatchProjector. If None, instantiates a new
                   projector and fits normalization statistics on train_patches only.

    Returns:
        Tuple containing (train_dataset, [val_dataset], [test_dataset], projector).
    """
    raw_train = load_raw_patches(train_patches)
    train_64 = flatten_latent_patches(raw_train, patch_dim=64)

    if projector is None:
        # Instantiate and fit normalization strictly on training patches
        projector = FixedPatchProjector(
            in_dim=64,
            out_dim=6,
            seed=projector_seed,
            norm_type="standard",
        )
        projector.fit(train_64)

    train_dataset = LatentPatchDenoisingDataset(
        patches=train_64,
        projector=projector,
        noise_std=noise_std,
        seed=dataset_seed,
    )

    results = [train_dataset]

    if val_patches is not None:
        val_dataset = LatentPatchDenoisingDataset(
            patches=val_patches,
            projector=projector,
            noise_std=noise_std,
            seed=dataset_seed + 1,  # Independent deterministic seed for validation noise
        )
        results.append(val_dataset)

    if test_patches is not None:
        test_dataset = LatentPatchDenoisingDataset(
            patches=test_patches,
            projector=projector,
            noise_std=noise_std,
            seed=dataset_seed + 2,  # Independent deterministic seed for test noise
        )
        results.append(test_dataset)

    results.append(projector)
    return tuple(results)


def get_latent_patch_denoising_dataloaders(
    train_patches: Union[torch.Tensor, str, Path],
    val_patches: Optional[Union[torch.Tensor, str, Path]] = None,
    test_patches: Optional[Union[torch.Tensor, str, Path]] = None,
    batch_size: int = 32,
    noise_std: float = 0.2,
    projector_seed: int = 42,
    dataset_seed: int = 42,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, ...]:
    """Construct PyTorch DataLoaders for Module 2 Stage 2."""
    datasets_and_proj = prepare_latent_patch_denoising_datasets(
        train_patches=train_patches,
        val_patches=val_patches,
        test_patches=test_patches,
        noise_std=noise_std,
        projector_seed=projector_seed,
        dataset_seed=dataset_seed,
    )

    loaders = []
    # Train loader
    loaders.append(
        DataLoader(
            datasets_and_proj[0],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )
    )

    # Val / Test loaders
    for d in datasets_and_proj[1:-1]:
        loaders.append(
            DataLoader(
                d,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        )

    # Append the frozen projector as final element for downstream reference
    loaders.append(datasets_and_proj[-1])
    return tuple(loaders)
