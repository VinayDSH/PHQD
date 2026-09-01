"""
Dataset and DataLoader layer for PHQD Module 1.

Provides deterministic train/validation splitting and DataLoader construction
for the Fashion-MNIST dataset.
"""

from typing import Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST


def get_fashion_mnist_transforms() -> transforms.Compose:
    """Return standard torchvision transforms mapping images to float32 tensors in [0, 1].

    Output shape: [1, 28, 28], dtype: torch.float32, value range: [0.0, 1.0].
    """
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_fashion_mnist_dataloaders(
    root: Union[str, Path] = "data/raw",
    batch_size: int = 128,
    val_size: Union[int, float] = 10000,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct deterministic train, validation, and test DataLoaders for Fashion-MNIST.

    Args:
        root: Directory path where Fashion-MNIST dataset is stored or downloaded.
              Defaults to "data/raw".
        batch_size: Batch size for loaders. Defaults to 128.
        val_size: Validation set size from official 60,000-image training split.
                  If int, specifies exact sample count (e.g., 10000).
                  If float in (0, 1), specifies fraction of the 60,000 images.
                  Defaults to 10000.
        seed: Random seed for deterministic train/validation splitting. Defaults to 42.
        num_workers: DataLoader worker count. Defaults to 0.
        pin_memory: Whether to pin memory in DataLoaders. Defaults to False.
        download: Whether to download dataset if not present. Defaults to True.

    Returns:
        Tuple containing (train_loader, val_loader, test_loader).
    """
    root_str = str(root)
    transform = get_fashion_mnist_transforms()

    # Load official 60,000-image training dataset (executed only on function call, not import)
    full_train_dataset = FashionMNIST(
        root=root_str,
        train=True,
        transform=transform,
        download=download,
    )

    # Load official 10,000-image test dataset
    test_dataset = FashionMNIST(
        root=root_str,
        train=False,
        transform=transform,
        download=download,
    )

    total_train_samples = len(full_train_dataset)

    # Compute train/validation split sizes
    if isinstance(val_size, float):
        if not (0.0 < val_size < 1.0):
            raise ValueError(f"val_size as a float must be in (0, 1), got {val_size}")
        val_count = int(val_size * total_train_samples)
    elif isinstance(val_size, int):
        if not (0 < val_size < total_train_samples):
            raise ValueError(
                f"val_size as an int must be between 1 and {total_train_samples - 1}, got {val_size}"
            )
        val_count = val_size
    else:
        raise TypeError(f"val_size must be int or float, got {type(val_size)}")

    train_count = total_train_samples - val_count

    # Deterministic split using a dedicated PyTorch generator seeded with `seed`
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_count, val_count],
        generator=split_generator,
    )

    # Seeded shuffle generator for train DataLoader reproducibility
    train_shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_shuffle_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
