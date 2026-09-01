"""
Unit tests for Module 1 Fashion-MNIST dataset and DataLoader layer.

Verifies:
- Image tensor shape is [1, 28, 28]
- Image dtype is float32
- Image values are within [0, 1]
- Train/validation/test loaders can produce batches
- Train and validation sample counts sum to 60,000
- Test sample count is 10,000
- Train/validation split is deterministic for seed 42
"""

import sys
from pathlib import Path
import unittest
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import get_fashion_mnist_dataloaders


class TestFashionMNISTDataLoaders(unittest.TestCase):
    """Test suite for Fashion-MNIST data loaders and preprocessing."""

    @classmethod
    def setUpClass(cls):
        # Instantiate dataloaders once for suite execution
        cls.batch_size = 128
        cls.seed = 42
        cls.train_loader, cls.val_loader, cls.test_loader = get_fashion_mnist_dataloaders(
            root="data/raw",
            batch_size=cls.batch_size,
            val_size=10000,
            seed=cls.seed,
            download=False,
        )

    def test_image_tensor_shape(self):
        """Verify image tensor shape is [1, 28, 28] for individual images."""
        images, labels = next(iter(self.train_loader))
        # Batch shape is [batch_size, 1, 28, 28]
        self.assertEqual(images.shape[1:], torch.Size([1, 28, 28]))
        # Single image shape is [1, 28, 28]
        self.assertEqual(images[0].shape, torch.Size([1, 28, 28]))

    def test_image_dtype(self):
        """Verify image tensor dtype is float32."""
        images, labels = next(iter(self.train_loader))
        self.assertEqual(images.dtype, torch.float32)

    def test_image_values_range(self):
        """Verify image pixel values fall strictly within [0.0, 1.0]."""
        images, labels = next(iter(self.train_loader))
        min_val = images.min().item()
        max_val = images.max().item()
        self.assertGreaterEqual(min_val, 0.0, f"Minimum value {min_val} < 0.0")
        self.assertLessEqual(max_val, 1.0, f"Maximum value {max_val} > 1.0")

    def test_loaders_produce_batches(self):
        """Verify train, validation, and test loaders can produce batches with valid shapes."""
        for loader_name, loader in [
            ("train", self.train_loader),
            ("validation", self.val_loader),
            ("test", self.test_loader),
        ]:
            images, labels = next(iter(loader))
            self.assertEqual(
                images.shape[0],
                self.batch_size,
                f"{loader_name} loader batch size mismatch",
            )
            self.assertEqual(
                images.shape[1:],
                torch.Size([1, 28, 28]),
                f"{loader_name} loader image dimension mismatch",
            )
            self.assertEqual(
                labels.shape[0],
                self.batch_size,
                f"{loader_name} loader label batch size mismatch",
            )

    def test_sample_counts(self):
        """Verify train + validation sum to 60,000 and test count is 10,000."""
        train_count = len(self.train_loader.dataset)
        val_count = len(self.val_loader.dataset)
        test_count = len(self.test_loader.dataset)

        self.assertEqual(
            train_count + val_count,
            60000,
            f"Train ({train_count}) and val ({val_count}) do not sum to 60,000",
        )
        self.assertEqual(
            test_count,
            10000,
            f"Test sample count {test_count} != 10,000",
        )

    def test_deterministic_split_seed_42(self):
        """Verify train/validation split is completely deterministic for seed 42."""
        train_1, val_1, _ = get_fashion_mnist_dataloaders(
            root="data/raw",
            batch_size=self.batch_size,
            val_size=10000,
            seed=42,
            download=False,
        )
        train_2, val_2, _ = get_fashion_mnist_dataloaders(
            root="data/raw",
            batch_size=self.batch_size,
            val_size=10000,
            seed=42,
            download=False,
        )

        # Indices of the subsets must match identically
        self.assertEqual(train_1.dataset.indices, train_2.dataset.indices)
        self.assertEqual(val_1.dataset.indices, val_2.dataset.indices)

        # Verify a different seed produces different indices
        train_diff, val_diff, _ = get_fashion_mnist_dataloaders(
            root="data/raw",
            batch_size=self.batch_size,
            val_size=10000,
            seed=999,
            download=False,
        )
        self.assertNotEqual(train_1.dataset.indices, train_diff.dataset.indices)
        self.assertNotEqual(val_1.dataset.indices, val_diff.dataset.indices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
