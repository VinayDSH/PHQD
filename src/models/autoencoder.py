"""
Convolutional Autoencoder baseline for PHQD Module 1.

Maps single-channel 28x28 images to a continuous 4x8x8 spatial latent grid
and reconstructs back to 1x28x28.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Baseline Convolutional Autoencoder for 28x28 grayscale images.

    Architecture:
        Encoder:
            - Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  -> [B, 32, 14, 14]
            - ReLU
            - Conv2d(32, 64, kernel_size=3, stride=1, padding=1)           -> [B, 64, 14, 14]
            - ReLU
            - Conv2d(64, latent_channels, kernel_size=4, stride=2, padding=2) -> [B, 4, 8, 8]
            (Unactivated continuous latent representation)

        Decoder:
            - ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=2) -> [B, 64, 14, 14]
            - ReLU
            - Conv2d(64, 32, kernel_size=3, stride=1, padding=1)           -> [B, 32, 14, 14]
            - ReLU
            - ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)      -> [B, 1, 28, 28]
            - Sigmoid
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder: [B, 1, 28, 28] -> [B, 4, 8, 8]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, kernel_size=4, stride=2, padding=2),
        )

        # Decoder: [B, 4, 8, 8] -> [B, 1, 28, 28]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image into continuous 4x8x8 spatial latent tensor.

        Args:
            x: Input tensor of shape [B, 1, 28, 28].

        Returns:
            Latent tensor of shape [B, 4, 8, 8].
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode continuous 4x8x8 spatial latent tensor into reconstructed image.

        Args:
            z: Latent tensor of shape [B, 4, 8, 8].

        Returns:
            Reconstructed image tensor of shape [B, 1, 28, 28] with values in [0, 1].
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass: encode followed by decode.

        Args:
            x: Input tensor of shape [B, 1, 28, 28].

        Returns:
            Reconstructed image tensor of shape [B, 1, 28, 28].
        """
        z = self.encode(x)
        return self.decode(z)
