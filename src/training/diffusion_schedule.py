"""
Forward Diffusion Noise Schedule over Latent Patches for PHQD Module 3 Milestone 1.

Provides:
- LinearDiffusionSchedule: Parameter-free PyTorch nn.Module precomputing the linear beta
  schedule and forward noising coefficients over discrete timesteps t in {0, ..., T-1}.
- linear_beta_schedule: Functional helper generating linear beta progressions.
- DiffusionSchedule: Alias for LinearDiffusionSchedule.

Key Mathematical Properties:
1. Linear Beta Schedule:
   beta_t = beta_start + (t / (T - 1)) * (beta_end - beta_start)
   for t in {0, ..., T-1}.
2. Alpha Factors:
   alpha_t = 1 - beta_t
   alpha_bar_t = prod_{s=0}^t alpha_s (cumulative product)
3. Forward Diffusion Equation:
   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
   where epsilon ~ N(0, I).
4. Signal-to-Noise Ratio (SNR):
   SNR(t) = alpha_bar_t / (1 - alpha_bar_t)

Implementation Details:
- Non-Trainable Persistent Buffers: Exactly five core buffers are registered:
  'betas', 'alphas', 'alphas_cumprod', 'sqrt_alphas_cumprod', and 'sqrt_one_minus_alphas_cumprod'.
- Latent Patch Broadcasting: Supports batched 5D latent patch tensors [B, N, C, P_h, P_w],
  3D flattened/projected patches [B, N, D], 4D latent grids [B, C, H, W], 2D latents [B, D],
  and unbatched patch representations with scalar timesteps.
- Deterministic Seeding: Independent RNG via torch.Generator or seed parameter with strict
  rejection of simultaneous seed + generator, and cross-device generator compatibility checks.
- Numerical Integrity: Rigorous finite-value checks (NaN/Inf) and timestep range validation.
"""

from typing import Tuple, Dict, Any, Optional, Union
import math
import torch
import torch.nn as nn


def linear_beta_schedule(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a 1-D linear beta schedule tensor.

    Args:
        timesteps: Total number of diffusion steps T >= 1. Defaults to 1000.
        beta_start: Minimum variance schedule value at t=0. Defaults to 1e-4.
        beta_end: Maximum variance schedule value at t=T-1. Defaults to 0.02.
        dtype: PyTorch floating-point dtype. Defaults to torch.float32.

    Returns:
        1-D torch.Tensor of shape (timesteps,) containing linearly spaced beta values.

    Raises:
        TypeError: If timesteps is not an integer.
        ValueError: If timesteps < 1, or beta values are out of bounds (0 < beta_start < beta_end < 1).
    """
    if not isinstance(timesteps, int):
        raise TypeError(f"timesteps must be an integer, got {type(timesteps).__name__}")
    if timesteps < 1:
        raise ValueError(f"timesteps must be >= 1, got {timesteps}")
    if not (0.0 < beta_start < 1.0):
        raise ValueError(f"beta_start must be in (0, 1), got {beta_start}")
    if not (0.0 < beta_end < 1.0):
        raise ValueError(f"beta_end must be in (0, 1), got {beta_end}")

    if timesteps == 1:
        return torch.tensor([beta_start], dtype=dtype)

    if beta_start >= beta_end:
        raise ValueError(f"beta_start ({beta_start}) must be strictly less than beta_end ({beta_end})")

    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)


class LinearDiffusionSchedule(nn.Module):
    """Classical forward diffusion schedule with linear variance scheduling over latent patches.

    Precomputes and registers the exact five core diffusion buffers:
      - 'betas': beta_t
      - 'alphas': alpha_t = 1 - beta_t
      - 'alphas_cumprod': alpha_bar_t = prod_{s=0}^t alpha_s
      - 'sqrt_alphas_cumprod': sqrt(alpha_bar_t)
      - 'sqrt_one_minus_alphas_cumprod': sqrt(1 - alpha_bar_t)

    Applies the closed-form forward diffusion equation:
      x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize linear diffusion schedule buffers.

        Args:
            timesteps: Total diffusion timesteps T >= 1. Defaults to 1000.
            beta_start: Beta value at t=0. Defaults to 1e-4.
            beta_end: Beta value at t=T-1. Defaults to 0.02.
            dtype: Tensor floating-point precision. Defaults to torch.float32.
        """
        super().__init__()

        betas = linear_beta_schedule(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dtype=dtype,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Register exactly the five core buffers
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alphas", alphas, persistent=True)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod, persistent=True)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod, persistent=True)

        self._timesteps = timesteps
        self._beta_start = float(beta_start)
        self._beta_end = float(beta_end)

    @property
    def timesteps(self) -> int:
        """Total number of diffusion steps T."""
        return self._timesteps

    @property
    def beta_start(self) -> float:
        """Initial beta variance schedule value at t=0."""
        return self._beta_start

    @property
    def beta_end(self) -> float:
        """Final beta variance schedule value at t=T-1."""
        return self._beta_end

    @property
    def snr(self) -> torch.Tensor:
        """Explicit Signal-to-Noise Ratio for all timesteps: SNR(t) = alpha_bar_t / (1 - alpha_bar_t)."""
        return self.alphas_cumprod / (1.0 - self.alphas_cumprod)

    def get_snr(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Retrieve the SNR for specific timestep(s) t.

        Args:
            t: Integer timestep or 1-D tensor of timesteps.

        Returns:
            torch.Tensor of SNR values matching the shape of t.
        """
        t_tensor = self._validate_and_format_timesteps(t, device=self.alphas_cumprod.device)
        return self.snr[t_tensor]

    def _validate_and_format_timesteps(
        self,
        t: Union[int, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Validate timestep ranges and format as long tensor on target device."""
        if isinstance(t, int):
            if t < 0 or t >= self._timesteps:
                raise ValueError(
                    f"Timestep {t} is out of bounds. Must be in [0, {self._timesteps - 1}]."
                )
            return torch.tensor([t], dtype=torch.long, device=device)

        if isinstance(t, torch.Tensor):
            if not torch.is_floating_point(t) and not torch.is_complex(t):
                # Integer tensor
                pass
            else:
                raise TypeError(f"Timestep tensor must have integer dtype, got {t.dtype}")

            if (t < 0).any() or (t >= self._timesteps).any():
                out_of_bounds = t[(t < 0) | (t >= self._timesteps)].tolist()
                raise ValueError(
                    f"Timestep tensor contains values outside [0, {self._timesteps - 1}]: {out_of_bounds}"
                )
            return t.to(device=device, dtype=torch.long)

        raise TypeError(f"Timestep must be int or torch.Tensor, got {type(t).__name__}")

    def _extract(
        self,
        buffer: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size,
        is_batched: bool = True,
    ) -> torch.Tensor:
        """Extract coefficients from buffer at indices t and reshape for correct broadcasting.

        Args:
            buffer: 1-D buffer tensor of shape (T,).
            t: Formatted 0-D or 1-D long tensor of indices.
            x_shape: Shape of target latent patch tensor x_0.
            is_batched: Whether x_0 has a leading batch dimension.

        Returns:
            Broadcastable tensor with singleton trailing dimensions.
        """
        if not is_batched:
            # Unbatched tensor: t must represent a single timestep
            scalar_idx = t.squeeze().item() if t.numel() == 1 else t.item()
            val = buffer[scalar_idx]
            # Reshape to (1, 1, ..., 1) matching the rank of x_0
            return val.view(*((1,) * len(x_shape)))

        # Batched tensor: t can be scalar (0-D/1-element) or 1-D with length equal to batch size
        if t.ndim == 0 or t.numel() == 1:
            val = buffer[t.squeeze()]
            # Shape (1, 1, ..., 1) broadcasts across batch and all feature dimensions
            return val.view(1, *((1,) * (len(x_shape) - 1)))

        # Vector t of length B: shape (B, 1, ..., 1)
        extracted = buffer.gather(0, t)
        target_shape = (extracted.shape[0],) + (1,) * (len(x_shape) - 1)
        return extracted.view(*target_shape)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        is_batched: Optional[bool] = None,
    ) -> torch.Tensor:
        """Sample noisy latents x_t at timestep(s) t according to the forward diffusion equation:

            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0: Clean latent tensor. Supports arbitrary patch geometries:
                 - Batched 5D latent patches: [B, N, C, P_h, P_w] (e.g. [B, 9, 4, 4, 4])
                 - Batched 3D flattened/projected patches: [B, N, D]
                 - Batched 4D latent grid: [B, C, H, W]
                 - Batched 2D vectors: [B, D]
                 - Unbatched patch tensors: [N, C, P_h, P_w] or [N, D]
            t: Timestep(s). Can be an integer, 0-D tensor, or 1-D tensor of length B.
            noise: Optional pre-generated noise tensor epsilon ~ N(0, I) matching x_0.
            generator: Optional torch.Generator for reproducible noise sampling.
            seed: Optional integer seed for deterministic noise sampling.
            is_batched: Explicitly specify whether x_0 has a batch dimension. If None,
                        inferred from tensor rank and shape.

        Returns:
            Noisy latent tensor x_t matching shape, dtype, and device of x_0.

        Raises:
            TypeError: If x_0 is not a tensor, or seed is not an integer.
            ValueError: If inputs contain non-finite values (NaN/Inf), both seed and generator
                        are supplied, generator device is incompatible, or vector timesteps
                        are provided for unbatched tensors.
        """
        # 1. Validate x_0
        if not isinstance(x_0, torch.Tensor):
            raise TypeError(f"x_0 must be a torch.Tensor, got {type(x_0).__name__}")
        if not torch.isfinite(x_0).all():
            raise ValueError("Input tensor x_0 contains non-finite values (NaN or Inf).")

        # 2. Determine batching semantics
        if is_batched is None:
            # In PHQD Module 1:
            # 5D [B, 9, 4, 4, 4] is batched patches.
            # 4D [9, 4, 4, 4] is unbatched image patches.
            # 4D [B, 4, 8, 8] is batched latent grids.
            # 3D [C, 8, 8] is unbatched latent grid.
            # 1D [D] is unbatched vector.
            if x_0.ndim == 5:
                inferred_batched = True
            elif x_0.ndim == 4 and x_0.shape == (9, 4, 4, 4):
                inferred_batched = False
            elif x_0.ndim <= 1:
                inferred_batched = False
            else:
                inferred_batched = True
        else:
            inferred_batched = bool(is_batched)

        # 3. Format and validate timestep(s) t
        t_tensor = self._validate_and_format_timesteps(t, device=self.alphas_cumprod.device)

        # 4. Reject vector timesteps for unbatched tensors
        if not inferred_batched:
            if t_tensor.numel() > 1:
                raise ValueError(
                    f"Vector timesteps of length {t_tensor.numel()} are not supported for unbatched tensors. "
                    "Timestep t must be a scalar (int or 0-D tensor)."
                )
        else:
            # Batched tensor: if t has multiple elements, its length must match batch size
            if t_tensor.numel() > 1:
                if t_tensor.numel() != x_0.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: x_0 has batch size {x_0.shape[0]}, "
                        f"but timestep vector t has length {t_tensor.numel()}."
                    )

        # 5. Handle noise generation, determinism, and compatibility
        if seed is not None and generator is not None:
            raise ValueError("Cannot specify both 'seed' and 'generator' simultaneously. Choose either seed or generator.")

        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
            gen_device = x_0.device.type if x_0.device.type in ("cpu", "cuda") else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        if generator is not None:
            if generator.device.type != x_0.device.type:
                raise ValueError(
                    f"Generator device '{generator.device.type}' does not match tensor device '{x_0.device.type}'."
                )

        if noise is None:
            noise = torch.randn(
                x_0.shape,
                generator=generator,
                device=x_0.device,
                dtype=x_0.dtype,
            )
        else:
            if not isinstance(noise, torch.Tensor):
                raise TypeError(f"noise must be a torch.Tensor, got {type(noise).__name__}")
            if noise.shape != x_0.shape:
                raise ValueError(f"Noise shape {noise.shape} does not match x_0 shape {x_0.shape}.")
            if noise.device != x_0.device:
                raise ValueError(f"Noise device '{noise.device}' does not match x_0 device '{x_0.device}'.")
            if noise.dtype != x_0.dtype:
                raise ValueError(f"Noise dtype '{noise.dtype}' does not match x_0 dtype '{x_0.dtype}'.")
            if not torch.isfinite(noise).all():
                raise ValueError("Noise tensor contains non-finite values (NaN or Inf).")

        # 6. Extract forward diffusion coefficients
        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod,
            t_tensor,
            x_0.shape,
            is_batched=inferred_batched,
        ).to(device=x_0.device, dtype=x_0.dtype)

        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t_tensor,
            x_0.shape,
            is_batched=inferred_batched,
        ).to(device=x_0.device, dtype=x_0.dtype)

        # 7. Apply forward diffusion closed-form equation: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        if not torch.isfinite(x_t).all():
            raise ValueError("Forward diffusion produced non-finite values (NaN or Inf).")

        return x_t

    def sample_noisy_latents(
        self,
        x_0: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        is_batched: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy latents x_t and return both (x_t, noise) for loss computation.

        Returns:
            Tuple of (x_t, noise) where both tensors share shape, dtype, and device with x_0.
        """
        # If noise is not provided, generate it beforehand so the exact noise target is returned
        if noise is None:
            if seed is not None and generator is not None:
                raise ValueError("Cannot specify both 'seed' and 'generator' simultaneously.")
            if seed is not None:
                if not isinstance(seed, int):
                    raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
                gen_device = x_0.device.type if x_0.device.type in ("cpu", "cuda") else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(seed)
            if generator is not None and generator.device.type != x_0.device.type:
                raise ValueError(
                    f"Generator device '{generator.device.type}' does not match tensor device '{x_0.device.type}'."
                )

            noise = torch.randn(
                x_0.shape,
                generator=generator,
                device=x_0.device,
                dtype=x_0.dtype,
            )

        x_t = self.q_sample(
            x_0=x_0,
            t=t,
            noise=noise,
            is_batched=is_batched,
        )
        return x_t, noise

    def forward(
        self,
        x_0: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        is_batched: Optional[bool] = None,
    ) -> torch.Tensor:
        """Alias for q_sample."""
        return self.q_sample(
            x_0=x_0,
            t=t,
            noise=noise,
            generator=generator,
            seed=seed,
            is_batched=is_batched,
        )

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        is_batched: Optional[bool] = None,
    ) -> torch.Tensor:
        """Synonym for q_sample."""
        return self.q_sample(
            x_0=x_0,
            t=t,
            noise=noise,
            generator=generator,
            seed=seed,
            is_batched=is_batched,
        )


# Direct alias
DiffusionSchedule = LinearDiffusionSchedule
