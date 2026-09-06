"""
Diffusion-to-PQC Denoiser Interface for PHQD Module 3 Milestone 2A.

Provides:
- DiffusionPQCDenoiser: Hybrid neural-quantum module implementing the pipeline:
  [B, 9, 4, 4, 4] -> [B, 9, 64] -> fixed 64->6 projection -> [B, 9, 6]
                  -> diffusion q_sample -> [B, 9, 6] -> shared PQC denoiser -> [B, 9, 6].

Architectural Principles:
1. Shared Quantum Weights: A single 6-qubit PQC instance with variational depth D=3
   (default 18 trainable quantum parameters) is shared across all 9 spatial patch locations.
2. Distinguishable Patch Positions: The 9 patch positions remain explicitly distinguishable
   in the output tensor geometry [B, 9, 6].
3. Zero Trainable Projection Parameters & Scientific Integrity:
   - Experimental Path: Training and evaluation experiments MUST supply a pre-fitted
     FixedPatchProjector via DiffusionPQCDenoiser(projector=...) containing normalization
     statistics fitted strictly on training patches only (preventing data leakage).
   - Test Fallback: The automatically created projector (when projector=None) uses uncalibrated
     placeholder normalization statistics (mean_64=0, std_6=1) strictly for isolated unit tests,
     shape verification, and smoke tests, and is not fitted for real experiments.
   - Zero Leakage: DiffusionPQCDenoiser deliberately does NOT fit the projector internally,
     ensuring normalization statistics cannot accidentally be computed over validation or test data.
4. Classical Forward Diffusion Schedule: Integrates the existing LinearDiffusionSchedule
   from Module 3 Milestone 1 without duplication of schedule logic.
5. Projected-Space Denoised Representation: Output is explicitly a denoised 6-D projected-space
   representation of shape [B, 9, 6] with values in [-1, 1] (Pauli-Z expectation values).
   No inverse projection or spatial patch reconstruction is performed in Milestone 2A.
"""

from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn

from src.models.pqc import PQCDenoiser
from src.models.projection import FixedPatchProjector
from src.training.diffusion_schedule import LinearDiffusionSchedule


class DiffusionPQCDenoiser(nn.Module):
    """Diffusion-to-PQC denoiser interface over spatial latent patches.

    Pipeline:
        x_0 [B, 9, 4, 4, 4] (or [B, 9, 64])
          -> fixed 64->6 projection -> [B, 9, 6]
          -> forward diffusion q_sample(t) -> [B, 9, 6]
          -> shared PQC denoiser -> [B, 9, 6] (in [-1, 1])

    Attributes:
        projector: Frozen FixedPatchProjector (0 trainable parameters).
        schedule: LinearDiffusionSchedule (5 core non-trainable buffers).
        pqc: PQCDenoiser (6 qubits, depth 3 -> 18 trainable parameters).
        uses_placeholder_projector: Bool indicating if the fallback uncalibrated projector is in use.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        pqc_depth: int = 3,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        projector: Optional[FixedPatchProjector] = None,
        schedule: Optional[LinearDiffusionSchedule] = None,
        projector_seed: int = 42,
        pqc_dev: Optional[Any] = None,
    ):
        """Initialize the Diffusion-to-PQC denoiser pipeline.

        Args:
            n_qubits: Number of qubits and projected dimension. Defaults to 6.
            pqc_depth: Variational circuit depth / layers. Defaults to 3 (locked Module 2 config).
            timesteps: Total diffusion schedule timesteps T >= 1. Defaults to 1000.
            beta_start: Beta schedule start value. Defaults to 1e-4.
            beta_end: Beta schedule end value. Defaults to 0.02.
            projector: Pre-fitted FixedPatchProjector containing training-derived normalization
                       statistics (the intended experimental path). If None, an uncalibrated
                       fallback projector with placeholder statistics is instantiated strictly
                       for unit testing and smoke tests.
            schedule: Optional pre-existing LinearDiffusionSchedule. If None, instantiates a new
                      LinearDiffusionSchedule.
            projector_seed: Random seed for semi-orthogonal projection matrix if projector is None.
            pqc_dev: Optional PennyLane device for PQC. Defaults to 'default.qubit'.
        """
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.pqc_depth = int(pqc_depth)

        # 1. Setup Fixed Projector (strictly 0 trainable parameters)
        # Methodological note: DiffusionPQCDenoiser does not fit the projector internally,
        # ensuring normalization statistics are derived strictly from training data without leakage.
        if projector is not None:
            if not isinstance(projector, FixedPatchProjector):
                raise TypeError(f"projector must be a FixedPatchProjector, got {type(projector).__name__}")
            if projector.out_dim != self.n_qubits:
                raise ValueError(
                    f"projector out_dim ({projector.out_dim}) must match n_qubits ({self.n_qubits})"
                )
            self.projector = projector
            self._uses_placeholder_projector = False
        else:
            # Fallback placeholder projector with uncalibrated zero-mean / unit-variance statistics.
            # Intended strictly for unit tests and shape verification without requiring training data.
            # Real training and evaluation experiments MUST supply a pre-fitted FixedPatchProjector.
            self.projector = FixedPatchProjector(
                in_dim=64,
                out_dim=self.n_qubits,
                seed=projector_seed,
                norm_type="standard",
                mean_64=torch.zeros(64),
                std_6=torch.ones(self.n_qubits),
            )
            self._uses_placeholder_projector = True

        # 2. Setup Classical Diffusion Schedule (strictly 0 trainable parameters)
        if schedule is not None:
            if not isinstance(schedule, LinearDiffusionSchedule):
                raise TypeError(f"schedule must be a LinearDiffusionSchedule, got {type(schedule).__name__}")
            self.schedule = schedule
        else:
            self.schedule = LinearDiffusionSchedule(
                timesteps=timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
            )

        # 3. Setup Shared PQC Denoiser (hardware-efficient RY + CNOT ansatz)
        # Default: 6 qubits x depth 3 = exactly 18 trainable parameters
        self.pqc = PQCDenoiser(
            n_qubits=self.n_qubits,
            n_layers=self.pqc_depth,
            dev=pqc_dev,
        )

    @property
    def num_trainable_parameters(self) -> int:
        """Total trainable parameters across the entire module (strictly PQC weights)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_pqc_parameters(self) -> int:
        """Total trainable parameters in the PQC denoiser (6 * pqc_depth)."""
        return self.pqc.num_parameters

    @property
    def num_projection_parameters(self) -> int:
        """Total trainable parameters in the projection layer (strictly 0)."""
        return self.projector.num_parameters

    @property
    def uses_placeholder_projector(self) -> bool:
        """Whether the model is using the fallback placeholder projector.

        Returns True if the projector was instantiated with default placeholder statistics
        (zero mean, unit std) for testing, or False if a pre-fitted experimental projector
        was provided.
        """
        return self._uses_placeholder_projector

    @property
    def timesteps(self) -> int:
        """Total number of diffusion steps T."""
        return self.schedule.timesteps

    def project_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw or flattened spatial latent patches to 6-D projected space.

        Accepts only [B, 9, 4, 4, 4] or [B, 9, 64].
        Pre-projected [B, 9, 6] tensors are strictly rejected to prevent silent bypass.

        Args:
            x: Input patch tensor of shape [B, 9, 4, 4, 4] or [B, 9, 64].

        Returns:
            Projected patch tensor of shape [B, 9, 6].

        Raises:
            TypeError: If x is not a torch.Tensor.
            ValueError: If x contains non-finite values or does not match [B, 9, 4, 4, 4] or [B, 9, 64].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
        if not torch.isfinite(x).all():
            raise ValueError("Input patches contain non-finite values (NaN or Inf).")

        # Explicitly reject tensors that do not match [B, 9, 4, 4, 4] or [B, 9, 64]
        is_5d_patch = (x.dim() == 5 and x.shape[1:] == (9, 4, 4, 4))
        is_3d_flattened = (x.dim() == 3 and x.shape[1:] == (9, 64))

        if not (is_5d_patch or is_3d_flattened):
            raise ValueError(
                f"project_patches requires input shape [B, 9, 4, 4, 4] or [B, 9, 64], got {list(x.shape)}. "
                f"Pre-projected [B, 9, {self.n_qubits}] tensors are not accepted by project_patches()."
            )

        b = x.shape[0]
        # Flatten [B, 9, 4, 4, 4] -> [B, 9, 64]
        if is_5d_patch:
            x_64 = x.reshape(b, 9, 64)
        else:
            x_64 = x

        # Forward through frozen FixedPatchProjector -> [B, 9, 6]
        return self.projector(x_64)

    def diffuse_projected_patches(
        self,
        z: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply forward diffusion noise to projected patches [B, 9, 6].

        Args:
            z: Projected patch tensor of shape [B, 9, 6].
            t: Timestep(s). Scalar int, 0-D tensor, or 1-D tensor of length B.
            noise: Optional external noise tensor matching [B, 9, 6].
            seed: Optional random seed for noise generation.
            generator: Optional torch.Generator for noise generation.

        Returns:
            Noisy projected patches [B, 9, 6].
        """
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(z).__name__}")
        if z.dim() != 3 or z.shape[1:] != (9, self.n_qubits):
            raise ValueError(
                f"Expected projected patches of shape [B, 9, {self.n_qubits}], got {list(z.shape)}"
            )

        return self.schedule.q_sample(
            x_0=z,
            t=t,
            noise=noise,
            seed=seed,
            generator=generator,
        )

    def denoise_patches(self, z_noisy: torch.Tensor) -> torch.Tensor:
        """Denoise noisy projected patches [B, 9, 6] using the shared PQC denoiser.

        The PQC weights are strictly shared across all 9 patch positions by reshaping
        [B, 9, 6] -> [B * 9, 6] before evaluation, and restoring to [B, 9, 6] to keep
        the 9 spatial patch positions distinct and distinguishable.

        Args:
            z_noisy: Noisy projected patch tensor of shape [B, 9, 6].

        Returns:
            Denoised projected patch tensor of shape [B, 9, 6] with values in [-1, 1].
        """
        if not isinstance(z_noisy, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(z_noisy).__name__}")
        if z_noisy.dim() != 3 or z_noisy.shape[1:] != (9, self.n_qubits):
            raise ValueError(
                f"Expected noisy patches of shape [B, 9, {self.n_qubits}], got {list(z_noisy.shape)}"
            )
        if not torch.isfinite(z_noisy).all():
            raise ValueError("Noisy patches contain non-finite values (NaN or Inf).")

        b = z_noisy.shape[0]

        # Reshape to [B * 9, 6] to process all 9 patch positions with the shared PQC
        z_flat = z_noisy.reshape(b * 9, self.n_qubits)

        # Evaluate through PQC Denoiser (Pauli-Z expectation measurements in [-1, 1])
        denoised_flat = self.pqc(z_flat)

        # Reshape back to [B, 9, 6] ensuring all 9 patch positions remain distinguishable
        return denoised_flat.reshape(b, 9, self.n_qubits)

    def forward(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        is_pre_projected: bool = False,
    ) -> torch.Tensor:
        """Run the full Milestone 2A forward pipeline:

            Input -> (projection if not pre-projected) -> q_sample(t) -> shared PQC -> [B, 9, 6].

        Args:
            x: Input patch tensor:
               - If is_pre_projected=False: shape [B, 9, 4, 4, 4] or [B, 9, 64].
               - If is_pre_projected=True: shape [B, 9, 6].
            t: Diffusion timestep(s). Scalar integer, 0-D tensor, or 1-D tensor of length B.
            noise: Optional external noise tensor matching projected shape [B, 9, 6].
            seed: Optional integer seed for deterministic noise sampling.
            generator: Optional torch.Generator for noise sampling.
            is_pre_projected: If True, skips projection and treats x as pre-projected [B, 9, 6].
                              If False, validates that x is [B, 9, 4, 4, 4] or [B, 9, 64]
                              and projects via project_patches().

        Returns:
            Denoised 6-D projected-space representation of shape [B, 9, 6].
            Note: Output is strictly in projected space; no inverse projection is performed.
        """
        if is_pre_projected:
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
            if x.dim() != 3 or x.shape[1:] != (9, self.n_qubits):
                raise ValueError(
                    f"is_pre_projected=True requires input shape [B, 9, {self.n_qubits}], got {list(x.shape)}"
                )
            if not torch.isfinite(x).all():
                raise ValueError("Input pre-projected patches contain non-finite values (NaN or Inf).")
            z_clean = x
        else:
            # project_patches strictly accepts [B, 9, 4, 4, 4] or [B, 9, 64]
            z_clean = self.project_patches(x)

        # 2. Forward diffusion q_sample
        z_noisy = self.diffuse_projected_patches(
            z=z_clean,
            t=t,
            noise=noise,
            seed=seed,
            generator=generator,
        )

        # 3. Shared PQC denoising
        z_denoised = self.denoise_patches(z_noisy)

        return z_denoised

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        is_pre_projected: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Execute forward pass and return all intermediate tensors for loss computation and inspection.

        Returns:
            Dict containing:
                - 'clean_projected': [B, 9, 6]
                - 'noisy_projected': [B, 9, 6]
                - 'denoised_projected': [B, 9, 6]
                - 'noise': [B, 9, 6] (the noise tensor added during diffusion)
        """
        if is_pre_projected:
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
            if x.dim() != 3 or x.shape[1:] != (9, self.n_qubits):
                raise ValueError(
                    f"is_pre_projected=True requires input shape [B, 9, {self.n_qubits}], got {list(x.shape)}"
                )
            z_clean = x
        else:
            z_clean = self.project_patches(x)

        # Sample noisy latents and retrieve noise
        z_noisy, sample_noise = self.schedule.sample_noisy_latents(
            x_0=z_clean,
            t=t,
            noise=noise,
            seed=seed,
            generator=generator,
        )

        z_denoised = self.denoise_patches(z_noisy)

        return {
            "clean_projected": z_clean,
            "noisy_projected": z_noisy,
            "denoised_projected": z_denoised,
            "noise": sample_noise,
        }
