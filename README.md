\# PHQD



Patch-Wise Hybrid Quantum Diffusion for Image Generation.



\## Research Objective



This project investigates a hybrid quantum-classical diffusion

architecture for image generation using spatial latent patches.



\## Modules



\### Module 1 — Classical Autoencoder \& Data Pipeline

\- Train convolutional autoencoder

\- Produce approximately 8x8 spatial latent representation

\- Validate reconstruction using PSNR and SSIM

\- Implement configurable patch slicing

\- Freeze the autoencoder

\- Export latent-patch dataset



\### Module 2 — PQC Circuit Design \& Simulator Setup

\- Implement angle encoding

\- Implement hardware-efficient variational circuit

\- Wrap the PQC using TorchLayer

\- Validate gradients using a toy denoising task

\- Benchmark qubit count and circuit depth



\### Module 3 — Hybrid Loss, Diffusion Schedule \& Boundary Stitching

\- Implement patch diffusion

\- Implement quantum denoiser

\- Implement patch stitching

\- Implement seam-consistency loss

\- Perform seam-loss ablations



\### Module 4 — Benchmarking \& Evaluation

\- Compare against classical and quantum baselines

\- Evaluate FID, LPIPS, PSNR and SSIM

\- Measure wall-clock time

\- Measure parameter count

\- Run required ablations

\- Use multiple random seeds



\### Module 5 — EPQD Extension

Optional and gated extension.



This module will only be implemented after Module 4

has produced sufficient evidence to justify the additional

quantum resources.



\## Project Structure



\- `src/` — source code

\- `scripts/` — reproducible training/evaluation scripts

\- `notebooks/` — exploratory notebooks

\- `configs/` — experiment configurations

\- `experiments/` — module-specific experiments

\- `data/` — datasets and latent data

\- `checkpoints/` — trained models

\- `results/` — experiment results

\- `paper/` — research paper materials



\## Status



Project infrastructure initialized.



Module 1 implementation has not yet started.

