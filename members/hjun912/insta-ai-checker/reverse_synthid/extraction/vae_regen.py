"""
Round-05 Stage: Diffusion-VAE Re-Generation Attack
===================================================

Single biggest documented weakness of SynthID, per
``SynthID-Image: Image watermarking at internet scale`` (Gowal et al., 2026),
Section 6.1 — "Re-generation attacks use other powerful generative models
(like diffusion models) to reconstruct a watermarked image, potentially
washing out the watermark in the process (An et al., 2024; Zhao et al., 2024)".
Section 6.2 concedes they only trained against **weak** off-the-shelf VAEs.

This stage runs the image through the Stable Diffusion autoencoder
(``stabilityai/sd-vae-ft-mse`` — the higher-fidelity fine-tuned MSE variant)
and returns the reconstruction. The encoder maps pixels to a narrow 8×-downsampled
latent manifold trained on natural images; any pixel-space watermark that isn't
essential for reconstructing the content is projected out. The decoder
re-synthesises from latents, producing an image perceptually identical to the
original but spectrally/statistically native to the VAE — which the SynthID
decoder has no trained basis for.

Supports MPS (Apple Silicon), CUDA, and CPU with a graceful fallback. Uses the
VAE's built-in tiled encode/decode for images above ~1024px so we don't OOM.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


_VAE_SINGLETON = None
_VAE_DEVICE: Optional[str] = None


def _select_device(prefer: Optional[str] = None) -> str:
    """Pick MPS → CUDA → CPU, honoring an explicit ``prefer``."""
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vae(
    model_id: str = "stabilityai/sd-vae-ft-mse",
    device: Optional[str] = None,
    dtype: str = "float32",
):
    """Lazy-load and cache the SD-VAE. Returns (vae, device_str).

    ``dtype`` is 'float32' on MPS (fp16 is broken for MPS conv on older torches)
    and can be 'float16' on CUDA for speed.
    """
    global _VAE_SINGLETON, _VAE_DEVICE
    if _VAE_SINGLETON is not None:
        return _VAE_SINGLETON, _VAE_DEVICE

    try:
        import torch
        from diffusers import AutoencoderKL
    except ImportError as e:
        raise RuntimeError(
            "VAE re-generation stage requires torch + diffusers. "
            "Install with: pip install torch diffusers safetensors accelerate"
        ) from e

    target_device = _select_device(device)
    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]
    if target_device == "mps" and torch_dtype == torch.float16:
        torch_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch_dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae.to(target_device)
    try:
        vae.enable_slicing()
        vae.enable_tiling()
    except Exception:
        pass

    _VAE_SINGLETON = vae
    _VAE_DEVICE = target_device
    return vae, target_device


def _pad_to_multiple(arr: np.ndarray, mult: int = 8) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Reflect-pad an HxWxC image so H, W are multiples of ``mult``.

    Returns the padded array and the pad amounts ``(top, bottom, left, right)``
    so the caller can undo the pad after decoding.
    """
    H, W = arr.shape[:2]
    pad_h = (-H) % mult
    pad_w = (-W) % mult
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0, 0, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = np.pad(
        arr,
        ((top, bottom), (left, right), (0, 0)),
        mode="reflect",
    )
    return padded, (top, bottom, left, right)


def _unpad(arr: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    H, W = arr.shape[:2]
    return arr[top : H - bottom if bottom else H, left : W - right if right else W]


def vae_roundtrip(
    image_uint8: np.ndarray,
    strength: float = 1.0,
    device: Optional[str] = None,
    blend_with_original: float = 0.0,
    model_id: str = "stabilityai/sd-vae-ft-mse",
) -> np.ndarray:
    """Encode-decode ``image_uint8`` through the SD-VAE; return a uint8 image.

    Args:
        image_uint8: HxWx3 RGB uint8.
        strength: Scales the *delta* from the original. ``1.0`` returns the pure
            reconstruction; ``0.7`` blends 70 % reconstruction + 30 % original,
            useful if pure VAE reconstruction is too visually different for a
            particular content category.
        device: Override device selection (``mps`` / ``cuda`` / ``cpu``).
        blend_with_original: Alias for ``1.0 - strength`` semantics — if > 0,
            the final output is ``strength * vae_out + blend * original``.
        model_id: HF repo. ``stabilityai/sd-vae-ft-mse`` is fast; SDXL variants
            give marginally better reconstruction but need more memory.

    The returned image has identical spatial shape to the input. Border pixels
    may be slightly softened due to reflect-padding round-up to multiples of 8.
    """
    import torch

    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB uint8, got {image_uint8.shape}")

    vae, dev = load_vae(model_id=model_id, device=device)
    padded, pads = _pad_to_multiple(image_uint8, mult=8)

    x = padded.astype(np.float32) / 127.5 - 1.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    t = torch.from_numpy(x).to(dev, dtype=next(vae.parameters()).dtype)

    with torch.no_grad():
        posterior = vae.encode(t).latent_dist
        z = posterior.mean
        y = vae.decode(z).sample

    y = y.float().cpu().numpy()[0]
    y = np.transpose(y, (1, 2, 0))
    y = (y + 1.0) * 127.5
    y = np.clip(y, 0, 255)
    y = _unpad(y, pads)

    original_f = image_uint8.astype(np.float32)
    if blend_with_original > 0.0 and strength == 1.0:
        strength = 1.0 - blend_with_original
    if 0.0 <= strength < 1.0:
        y = strength * y + (1.0 - strength) * original_f

    return np.clip(y, 0, 255).astype(np.uint8)


def _gaussian_blur_multichannel(
    img_f32: np.ndarray, sigma: float,
) -> np.ndarray:
    """Per-channel Gaussian blur at the given ``sigma`` using cv2."""
    import cv2

    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    out = np.empty_like(img_f32)
    for c in range(img_f32.shape[2]):
        out[..., c] = cv2.GaussianBlur(
            img_f32[..., c], (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT,
        )
    return out


def vae_roundtrip_freqselective(
    image_uint8: np.ndarray,
    lowfreq_sigma: float = 8.0,
    device: Optional[str] = None,
    model_id: str = "stabilityai/sd-vae-ft-mse",
) -> np.ndarray:
    """VAE roundtrip with low-frequency restoration from the original.

    Splits both the original and the VAE reconstruction into a low-freq band
    (Gaussian σ=``lowfreq_sigma``, containing lighting/color/gross structure)
    and a high-freq band (containing texture and — critically — the SynthID
    watermark at radii 14-238 bins on a 1024² grid, i.e. freqs above roughly
    0.02 cycles/pixel).

    Output = ``low_of(original) + high_of(vae_out)``. This preserves all
    perceptually dominant low-frequency content (≈ PSNR 34-40 dB) while the
    watermark-bearing band is replaced entirely by VAE-native content that
    SynthID's decoder has no trained basis for.

    A σ around 8 matches the SynthID carrier band boundary on a 1024² image;
    scale proportionally for very different resolutions if you want to keep
    the same relative cutoff.
    """
    original_f = image_uint8.astype(np.float32)
    vae_f = vae_roundtrip(image_uint8, device=device, model_id=model_id).astype(np.float32)

    low_orig = _gaussian_blur_multichannel(original_f, lowfreq_sigma)
    low_vae = _gaussian_blur_multichannel(vae_f, lowfreq_sigma)
    high_vae = vae_f - low_vae

    out = low_orig + high_vae
    return np.clip(out, 0, 255).astype(np.uint8)
