#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Degradation-Aware Dynamic N Selection for DifFace
# Computes degradation severity from a low-quality input and maps it
# to an adaptive starting timestep N for the diffusion reverse process.

import cv2
import numpy as np


def estimate_degradation_severity(lq_image):
    """
    Estimate the degradation severity of a low-quality face image.

    Uses two complementary metrics:
      1. Laplacian variance (blur detection) — lower variance = more blur = more degraded
      2. Gaussian difference (noise estimation) — higher residual energy = more noise = more degraded

    Args:
        lq_image: numpy array, H x W x C, uint8 or float32 [0,1], RGB or BGR.
                  Can also be a file path (str).

    Returns:
        severity: float in [0, 1], where 0 = clean / minimal degradation,
                  1 = severely degraded.
        info: dict with sub-scores for logging.
    """
    # --- Load / convert image ---
    if isinstance(lq_image, str):
        img = cv2.imread(lq_image, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {lq_image}")
    else:
        img = lq_image.copy()
        # Convert float [0,1] to uint8 if needed
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Convert to grayscale for analysis
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    else:
        gray = img

    # Resize to a canonical size for consistent scoring across resolutions
    canonical_size = (256, 256)
    gray_resized = cv2.resize(gray, canonical_size, interpolation=cv2.INTER_AREA)

    # ----------------------------------------------------------------
    # 1) Blur detection via Laplacian variance
    #    A sharp image has high Laplacian variance; a blurry one has low.
    # ----------------------------------------------------------------
    laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
    lap_var = laplacian.var()

    # Empirical thresholds for face images:
    #   - Very blurry faces: lap_var < 50
    #   - Moderately blurry:  50 - 300
    #   - Sharp faces:        lap_var > 800
    LAP_LOW = 30.0    # below this → severity = 1
    LAP_HIGH = 800.0  # above this → severity = 0

    blur_score = 1.0 - np.clip((lap_var - LAP_LOW) / (LAP_HIGH - LAP_LOW), 0.0, 1.0)

    # ----------------------------------------------------------------
    # 2) Noise estimation via Difference of Gaussians (DoG)
    #    Subtract a mildly smoothed version from a more smoothed version;
    #    the residual captures high-frequency noise energy.
    # ----------------------------------------------------------------
    gray_f = gray_resized.astype(np.float64)
    blur_small = cv2.GaussianBlur(gray_f, (3, 3), 0.5)
    blur_large = cv2.GaussianBlur(gray_f, (7, 7), 1.5)
    dog = blur_small - blur_large
    noise_energy = np.std(dog)

    # Empirical thresholds:
    #   - Clean images:  noise_energy < 3
    #   - Noisy images:  noise_energy > 15
    NOISE_LOW = 2.0
    NOISE_HIGH = 18.0

    noise_score = np.clip((noise_energy - NOISE_LOW) / (NOISE_HIGH - NOISE_LOW), 0.0, 1.0)

    # ----------------------------------------------------------------
    # 3) Combine: weighted average (blur is usually dominant in face restoration)
    # ----------------------------------------------------------------
    BLUR_WEIGHT = 0.65
    NOISE_WEIGHT = 0.35
    severity = float(np.clip(BLUR_WEIGHT * blur_score + NOISE_WEIGHT * noise_score, 0.0, 1.0))

    info = {
        'laplacian_var': float(lap_var),
        'blur_score': float(blur_score),
        'noise_energy': float(noise_energy),
        'noise_score': float(noise_score),
        'severity': severity,
    }

    return severity, info


def select_N_adaptive(lq_image, N_min=250, N_max=500):
    """
    Dynamically select the starting timestep N based on image degradation severity.

    Higher severity → larger N (more diffusion steps for heavier restoration).
    Lower severity → smaller N (fewer steps preserve fidelity of less-degraded input).

    IMPORTANT: This codebase uses ddim250 respacing by default. In that regime,
    the timestep index ranges from 0 to 249. The paper's T_s = 400 (out of 1000)
    maps to index ≈ 100 in the respaced schedule. So N_min/N_max here are in the
    **original** 1000-step domain and will be converted to the respaced domain.

    Args:
        lq_image: numpy array (H x W x C) or file path string.
        N_min: int, minimum starting timestep (original 1000-step domain).
                Default 250 → respaced index ≈ 62.
        N_max: int, maximum starting timestep (original 1000-step domain).
                Default 500 → respaced index ≈ 125.

    Returns:
        N_respaced: int, the selected starting timestep in the respaced (ddim250) domain.
        severity: float, the degradation severity score.
        info: dict, detailed breakdown of the degradation analysis.
    """
    severity, info = estimate_degradation_severity(lq_image)

    # Linear interpolation in the original 1000-step domain
    N_original = int(round(N_min + severity * (N_max - N_min)))

    # Convert to respaced domain: original_steps=1000, respaced_steps=250
    # respaced_index = original_timestep * (respaced_steps / original_steps)
    ORIGINAL_STEPS = 1000
    RESPACED_STEPS = 250
    N_respaced = int(round(N_original * RESPACED_STEPS / ORIGINAL_STEPS))

    # Clamp to valid range
    N_respaced = max(1, min(N_respaced, RESPACED_STEPS - 1))

    info['N_original_domain'] = N_original
    info['N_respaced'] = N_respaced

    return N_respaced, severity, info


def create_comparison_image(lq_img, fixed_result, adaptive_result,
                            severity, N_fixed, N_adaptive):
    """
    Create a side-by-side comparison image:
      [Input | Fixed N Result | Adaptive N Result]

    Args:
        lq_img: H x W x C, uint8, BGR or RGB
        fixed_result: H x W x C, uint8, same order
        adaptive_result: H x W x C, uint8, same order
        severity: float, degradation severity
        N_fixed: int, the fixed N used
        N_adaptive: int, the adaptive N used

    Returns:
        comparison: numpy array, H x (3W + gaps) x C, uint8
    """
    # Ensure all images are the same size
    h, w = lq_img.shape[:2]
    fixed_result = cv2.resize(fixed_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
    adaptive_result = cv2.resize(adaptive_result, (w, h), interpolation=cv2.INTER_LANCZOS4)

    gap = 4  # pixel gap between panels
    gap_color = (255, 255, 255)  # white separator

    # Create separator strip
    separator = np.full((h, gap, 3), gap_color, dtype=np.uint8)

    # Stack horizontally
    comparison = np.concatenate([lq_img, separator, fixed_result, separator, adaptive_result], axis=1)

    # Add text labels at the top
    total_w = comparison.shape[1]
    header_h = 40
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    # Label positions (centered in each panel)
    labels = [
        f"Input (severity={severity:.2f})",
        f"Fixed N={N_fixed}",
        f"Adaptive N={N_adaptive}",
    ]
    panel_starts = [0, w + gap, 2 * (w + gap)]

    for label, x_start in zip(labels, panel_starts):
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x = x_start + (w - text_size[0]) // 2
        y = header_h - 10
        cv2.putText(header, label, (max(x, 0), y), font, font_scale, color, thickness, cv2.LINE_AA)

    comparison = np.concatenate([header, comparison], axis=0)

    return comparison


if __name__ == '__main__':
    """Quick test: run on a single image to see degradation analysis."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python adaptive_n.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    severity, info = estimate_degradation_severity(img_path)
    N_respaced, _, _ = select_N_adaptive(img_path)

    print(f"\n{'='*60}")
    print(f"  Degradation Analysis: {img_path}")
    print(f"{'='*60}")
    print(f"  Laplacian variance : {info['laplacian_var']:.2f}")
    print(f"  Blur score         : {info['blur_score']:.3f}")
    print(f"  Noise energy (DoG) : {info['noise_energy']:.2f}")
    print(f"  Noise score        : {info['noise_score']:.3f}")
    print(f"  Combined severity  : {severity:.3f}")
    print(f"  Selected N (ddim250 respaced): {N_respaced}")
    print(f"  (Original fixed N=100, paper T_s=400)")
    print(f"{'='*60}\n")
