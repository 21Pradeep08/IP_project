#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Ensemble Multi-seed Strategies for DifFace
# Exploits the stochastic diversity of diffusion sampling (see paper Fig. 11, 17)
# Strategies: mean averaging, weighted averaging, and Best-of-N selection.

import os
import cv2
import random
import shutil
import tempfile
import numpy as np
from pathlib import Path

import torch


def set_all_seeds(seed):
    """Set random seed for all relevant RNGs (torch, numpy, python random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_sharpness(image):
    """
    Compute sharpness score of an image using Laplacian variance.

    Args:
        image: H x W x C numpy array, uint8

    Returns:
        sharpness: float, higher = sharper
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def ensemble_restore(sampler, in_path, out_dir, num_seeds=5, seeds=None,
                     start_timesteps=100, task='restoration', eta=0.5,
                     gamma=0.0, bs=1, draw_box=False, aligned=True):
    """
    Run DifFace inference num_seeds times with different random seeds and
    average the results in pixel space (equal weights).

    Args:
        sampler: DifFaceSampler instance
        in_path: str, path to a single input image
        out_dir: str, output directory for results
        num_seeds: int, number of seeds to use
        seeds: list[int] or None, explicit seed list (overrides num_seeds)
        start_timesteps: int, starting timestep N (respaced domain)
        task: 'restoration' or 'inpainting'
        eta: ddim eta parameter
        gamma: regularization parameter
        bs: batch size
        draw_box: draw face boxes (unaligned mode)
        aligned: whether faces are aligned

    Returns:
        ensemble_result: H x W x C numpy array, uint8, BGR
        individual_results: list of (seed, image, sharpness) tuples
    """
    if seeds is None:
        seeds = list(range(num_seeds))

    in_path = Path(in_path)
    out_dir = Path(out_dir)

    individual_results = []
    all_images_float = []

    for i, seed in enumerate(seeds):
        print(f"    [Ensemble] Seed {i+1}/{len(seeds)}: seed={seed}", flush=True)

        # Reset all random states
        set_all_seeds(seed)
        sampler.setup_seed(seed)

        # Run inference into a temporary directory
        with tempfile.TemporaryDirectory() as temp_out:
            sampler.inference(
                in_path=str(in_path),
                out_path=temp_out,
                bs=bs,
                start_timesteps=start_timesteps,
                task=task,
                need_restoration=True,
                gamma=gamma,
                num_update=1,
                draw_box=draw_box,
                suffix=None,
                eta=eta if task == 'restoration' else 1.0,
                mask_back=True,
            )

            # Read the result
            if aligned:
                res_dir = Path(temp_out) / 'restored_faces'
            else:
                res_dir = Path(temp_out) / 'restored_image'

            res_files = sorted(res_dir.glob('*.png'))
            if not res_files:
                print(f"    [WARN] No output for seed={seed}, skipping")
                continue

            result_img = cv2.imread(str(res_files[0]))  # BGR, uint8

        # Compute sharpness
        sharpness = compute_sharpness(result_img)
        print(f"      Sharpness score: {sharpness:.1f}", flush=True)

        individual_results.append((seed, result_img, sharpness))
        all_images_float.append(result_img.astype(np.float64))

    if not all_images_float:
        raise RuntimeError("Ensemble produced no valid outputs!")

    # ---- Equal-weight averaging ----
    ensemble_avg = np.mean(np.stack(all_images_float, axis=0), axis=0)
    ensemble_result = np.clip(ensemble_avg, 0, 255).astype(np.uint8)

    print(f"    [Ensemble] Mean averaging done over {len(all_images_float)} seeds", flush=True)

    return ensemble_result, individual_results


def weighted_ensemble_restore(sampler, in_path, out_dir, num_seeds=5, seeds=None,
                              start_timesteps=100, task='restoration', eta=0.5,
                              gamma=0.0, bs=1, draw_box=False, aligned=True):
    """
    Run DifFace inference num_seeds times with different random seeds and
    average the results using sharpness-based weights (sharper outputs
    contribute more to the final result).

    Returns:
        ensemble_result: H x W x C numpy array, uint8, BGR
        individual_results: list of (seed, image, sharpness) tuples
        weights: list of float weights assigned to each seed
    """
    # First, get all individual results using the base ensemble function
    _, individual_results = ensemble_restore(
        sampler=sampler,
        in_path=in_path,
        out_dir=out_dir,
        num_seeds=num_seeds,
        seeds=seeds,
        start_timesteps=start_timesteps,
        task=task,
        eta=eta,
        gamma=gamma,
        bs=bs,
        draw_box=draw_box,
        aligned=aligned,
    )

    # ---- Compute sharpness-based weights ----
    sharpness_scores = [s for _, _, s in individual_results]
    total_sharpness = sum(sharpness_scores)

    if total_sharpness == 0:
        # Fallback to equal weights if all sharpness scores are zero
        weights = [1.0 / len(individual_results)] * len(individual_results)
    else:
        weights = [s / total_sharpness for s in sharpness_scores]

    print(f"\n    [Weighted Ensemble] Weights:", flush=True)
    for (seed, _, sharpness), w in zip(individual_results, weights):
        print(f"      Seed {seed}: sharpness={sharpness:.1f}, weight={w:.4f}", flush=True)

    # ---- Weighted averaging ----
    weighted_sum = np.zeros_like(individual_results[0][1], dtype=np.float64)
    for (_, img, _), w in zip(individual_results, weights):
        weighted_sum += img.astype(np.float64) * w

    ensemble_result = np.clip(weighted_sum, 0, 255).astype(np.uint8)

    print(f"    [Weighted Ensemble] Weighted averaging done over {len(individual_results)} seeds", flush=True)

    return ensemble_result, individual_results, weights


def best_of_n_restore(sampler, in_path, out_dir, num_seeds=5, seeds=None,
                      start_timesteps=100, task='restoration', eta=0.5,
                      gamma=0.0, bs=1, draw_box=False, aligned=True):
    """
    Run DifFace inference num_seeds times with different random seeds and
    select the BEST single output based on sharpness (Laplacian variance).

    Unlike pixel-space averaging, this preserves full image detail because
    the final result IS one of the actual model outputs, not a blend.
    By running multiple seeds and picking the sharpest, we exploit diffusion
    stochasticity to guarantee a result at least as good as single-seed,
    and usually better.

    Returns:
        best_result: H x W x C numpy array, uint8, BGR (the winning output)
        individual_results: list of (seed, image, sharpness) tuples
        best_idx: int, index of the selected best result
    """
    if seeds is None:
        seeds = list(range(num_seeds))

    in_path = Path(in_path)
    individual_results = []

    for i, seed in enumerate(seeds):
        print(f"    [Best-of-N] Seed {i+1}/{len(seeds)}: seed={seed}", flush=True)

        # Reset all random states
        set_all_seeds(seed)
        sampler.setup_seed(seed)

        # Run inference into a temporary directory
        with tempfile.TemporaryDirectory() as temp_out:
            sampler.inference(
                in_path=str(in_path),
                out_path=temp_out,
                bs=bs,
                start_timesteps=start_timesteps,
                task=task,
                need_restoration=True,
                gamma=gamma,
                num_update=1,
                draw_box=draw_box,
                suffix=None,
                eta=eta if task == 'restoration' else 1.0,
                mask_back=True,
            )

            # Read the result
            if aligned:
                res_dir = Path(temp_out) / 'restored_faces'
            else:
                res_dir = Path(temp_out) / 'restored_image'

            res_files = sorted(res_dir.glob('*.png'))
            if not res_files:
                print(f"    [WARN] No output for seed={seed}, skipping")
                continue

            result_img = cv2.imread(str(res_files[0]))  # BGR, uint8

        # Compute sharpness
        sharpness = compute_sharpness(result_img)
        print(f"      Sharpness score: {sharpness:.1f}", flush=True)

        individual_results.append((seed, result_img, sharpness))

    if not individual_results:
        raise RuntimeError("Best-of-N produced no valid outputs!")

    # ---- Select the best (sharpest) output ----
    best_idx = max(range(len(individual_results)), key=lambda i: individual_results[i][2])
    best_seed, best_result, best_sharpness = individual_results[best_idx]

    print(f"\n    [Best-of-N] Winner: Seed {best_seed} "
          f"(sharpness={best_sharpness:.1f}, index={best_idx})", flush=True)
    print(f"    [Best-of-N] Sharpness range: "
          f"{min(s for _,_,s in individual_results):.1f} - "
          f"{max(s for _,_,s in individual_results):.1f}", flush=True)

    return best_result, individual_results, best_idx


def create_ensemble_comparison_strip(lq_img, individual_results, ensemble_result,
                                     ensemble_mode='mean', weights=None,
                                     best_idx=None):
    """
    Create a visual comparison strip:
      [Input | Seed 0 | Seed 1 | ... | Seed N | Final Result]

    Args:
        lq_img: H x W x C, uint8, BGR
        individual_results: list of (seed, image, sharpness)
        ensemble_result: H x W x C, uint8, BGR
        ensemble_mode: 'mean', 'weighted', or 'best'
        weights: list of floats (only for weighted mode)
        best_idx: int, index of the winning seed (only for best mode)

    Returns:
        strip: numpy array, uint8, BGR
    """
    h, w = lq_img.shape[:2]

    # Collect all panels
    panels = []
    labels = []

    # Input panel
    panels.append(cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_LANCZOS4))
    labels.append("Input (LQ)")

    # Individual seed panels
    for i, (seed, img, sharpness) in enumerate(individual_results):
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        # Highlight the winning seed in Best-of-N mode with green border
        if ensemble_mode == 'best' and best_idx is not None and i == best_idx:
            border = 3
            cv2.rectangle(resized, (0, 0), (w-1, h-1), (0, 255, 0), border)
            labels.append(f"BEST Seed {seed} (sharp={sharpness:.0f})")
        elif weights is not None:
            labels.append(f"Seed {seed} (s={sharpness:.0f}, w={weights[i]:.2f})")
        else:
            labels.append(f"Seed {seed} (sharp={sharpness:.0f})")
        panels.append(resized)

    # Final result panel
    panels.append(cv2.resize(ensemble_result, (w, h), interpolation=cv2.INTER_LANCZOS4))
    if ensemble_mode == 'best':
        mode_label = "Best-of-N Selected"
    elif ensemble_mode == 'weighted':
        mode_label = "Weighted Avg"
    else:
        mode_label = "Mean Avg"
    labels.append(f"Result ({mode_label})")

    # Build the strip
    gap = 3
    separator = np.full((h, gap, 3), 255, dtype=np.uint8)

    parts = []
    for i, panel in enumerate(panels):
        if i > 0:
            parts.append(separator)
        parts.append(panel)

    strip = np.concatenate(parts, axis=1)

    # Add header with labels
    total_w = strip.shape[1]
    header_h = 35
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    color = (255, 255, 255)

    # Calculate panel positions
    panel_x = 0
    for i, label in enumerate(labels):
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x = panel_x + (w - text_size[0]) // 2
        y = header_h - 10
        cv2.putText(header, label, (max(x, 2), y), font, font_scale, color, thickness, cv2.LINE_AA)
        panel_x += w + gap

    strip = np.concatenate([header, strip], axis=0)

    return strip


def save_ensemble_outputs(out_dir, img_stem, lq_img, individual_results,
                          ensemble_result, ensemble_mode='mean', weights=None,
                          best_idx=None):
    """
    Save all ensemble outputs to disk.

    Saves:
      - Individual seed results as {stem}_seed_{i}.png
      - Ensemble result as {stem}_ensemble.png
      - Comparison strip as {stem}_ensemble_comparison.png

    Args:
        out_dir: output directory path
        img_stem: filename stem of the input image
        lq_img: input low-quality image, BGR uint8
        individual_results: list of (seed, image, sharpness)
        ensemble_result: final ensemble image, BGR uint8
        ensemble_mode: 'mean', 'weighted', or 'best'
        weights: optional weight list for weighted mode
        best_idx: optional int, index of winning seed for best mode
    """
    out_dir = Path(out_dir)

    # Create subdirectories
    ensemble_dir = out_dir / 'ensemble_results'
    seeds_dir = out_dir / 'individual_seeds'
    comparison_dir = out_dir / 'comparisons'

    for d in [ensemble_dir, seeds_dir, comparison_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save individual seed results
    for seed, img, sharpness in individual_results:
        save_path = seeds_dir / f'{img_stem}_seed_{seed}.png'
        cv2.imwrite(str(save_path), img)

    # Save ensemble result
    ensemble_path = ensemble_dir / f'{img_stem}_ensemble.png'
    cv2.imwrite(str(ensemble_path), ensemble_result)

    # Save comparison strip
    strip = create_ensemble_comparison_strip(
        lq_img=lq_img,
        individual_results=individual_results,
        ensemble_result=ensemble_result,
        ensemble_mode=ensemble_mode,
        weights=weights,
        best_idx=best_idx,
    )
    strip_path = comparison_dir / f'{img_stem}_ensemble_comparison.png'
    cv2.imwrite(str(strip_path), strip)

    print(f"    Saved: {ensemble_path}")
    print(f"    Saved: {strip_path}")
    print(f"    Individual seeds saved to: {seeds_dir}")

    return ensemble_path, strip_path


if __name__ == '__main__':
    """Quick test: compute sharpness on a given image."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ensemble.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read: {sys.argv[1]}")
        sys.exit(1)

    sharpness = compute_sharpness(img)
    print(f"Sharpness (Laplacian variance): {sharpness:.2f}")
