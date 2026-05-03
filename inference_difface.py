#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-02 20:43:41
# Modified: Added Degradation-Aware Dynamic N Selection and Ensemble Multi-seed Averaging

import os
import cv2
import torch
import shutil
import argparse
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte

from utils import util_opts
from utils import util_image
from utils import util_common

from sampler import DifFaceSampler
from ResizeRight.resize_right import resize
from basicsr.utils.download_util import load_file_from_url

from adaptive_n import estimate_degradation_severity, select_N_adaptive, create_comparison_image
from ensemble import (ensemble_restore, weighted_ensemble_restore,
                      best_of_n_restore, save_ensemble_outputs, compute_sharpness)

_START_TIMESTEPS = {'restoration': 100, 'inpainting': 120}
_GAMMA = {'restoration': 0.0, 'inpainting': 0.5}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-i",
            "--in_path",
            type=str,
            default='./testdata/cropped_faces',
            help='Folder to save the low quality image',
            )
    parser.add_argument(
            "-o",
            "--out_path",
            type=str,
            default='./results',
            help='Folder to save the restored results',
            )
    parser.add_argument(
            "--aligned",
            action='store_true',
            help='Input are alinged faces',
            )
    parser.add_argument(
            "--use_fp16",
            action='store_true',
            help='Activate float16 for inference',
            )
    parser.add_argument(
            "--task",
            type=str,
            default='restoration',
            choices=['restoration', 'inpainting'],
            help='Task',
            )
    parser.add_argument(
            "--eta",
            type=float,
            default=0.5,
            help='Hyper-parameter eta in ddim',
            )
    parser.add_argument(
            "--bs",
            type=int,
            default=1,
            help='Batch size for inference',
            )
    parser.add_argument(
            "--seed",
            type=int,
            default=12345,
            help='Random Seed',
            )
    parser.add_argument(
            "--draw_box",
            action='store_true',
            help='Draw box for face in the unaligned case',
            )
    # ---- Delta 1: Adaptive N arguments ----
    parser.add_argument(
            "--adaptive_N",
            action='store_true',
            help='Enable Degradation-Aware Dynamic N Selection instead of fixed N',
            )
    parser.add_argument(
            "--N_min",
            type=int,
            default=250,
            help='Minimum starting timestep in original 1000-step domain (adaptive mode)',
            )
    parser.add_argument(
            "--N_max",
            type=int,
            default=500,
            help='Maximum starting timestep in original 1000-step domain (adaptive mode)',
            )
    parser.add_argument(
            "--compare",
            action='store_true',
            help='Generate side-by-side comparison images',
            )
    # ---- Delta 2: Ensemble arguments ----
    parser.add_argument(
            "--ensemble",
            action='store_true',
            help='Enable Ensemble Multi-seed Averaging',
            )
    parser.add_argument(
            "--num_seeds",
            type=int,
            default=5,
            help='Number of seeds for ensemble (default: 5)',
            )
    parser.add_argument(
            "--ensemble_mode",
            type=str,
            default='best',
            choices=['best', 'mean', 'weighted'],
            help='Ensemble mode: best (sharpest single output), mean (equal avg), weighted (sharpness-weighted avg)',
            )
    args = parser.parse_args()

    # configurations
    if args.task == 'restoration':
        cfg_path = 'configs/sample/iddpm_ffhq512_swinir.yaml'
    elif args.task == 'inpainting':
        cfg_path = 'configs/sample/difface_inpainting_lama256.yaml'
    else:
        raise ValueError("Only accept task types of 'restoration' and 'inpainting'!")

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.seed = args.seed
    configs.diffusion.params.timestep_respacing = 'ddim250'

    # prepare the checkpoint
    if args.task == 'restoration':
        if not Path(configs.model_ir.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/swinir_restoration512_L1.pth",
                model_dir=str(Path(configs.model_ir.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model_ir.ckpt_path).name,
                )
        if not Path(configs.model.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth",
                model_dir=str(Path(configs.model.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model.ckpt_path).name,
                )
        configs.aligned = args.aligned
    elif args.task == 'inpainting':
        if not Path(configs.model_ir.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/lama_inpainting256.pth",
                model_dir=str(Path(configs.model_ir.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model_ir.ckpt_path).name,
                )
        if not Path(configs.model.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq256_ema750000.pth",
                model_dir=str(Path(configs.model.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model.ckpt_path).name,
                )
        configs.aligned = True
    else:
        raise ValueError("Only accept task types of 'restoration' and 'inpainting'!")

    if not configs.aligned and args.bs != 1:
        args.bs = 1
        print("Resetting batchsize to be 1 for unaligned case.")

    # build the sampler for diffusion
    sampler_dist = DifFaceSampler(
            configs,
            im_size=configs.model.params.image_size,
            use_fp16=args.use_fp16,
            )

    # ---- Determine starting timestep ----
    fixed_N = _START_TIMESTEPS[args.task]

    # ---- Print active feature flags ----
    print(f"\n{'='*65}")
    print(f"  DifFace Inference")
    print(f"  Delta 1 - Adaptive N:  {'ENABLED' if args.adaptive_N else 'DISABLED (fixed N=' + str(fixed_N) + ')'}")
    print(f"  Delta 2 - Ensemble:    {'ENABLED (' + args.ensemble_mode + ', ' + str(args.num_seeds) + ' seeds)' if args.ensemble else 'DISABLED (single seed)'}")
    if args.adaptive_N:
        print(f"  N_min={args.N_min} (orig domain) | N_max={args.N_max} (orig domain)")
    print(f"{'='*65}\n")

    # ---- Resolve image list ----
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if in_path.is_dir():
        im_paths = sorted([
            p for p in in_path.iterdir()
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']
        ])
    else:
        im_paths = [in_path]

    # ---- Process each image ----
    for im_path in im_paths:
        img_bgr = cv2.imread(str(im_path))
        if img_bgr is None:
            print(f"  [SKIP] Could not read: {im_path}")
            continue

        print(f"\n  --- Processing: {im_path.name} ---")

        # Step 1: Determine N (adaptive or fixed)
        if args.adaptive_N:
            adaptive_N, severity, info = select_N_adaptive(
                img_bgr, N_min=args.N_min, N_max=args.N_max
            )
            current_N = adaptive_N
            print(f"    [Adaptive N] Laplacian var={info['laplacian_var']:.1f}, "
                  f"blur={info['blur_score']:.3f}, noise={info['noise_score']:.3f}")
            print(f"    [Adaptive N] Severity={severity:.3f} -> N={adaptive_N} "
                  f"(respaced) [orig ~= {info['N_original_domain']}]")
        else:
            current_N = fixed_N
            print(f"    Using fixed N={fixed_N}")

        # Step 2: Run inference (ensemble or single)
        if args.ensemble:
            seeds = list(range(args.num_seeds))
            print(f"    [Ensemble] Mode={args.ensemble_mode}, seeds={seeds}")

            common_kwargs = dict(
                sampler=sampler_dist,
                in_path=str(im_path),
                out_dir=str(out_path),
                num_seeds=args.num_seeds,
                seeds=seeds,
                start_timesteps=current_N,
                task=args.task,
                eta=args.eta,
                gamma=_GAMMA[args.task],
                bs=args.bs,
                draw_box=args.draw_box,
                aligned=configs.aligned,
            )

            weights = None
            best_idx = None

            if args.ensemble_mode == 'best':
                ensemble_result, individual_results, best_idx = best_of_n_restore(**common_kwargs)
            elif args.ensemble_mode == 'weighted':
                ensemble_result, individual_results, weights = weighted_ensemble_restore(**common_kwargs)
            else:  # mean
                ensemble_result, individual_results = ensemble_restore(**common_kwargs)

            # Print result quality
            result_sharpness = compute_sharpness(ensemble_result)
            avg_individual_sharpness = np.mean([s for _, _, s in individual_results])
            print(f"\n    [Result] Final sharpness: {result_sharpness:.1f}")
            print(f"    [Result] Avg individual sharpness: {avg_individual_sharpness:.1f}")
            if args.ensemble_mode == 'best':
                improvement = ((result_sharpness / avg_individual_sharpness) - 1) * 100
                print(f"    [Result] Best-of-N improvement over average: {improvement:+.1f}%")

            # Save ensemble outputs
            save_ensemble_outputs(
                out_dir=str(out_path),
                img_stem=im_path.stem,
                lq_img=img_bgr,
                individual_results=individual_results,
                ensemble_result=ensemble_result,
                ensemble_mode=args.ensemble_mode,
                weights=weights,
                best_idx=best_idx,
            )

        else:
            # ---- Single-seed inference (original behavior) ----
            sampler_dist.inference(
                    in_path=str(im_path),
                    out_path=str(out_path),
                    bs=args.bs,
                    start_timesteps=current_N,
                    task=args.task,
                    need_restoration=True,
                    gamma=_GAMMA[args.task],
                    num_update=1,
                    draw_box=args.draw_box,
                    suffix=None,
                    eta=args.eta if args.task == 'restoration' else 1.0,
                    mask_back=True,
                    )

        # Step 3: Generate adaptive N comparison if requested (single-seed only)
        if args.compare and args.adaptive_N and not args.ensemble:
            comparison_dir = out_path / 'comparisons'
            util_common.mkdir(comparison_dir, parents=True)

            restored_face_dir = out_path / 'restored_faces'
            adaptive_result_path = restored_face_dir / f'{im_path.stem}.png'

            if adaptive_result_path.exists():
                adaptive_result = cv2.imread(str(adaptive_result_path))

                fixed_out_path = out_path / '_fixed_N_temp'
                util_common.mkdir(fixed_out_path, parents=True)

                sampler_dist.inference(
                        in_path=str(im_path),
                        out_path=str(fixed_out_path),
                        bs=args.bs,
                        start_timesteps=fixed_N,
                        task=args.task,
                        need_restoration=True,
                        gamma=_GAMMA[args.task],
                        num_update=1,
                        draw_box=args.draw_box,
                        suffix=None,
                        eta=args.eta if args.task == 'restoration' else 1.0,
                        mask_back=True,
                        )

                fixed_result_path = fixed_out_path / 'restored_faces' / f'{im_path.stem}.png'
                if fixed_result_path.exists():
                    fixed_result = cv2.imread(str(fixed_result_path))

                    comparison = create_comparison_image(
                        lq_img=img_bgr,
                        fixed_result=fixed_result,
                        adaptive_result=adaptive_result,
                        severity=severity,
                        N_fixed=fixed_N,
                        N_adaptive=adaptive_N,
                    )

                    comp_path = comparison_dir / f'{im_path.stem}_comparison.png'
                    cv2.imwrite(str(comp_path), comparison)
                    print(f"    Comparison saved: {comp_path}")

                if fixed_out_path.exists():
                    shutil.rmtree(str(fixed_out_path))

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"  All done! Results saved to: {out_path}")
    if args.ensemble:
        print(f"  Ensemble results: {out_path / 'ensemble_results'}")
        print(f"  Individual seeds: {out_path / 'individual_seeds'}")
        print(f"  Comparisons:      {out_path / 'comparisons'}")
    print(f"{'='*65}\n")

if __name__ == '__main__':
    main()
