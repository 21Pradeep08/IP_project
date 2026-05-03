import os
import cv2
import glob
import torch
import numpy as np
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from utils import util_image
import lpips

def main():
    gt_dir = 'testdata/cropped_faces'
    pred_dir = 'results/restored_faces'
    
    # Check if directories exist
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Error: Make sure Both {gt_dir} and {pred_dir} exist")
        return
        
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    
    if len(gt_paths) == 0:
        print(f"No images found in {gt_dir}")
        return
        
    psnr_list, ssim_list, lpips_list = [], [], []
    
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    lpips_fn.eval()

    for gt_path in gt_paths:
        img_name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, img_name)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction not found for {img_name}, skipping.")
            continue
            
        # Read images in BGR format (cv2 default)
        # Using basicsr calculate_psnr and calculate_ssim expects HWC [0, 255] RGB for typical setting,
        # but let's check basicsr implementation
        # calculate_psnr usually works on RGB or Grayscale, but image channels must match.
        
        # Read as RGB numpy arrays [0, 255]
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_COLOR)
        
        if gt_img is None or pred_img is None:
            continue
            
        # Convert BGR to RGB
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        
        # Ensure same size
        h, w = gt_img.shape[:2]
        pred_img_rs = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        psnr_val = calculate_psnr(pred_img_rs, gt_img, crop_border=0, input_order='HWC')
        ssim_val = calculate_ssim(pred_img_rs, gt_img, crop_border=0, input_order='HWC')
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        # LPIPS expects input in [-1, 1], CHW
        gt_tensor = util_image.imread(gt_path, chn='rgb', dtype='float32') # HWC, RGB [0,1]
        pred_tensor = util_image.imread(pred_path, chn='rgb', dtype='float32')
        
        # resize
        pred_tensor = cv2.resize(pred_tensor, (gt_tensor.shape[1], gt_tensor.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
        gt_tensor = torch.from_numpy(gt_tensor.transpose((2, 0, 1))).unsqueeze(0).cuda()
        pred_tensor = torch.from_numpy(pred_tensor.transpose((2, 0, 1))).unsqueeze(0).cuda()
        
        # Normalize to [-1, 1]
        gt_tensor = gt_tensor * 2.0 - 1.0
        pred_tensor = pred_tensor * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_val = lpips_fn(pred_tensor, gt_tensor).item()
        
        lpips_list.append(lpips_val)
        print(f"{img_name}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

    if len(psnr_list) > 0:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_lpips = sum(lpips_list) / len(lpips_list)
        print("="*40)
        print(f"Average PSNR:  {avg_psnr:.4f}")
        print(f"Average SSIM:  {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        print("="*40)
    else:
        print("No paired images were evaluated.")

if __name__ == '__main__':
    main()
