import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure

from model import LYT
from dataloader import create_dataloaders

import lpips
from skimage.color import rgb2lab, deltaE_ciede2000


# ================== GT Mean Correction ==================

def apply_gt_mean_correction(output, gt, eps=1e-8):
    """
    Apply GT-mean brightness correction.

    output: [B, C, H, W], range [0, 1]
    gt:     [B, C, H, W], range [0, 1]
    """
    mean_output = output.mean(dim=(1, 2, 3), keepdim=True)
    mean_gt = gt.mean(dim=(1, 2, 3), keepdim=True)

    corrected_output = output * (mean_gt / (mean_output + eps))
    corrected_output = torch.clamp(corrected_output, 0, 1)

    return corrected_output


# ================== Metrics ==================

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    img1, img2: [B, C, H, W], range [0, 1]
    """
    mse = F.mse_loss(img1, img2, reduction="mean")

    if mse.item() == 0:
        return float("inf")

    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0):
    """
    img1, img2: [B, C, H, W], range [0, 1]
    """
    ssim_val = structural_similarity_index_measure(
        img1,
        img2,
        data_range=max_pixel_value,
    )
    return ssim_val.item()


def calculate_lpips(img1, img2, lpips_model):
    """
    img1, img2: [B, C, H, W], range [0, 1]
    LPIPS expects input range [-1, 1].
    """
    img1_norm = img1 * 2.0 - 1.0
    img2_norm = img2 * 2.0 - 1.0

    lpips_val = lpips_model(img1_norm, img2_norm)
    return lpips_val.mean().item()


def calculate_ciede2000(tensor_output, tensor_gt):
    """
    Standard CIEDE2000 using skimage.rgb2lab.

    tensor_output, tensor_gt: [B, C, H, W], range [0, 1]
    """
    tensor_output = torch.clamp(tensor_output, 0, 1)
    tensor_gt = torch.clamp(tensor_gt, 0, 1)

    batch_delta_e = []

    for i in range(tensor_output.size(0)):
        output = tensor_output[i].permute(1, 2, 0).cpu().detach().numpy()
        gt = tensor_gt[i].permute(1, 2, 0).cpu().detach().numpy()

        output_lab = rgb2lab(output)
        gt_lab = rgb2lab(gt)

        delta_e = deltaE_ciede2000(gt_lab, output_lab).mean()
        batch_delta_e.append(delta_e)

    return float(np.mean(batch_delta_e))


# ================== Load Model ==================

def load_model(args, device):
    """
    Load SFH-Net / LYT with the specified FRB mode.

    Important:
    If your checkpoint is a raw state_dict, the script cannot know the FRB mode
    automatically. Therefore, --frb_mode and --cdm_frb_mode must match the training setting.
    """
    model = LYT(
        frb_mode=args.frb_mode,
        cdm_frb_mode=args.cdm_frb_mode,
        freq_mask_ratio=args.freq_mask_ratio,
    ).to(device)

    checkpoint = torch.load(args.weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]

        ckpt_frb_mode = checkpoint.get("frb_mode", None)
        ckpt_cdm_frb_mode = checkpoint.get("cdm_frb_mode", None)
        ckpt_freq_mask_ratio = checkpoint.get("freq_mask_ratio", None)
        print("Loaded checkpoint['model']")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Checkpoint frb_mode: {ckpt_frb_mode}")
        print(f"Checkpoint cdm_frb_mode: {ckpt_cdm_frb_mode}")
        print(f"Checkpoint freq_mask_ratio: {ckpt_freq_mask_ratio}")

        if ckpt_frb_mode is not None and ckpt_frb_mode != args.frb_mode:
            raise ValueError(
                f"FRB mode mismatch: "
                f"checkpoint={ckpt_frb_mode}, runtime={args.frb_mode}"
            )

        if (
            ckpt_cdm_frb_mode is not None
            and ckpt_cdm_frb_mode != args.cdm_frb_mode
        ):
            raise ValueError(
                f"CDM FRB mode mismatch: "
                f"checkpoint={ckpt_cdm_frb_mode}, "
                f"runtime={args.cdm_frb_mode}"
            )

        if (
            ckpt_freq_mask_ratio is not None
            and abs(ckpt_freq_mask_ratio - args.freq_mask_ratio) > 1e-8
        ):
            raise ValueError(
                f"Frequency ratio mismatch: "
                f"checkpoint={ckpt_freq_mask_ratio}, "
                f"runtime={args.freq_mask_ratio}"
            )

    else:
        state_dict = checkpoint
        print("Loaded raw state_dict")

    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]

        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]

        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    print("=" * 100)
    print(f"Model loaded from: {args.weights_path}")
    print(f"Runtime frb_mode: {args.frb_mode}")
    print(f"Runtime cdm_frb_mode: {args.cdm_frb_mode}")
    print(f"Runtime freq_mask_ratio: {args.freq_mask_ratio}")
    print("=" * 100)

    return model


# ================== Validation ==================

# ================== Validation ==================

def validate(
    model,
    dataloader,
    device,
    result_dir,
    lpips_model,
    save_images=True,
):
    """
    Evaluate only after GT-mean brightness correction.
    """
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_ciede = 0.0

    count = 0

    if save_images:
        os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low = torch.clamp(low, 0, 1).to(device)
            high = torch.clamp(high, 0, 1).to(device)

            output = model(low)

            if isinstance(output, (tuple, list)):
                output = output[0]

            output = torch.clamp(output, 0, 1)

            # Apply GT-mean brightness correction
            output_gtmean = apply_gt_mean_correction(output, high)

            if save_images:
                save_image(
                    output_gtmean,
                    os.path.join(result_dir, f"result_{idx}.png"),
                )

            # Compute metrics only on GT-mean corrected output
            psnr = calculate_psnr(output_gtmean, high)
            ssim = calculate_ssim(output_gtmean, high)
            lpips_value = calculate_lpips(
                output_gtmean,
                high,
                lpips_model,
            )
            ciede = calculate_ciede2000(output_gtmean, high)

            print(
                f"[{idx}] GT-mean | "
                f"PSNR: {psnr:.4f} | "
                f"SSIM: {ssim:.4f} | "
                f"LPIPS: {lpips_value:.4f} | "
                f"CIEDE2000: {ciede:.4f}"
            )
            print("-" * 120)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_value
            total_ciede += ciede

            count += 1

    if count == 0:
        raise RuntimeError(
            "No test samples were found. "
            "Please check --test_low and --test_high."
        )

    avg_metrics = {
        "psnr": total_psnr / count,
        "ssim": total_ssim / count,
        "lpips": total_lpips / count,
        "ciede": total_ciede / count,
    }

    return avg_metrics

# ================== Main ==================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_low",
        type=str,
        default="/home/zhanghuijie/work/LYT-Net-main-14-1-gan-gan-4/PyTorch/data/LOLv1/Test/input",
    )

    parser.add_argument(
        "--test_high",
        type=str,
        default="/home/zhanghuijie/work/LYT-Net-main-14-1-gan-gan-4/PyTorch/data/LOLv1/Test/target",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--frb_mode",
        type=str,
        default="full",
        choices=[
            "full",
            "old_collapse",
            "no_transform",
            "low_only",
            "high_only",
            "spatial_matched",
            "identity",
        ],
    )

    parser.add_argument(
        "--cdm_frb_mode",
        type=str,
        default="full",
        choices=[
            "full",
            "old_collapse",
            "no_transform",
            "low_only",
            "high_only",
            "spatial_matched",
            "identity",
        ],
    )

    parser.add_argument(
        "--freq_mask_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--result_root",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--no_save_images",
        action="store_true",
        help="Do not save output images.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_dir = os.path.join(
    args.result_root,
    args.dataset_name,
    "gt_mean",
)
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 100)
    print("Evaluation setting")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Test low: {args.test_low}")
    print(f"Test high: {args.test_high}")
    print(f"Weights path: {args.weights_path}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Result dir: {result_dir}")
    print(f"FRB mode: {args.frb_mode}")
    print(f"CDM FRB mode: {args.cdm_frb_mode}")
    print(f"Frequency mask ratio: {args.freq_mask_ratio}")
    print("=" * 100)

    _, test_loader = create_dataloaders(
        None,
        None,
        args.test_low,
        args.test_high,
        crop_size=None,
        batch_size=1,
    )

    print(f"Test loader: {len(test_loader)}")

    model = load_model(args, device)

    lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device)
    lpips_model.eval()

    avg_metrics = validate(
        model=model,
        dataloader=test_loader,
        device=device,
        result_dir=result_dir,
        lpips_model=lpips_model,
        save_images=(not args.no_save_images),
    )

    print("\n===== Final Results: with GT-mean correction =====")
    print(
        f"PSNR:      {avg_metrics['psnr']:.6f}  "
        f"higher is better"
    )
    print(
        f"SSIM:      {avg_metrics['ssim']:.6f}  "
        f"higher is better"
    )
    print(
        f"LPIPS:     {avg_metrics['lpips']:.6f}  "
        f"lower is better"
    )
    print(
        f"CIEDE2000: {avg_metrics['ciede']:.6f}  "
        f"lower is better"
    )


if __name__ == "__main__":
    main()
