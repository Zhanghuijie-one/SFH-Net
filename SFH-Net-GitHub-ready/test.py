import os
import argparse
import torch
import torch.nn.functional as F

from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure

from model import LYT
from dataloader import create_dataloaders


# ================== GT Mean Correction ==================

def apply_gt_mean_correction(output, gt, eps=1e-8):
    """
    使用 GT 均值校正输出亮度。

    output: [B, C, H, W], range [0, 1]
    gt:     [B, C, H, W], range [0, 1]

    return:
        corrected_output: [B, C, H, W]
    """
    mean_output = output.mean(dim=(1, 2, 3), keepdim=True)
    mean_gt = gt.mean(dim=(1, 2, 3), keepdim=True)

    corrected_output = output * (mean_gt / (mean_output + eps))
    corrected_output = torch.clamp(corrected_output, 0, 1)

    return corrected_output


# ================== Metrics ==================

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    img1, img2: torch.Tensor, [B, C, H, W], range [0, 1]
    """
    mse = F.mse_loss(img1, img2, reduction="mean")

    if mse.item() == 0:
        return float("inf")

    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0):
    """
    img1, img2: torch.Tensor, [B, C, H, W], range [0, 1]
    """
    ssim_val = structural_similarity_index_measure(
        img1,
        img2,
        data_range=max_pixel_value,
    )

    return ssim_val.item()


# ================== Build Model ==================

def build_model(args, device):
    if args.model_type == "sfhnet":
        model = LYT(
            frb_mode=args.frb_mode,
            cdm_frb_mode=args.cdm_frb_mode,
            freq_mask_ratio=args.freq_mask_ratio,
        ).to(device)

    elif args.model_type == "cidnet":
        from model_cidnet import CIDNet
        model = CIDNet().to(device)

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return model


def load_checkpoint(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]

        print("Loaded checkpoint['model']")
        print(f"Checkpoint model_type: {checkpoint.get('model_type', 'N/A')}")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best PSNR: {checkpoint.get('best_psnr', 'N/A')}")
        print(f"Best SSIM: {checkpoint.get('best_ssim', 'N/A')}")
        print(f"frb_mode: {checkpoint.get('frb_mode', 'N/A')}")
        print(f"cdm_frb_mode: {checkpoint.get('cdm_frb_mode', 'N/A')}")
        print(f"freq_mask_ratio: {checkpoint.get('freq_mask_ratio', 'N/A')}")
        print(f"use_gan: {checkpoint.get('use_gan', 'N/A')}")
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

    return model


# ================== Validation ==================

def validate(model, dataloader, device, result_dir, save_images=True):
    model.eval()

    total_gtmean_psnr = 0.0
    total_gtmean_ssim = 0.0

    count = 0

    gtmean_result_dir = os.path.join(result_dir, "gt_mean")

    if save_images:
        os.makedirs(gtmean_result_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low = torch.clamp(low, 0, 1).to(device)
            high = torch.clamp(high, 0, 1).to(device)

            output = model(low)

            # 如果某些模型返回 tuple/list，只取第一个输出
            if isinstance(output, (tuple, list)):
                output = output[0]

            output = torch.clamp(output, 0, 1)

            output_gtmean = apply_gt_mean_correction(output, high)

            if save_images:
                save_image(
                    output_gtmean,
                    os.path.join(gtmean_result_dir, f"result_{idx}.png"),
                )

            # ================== GT-Mean Metrics ==================
            gtmean_psnr = calculate_psnr(output_gtmean, high)
            gtmean_ssim = calculate_ssim(output_gtmean, high)

            print(
                f"[{idx}] "
                f"GTMean | "
                f"PSNR: {gtmean_psnr:.4f} | "
                f"SSIM: {gtmean_ssim:.4f}"
            )

            print("-" * 120)

            total_gtmean_psnr += gtmean_psnr
            total_gtmean_ssim += gtmean_ssim

            count += 1

    avg_metrics = {
        "psnr": total_gtmean_psnr / count,
        "ssim": total_gtmean_ssim / count,
    }

    return avg_metrics


# ================== Args ==================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sfhnet", "cidnet"],
        help="Model type: sfhnet or cidnet."
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to checkpoint."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name used for saving results."
    )

    parser.add_argument(
        "--test_low",
        type=str,
        default="/home/zhanghuijie/work/LYT-Net-main-14-1-gan-gan-4/PyTorch/data/LOLv1/Test/input"
    )

    parser.add_argument(
        "--test_high",
        type=str,
        default="/home/zhanghuijie/work/LYT-Net-main-14-1-gan-gan-4/PyTorch/data/LOLv1/Test/target"
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
        help="Only used for sfhnet."
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
        help="Only used for sfhnet."
    )

    parser.add_argument(
        "--freq_mask_ratio",
        type=float,
        default=0.1,
        help="Only used for sfhnet."
    )

    parser.add_argument(
        "--result_root",
        type=str,
        default="results_fair"
    )

    parser.add_argument(
        "--no_save_images",
        action="store_true",
        help="Do not save output images."
    )

    return parser.parse_args()


# ================== Main ==================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_dir = os.path.join(args.result_root, args.dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 100)
    print("Evaluation Protocol")
    print("=" * 100)
    print(f"Model type      : {args.model_type}")
    print(f"Weights path    : {args.weights_path}")
    print(f"Dataset name    : {args.dataset_name}")
    print(f"Test low        : {args.test_low}")
    print(f"Test high       : {args.test_high}")
    print(f"Result dir      : {result_dir}")
    print(f"Device          : {device}")

    if args.model_type == "sfhnet":
        print(f"FRB mode        : {args.frb_mode}")
        print(f"CDM FRB mode    : {args.cdm_frb_mode}")
        print(f"Freq mask ratio : {args.freq_mask_ratio}")

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

    model = build_model(args, device)
    model = load_checkpoint(model, args.weights_path, device)

    avg_metrics = validate(
        model=model,
        dataloader=test_loader,
        device=device,
        result_dir=result_dir,
        save_images=(not args.no_save_images),
    )

    print("\n" + "=" * 100)
    print("Final Results: with GT-mean correction (RGB)")
    print("=" * 100)
    print(f"PSNR: {avg_metrics['psnr']:.6f}  higher is better")
    print(f"SSIM: {avg_metrics['ssim']:.6f}  higher is better")


if __name__ == "__main__":
    main()
