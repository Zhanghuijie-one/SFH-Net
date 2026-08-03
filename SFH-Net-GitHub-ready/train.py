import os
import math
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import structural_similarity_index_measure

from model import LYT, ChromaDiscriminator
from losses import CombinedLoss
from dataloader import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser()

    # -------------------------
    # Model choice
    # -------------------------
    parser.add_argument(
        "--model_type",
        type=str,
        default="sfhnet",
        choices=["sfhnet", "cidnet"],
        help="Choose model: sfhnet or cidnet."
    )

    parser.add_argument(
        "--use_gan",
        action="store_true",
        help="Use GAN-assisted training. Only allowed for sfhnet."
    )

    # -------------------------
    # SFH-Net FRB settings
    # -------------------------
    parser.add_argument(
        "--frb_mode",
        type=str,
        default=None,
        choices=[
            "full",
            "old_collapse",
            "no_transform",
            "low_only",
            "high_only",
            "spatial_matched",
            "identity",
        ],
        help="FRB ablation mode. Only used for sfhnet."
    )

    parser.add_argument(
        "--cdm_frb_mode",
        type=str,
        default=None,
        choices=[
            "full",
            "old_collapse",
            "no_transform",
            "low_only",
            "high_only",
            "spatial_matched",
            "identity",
        ],
        help="CDM FRB mode. Only used for sfhnet."
    )

    parser.add_argument(
        "--freq_mask_ratio",
        type=float,
        default=None,
        help="Frequency mask ratio. Only used for sfhnet."
    )

    # -------------------------
    # Dataset paths
    # -------------------------
    parser.add_argument(
        "--train_low",
        type=str,
        default="/home/zhanghuijie/work/LYT-Net-main-14-1-gan-gan-4/PyTorch/data/LOLv1/Train/input"
    )

    parser.add_argument(
        "--train_high",
        type=str,
        default="/home/zhanghuijie/work/LYT-Net-main-14-1-gan-gan-4/PyTorch/data/LOLv1/Train/target"
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

    # -------------------------
    # Training hyperparameters
    # Keep these identical for CIDNet and SFH-Net w/o GAN.
    # -------------------------
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--warmup_epochs", type=int, default=40)

    parser.add_argument(
        "--lambda_gan_max",
        type=float,
        default=0.015,
        help="Maximum GAN loss weight. Only used when --use_gan is enabled."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for fair comparison."
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="ablation_model.pth",
        help="Checkpoint filename."
    )

    return parser.parse_args()


def set_random_seed(seed):
    """
    Make the training protocol as reproducible as possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(dim=1)
        img2_gray = img2.mean(dim=1)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()

        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-8)), 0, 1)

    mse = F.mse_loss(img1, img2, reduction="mean")
    if mse.item() == 0:
        return float("inf")

    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()

        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-8)), 0, 1)

    ssim_val = structural_similarity_index_measure(
        img1,
        img2,
        data_range=max_pixel_value
    )
    return ssim_val.item()


def validate(model, dataloader, device):
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for low, high in dataloader:
            low = torch.clamp(low, 0, 1).to(device)
            high = torch.clamp(high, 0, 1).to(device)

            output = model(low)
            output = torch.clamp(output, 0, 1)

            total_psnr += calculate_psnr(output, high, gt_mean=True)
            total_ssim += calculate_ssim(output, high, gt_mean=True)

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)

    return avg_psnr, avg_ssim


def compute_lambda_gan(epoch, warmup_epochs, num_epochs, max_lambda=0.015):
    if epoch < warmup_epochs:
        return 0.0

    progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
    return max_lambda * (0.5 - 0.5 * math.cos(math.pi * progress))


def build_model(args, device):
    """
    Build either SFH-Net or original HVI-CIDNet.

    For fair comparison:
        --model_type cidnet  -> original HVI-CIDNet
        --model_type sfhnet  -> your SFH-Net
    """
    if args.model_type == "sfhnet":
        model_kwargs = {}

        if args.frb_mode is not None:
            model_kwargs["frb_mode"] = args.frb_mode

        if args.cdm_frb_mode is not None:
            model_kwargs["cdm_frb_mode"] = args.cdm_frb_mode

        if args.freq_mask_ratio is not None:
            model_kwargs["freq_mask_ratio"] = args.freq_mask_ratio

        model = LYT(**model_kwargs).to(device)

    elif args.model_type == "cidnet":
        # 请把 HVI-CIDNet 原模型保存为 model_cidnet.py
        # 里面 class 名称保持 CIDNet
        from model_cidnet import CIDNet

        model = CIDNet().to(device)

        # 双保险：如果你忘了在 CIDNet 里写 self.hvi_converter = self.trans，
        # 这里也会自动补上。
        if not hasattr(model, "hvi_converter"):
            if hasattr(model, "trans"):
                model.hvi_converter = model.trans
            else:
                raise AttributeError(
                    "CIDNet must have either model.hvi_converter or model.trans for HVI loss."
                )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return model


def save_checkpoint(
    path,
    model,
    args,
    epoch,
    best_psnr,
    best_ssim,
):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    checkpoint = {
        "model": model.state_dict(),
        "model_type": args.model_type,
        "use_gan": args.use_gan,
        "epoch": epoch,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "seed": args.seed,

        # Training protocol
        "train_low": args.train_low,
        "train_high": args.train_high,
        "test_low": args.test_low,
        "test_high": args.test_high,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "warmup_epochs": args.warmup_epochs,
        "lambda_gan_max": args.lambda_gan_max,

        # SFH-Net settings
        "frb_mode": args.frb_mode,
        "cdm_frb_mode": args.cdm_frb_mode,
        "freq_mask_ratio": args.freq_mask_ratio,
    }

    torch.save(checkpoint, path)


def main():
    args = parse_args()

    # -------------------------
    # Strict control
    # -------------------------
    set_random_seed(args.seed)

    if args.model_type == "cidnet" and args.use_gan:
        raise ValueError(
            "For the fair HVI-CIDNet comparison, do not use GAN. "
            "Please run CIDNet without --use_gan."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("Training protocol")
    print("=" * 100)
    print(f"Model type       : {args.model_type}")
    print(f"Use GAN          : {args.use_gan}")
    print(f"LR               : {args.learning_rate}")
    print(f"Epochs           : {args.num_epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Crop size        : {args.crop_size}")
    print(f"Warmup epochs    : {args.warmup_epochs}")
    print(f"Seed             : {args.seed}")
    print(f"Device           : {device}")
    print(f"Train low        : {args.train_low}")
    print(f"Train high       : {args.train_high}")
    print(f"Test low         : {args.test_low}")
    print(f"Test high        : {args.test_high}")
    print(f"Save name        : {args.save_name}")

    if args.model_type == "sfhnet":
        print("-" * 100)
        print(f"FRB mode         : {args.frb_mode if args.frb_mode is not None else 'LYT default'}")
        print(f"CDM FRB mode     : {args.cdm_frb_mode if args.cdm_frb_mode is not None else 'LYT default'}")
        print(f"Freq mask ratio  : {args.freq_mask_ratio if args.freq_mask_ratio is not None else 'LYT default'}")
    print("=" * 100)

    # -------------------------
    # Data loaders
    # -------------------------
    train_loader, test_loader = create_dataloaders(
        args.train_low,
        args.train_high,
        args.test_low,
        args.test_high,
        crop_size=args.crop_size,
        batch_size=args.batch_size
    )

    print(f"Train steps: {len(train_loader)}")
    print(f"Val steps  : {len(test_loader)}")

    # -------------------------
    # Model
    # -------------------------
    model = build_model(args, device)

    # Make sure both SFH-Net and CIDNet have hvi_converter
    if not hasattr(model, "hvi_converter"):
        if hasattr(model, "trans"):
            model.hvi_converter = model.trans
        else:
            raise AttributeError(
                "The model must provide hvi_converter or trans for HVI loss computation."
            )

    # -------------------------
    # Loss and optimizers
    # -------------------------
    criterion = CombinedLoss(device)

    optimizer_G = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=args.num_epochs)

    if args.use_gan:
        discriminator = ChromaDiscriminator().to(device)
        gan_criterion = nn.BCEWithLogitsLoss()
        optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
        scheduler_D = CosineAnnealingLR(optimizer_D, T_max=args.num_epochs)
    else:
        discriminator = None
        gan_criterion = None
        optimizer_D = None
        scheduler_D = None

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(device.type == "cuda")
    )

    best_psnr = 0.0
    best_ssim = 0.0

    print("Training started.")

    for epoch in range(args.num_epochs):
        model.train()

        if discriminator is not None:
            discriminator.train()

        epoch_loss = 0.0
        lambda_gan = compute_lambda_gan(
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            num_epochs=args.num_epochs,
            max_lambda=args.lambda_gan_max
        )

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs_01 = torch.clamp(inputs, 0, 1)
            targets_01 = torch.clamp(targets, 0, 1)

            # ==========================================================
            # Reconstruction-only training
            #
            # This branch is used by:
            #   1) HVI-CIDNet retrained baseline
            #   2) SFH-Net w/o GAN
            #   3) warm-up stage of Full SFH-Net
            #
            # For fair comparison between HVI-CIDNet and SFH-Net w/o GAN,
            # both models run only this branch for all epochs.
            # ==========================================================
            if (not args.use_gan) or (epoch < args.warmup_epochs):
                optimizer_G.zero_grad(set_to_none=True)

                outputs = model(inputs_01)

                hvi_pred = model.hvi_converter.HVIT(outputs)
                hvi_true = model.hvi_converter.HVIT(targets_01)

                total_loss_G = criterion(
                    y_true=targets_01,
                    y_pred=outputs,
                    h_true=hvi_true[:, 0:1],
                    v_true=hvi_true[:, 1:2],
                    h_pred=hvi_pred[:, 0:1],
                    v_pred=hvi_pred[:, 1:2],
                    isgan=False
                )

                scaler.scale(total_loss_G).backward()
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer_G)
                scaler.update()

                epoch_loss += float(total_loss_G.detach().cpu().item())

            # ==========================================================
            # GAN-assisted training
            #
            # This branch is only used by:
            #   Full SFH-Net with --use_gan
            #
            # CIDNet is forbidden to enter this branch.
            # ==========================================================
            else:
                # -------------------------
                # Train Discriminator
                # -------------------------
                optimizer_D.zero_grad(set_to_none=True)

                with torch.no_grad():
                    outputs_for_D = model(inputs_01)

                    hvi_pred_D = model.hvi_converter.HVIT(outputs_for_D)
                    h_pred_D = hvi_pred_D[:, 0:1]
                    v_pred_D = hvi_pred_D[:, 1:2]

                    hvi_true = model.hvi_converter.HVIT(targets_01)
                    h_true_no_grad = hvi_true[:, 0:1]
                    v_true_no_grad = hvi_true[:, 1:2]

                real_in = torch.cat([h_true_no_grad, v_true_no_grad], dim=1)
                fake_in_detach = torch.cat([h_pred_D, v_pred_D], dim=1)

                disc_real = discriminator(real_in, real_in)
                disc_fake = discriminator(real_in, fake_in_detach)

                loss_D = 0.5 * (
                    gan_criterion(
                        disc_real,
                        torch.ones_like(disc_real, device=device)
                    )
                    +
                    gan_criterion(
                        disc_fake,
                        torch.zeros_like(disc_fake, device=device)
                    )
                )

                loss_D.backward()
                optimizer_D.step()

                # -------------------------
                # Train Generator
                # -------------------------
                optimizer_G.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=False):
                    outputs = model(inputs_01)

                    hvi_pred = model.hvi_converter.HVIT(outputs)
                    h_pred = hvi_pred[:, 0:1]
                    v_pred = hvi_pred[:, 1:2]

                    main_loss = criterion(
                        y_true=targets_01,
                        y_pred=outputs,
                        h_true=h_true_no_grad,
                        v_true=v_true_no_grad,
                        h_pred=h_pred,
                        v_pred=v_pred,
                        isgan=True
                    )

                    fake_in_for_G = torch.cat([h_pred, v_pred], dim=1)
                    disc_fake_for_G = discriminator(real_in, fake_in_for_G)

                    gan_loss_G = gan_criterion(
                        disc_fake_for_G,
                        torch.ones_like(disc_fake_for_G, device=device)
                    )

                    total_loss_G = main_loss + lambda_gan * gan_loss_G

                scaler.scale(total_loss_G).backward()
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer_G)
                scaler.update()

                epoch_loss += float(total_loss_G.detach().cpu().item())

        # -------------------------
        # Validation
        # -------------------------
        avg_psnr, avg_ssim = validate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"TrainLoss: {epoch_loss:.6f} | "
            f"Val PSNR: {avg_psnr:.4f} | "
            f"Val SSIM: {avg_ssim:.4f} | "
            f"lambda_gan: {lambda_gan:.6f}"
        )

        scheduler_G.step()

        if args.use_gan and epoch >= args.warmup_epochs:
            scheduler_D.step()

        # -------------------------
        # Save best by PSNR
        # -------------------------
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr

            save_checkpoint(
                path=args.save_name,
                model=model,
                args=args,
                epoch=epoch + 1,
                best_psnr=best_psnr,
                best_ssim=best_ssim
            )

            print(f"Saved best model by PSNR: {best_psnr:.4f}")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            print(f"New best SSIM reached: {best_ssim:.4f}")

    print("=" * 100)
    print("Training finished.")
    print(f"Best PSNR: {best_psnr:.4f}")
    print(f"Best SSIM: {best_ssim:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
