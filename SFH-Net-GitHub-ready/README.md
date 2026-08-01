# SFH-Net

Official PyTorch code for **Enhancing Low-Light Images through HVI Color Space with Spatial-Frequency Domain Integration**.

This release contains the training entry point, evaluation script, model and loss implementations, the controlled HVI-CIDNet comparison dependency, and a pretrained LOL-v1 checkpoint. The Python source files are copied unchanged from the experiment directory used for the revised manuscript.

## Environment

The reported environment is Python 3.9.21, PyTorch 2.5.1, and CUDA 11.8.

```bash
conda env create -f environment.yml
conda activate sfhnet
```

The first training run may download the ImageNet-pretrained VGG-19 weights used by the loss function.

## Dataset layout

Download LOL-v1 and arrange the paired images as follows. Corresponding input and target images must have matching filenames after lexicographic sorting.

```text
data/
`-- LOLv1/
    |-- Train/
    |   |-- input/
    |   `-- target/
    `-- Test/
        |-- input/
        `-- target/
```

The same four-directory layout can be used for other paired datasets by changing the command-line paths.

## Train

The following command reproduces the final SFH-Net training configuration on LOL-v1 with seed 1234:

```bash
python train.py \
  --model_type sfhnet \
  --use_gan \
  --frb_mode full \
  --cdm_frb_mode full \
  --freq_mask_ratio 0.1 \
  --train_low data/LOLv1/Train/input \
  --train_high data/LOLv1/Train/target \
  --test_low data/LOLv1/Test/input \
  --test_high data/LOLv1/Test/target \
  --learning_rate 0.0002 \
  --num_epochs 1000 \
  --batch_size 2 \
  --crop_size 256 \
  --warmup_epochs 40 \
  --lambda_gan_max 0.015 \
  --seed 1234 \
  --save_name weights/sfhnet_lolv1_seed1234_retrained.pth
```

For the additional independent runs, use the same command with `--seed 2024` or `--seed 3500` and change `--save_name` accordingly.

## Evaluate

Evaluate the included checkpoint using GT-mean brightness correction and report RGB PSNR and SSIM:

```bash
python test.py \
  --model_type sfhnet \
  --weights_path weights/sfhnet_lolv1_seed1234.pth \
  --dataset_name LOLv1_seed1234 \
  --test_low data/LOLv1/Test/input \
  --test_high data/LOLv1/Test/target \
  --frb_mode full \
  --cdm_frb_mode full \
  --freq_mask_ratio 0.1 \
  --result_root results
```

Add `--no_save_images` to report PSNR and SSIM without writing the GT-mean-corrected restored images.

## Complexity

```bash
python macs.py
```

The complexity script profiles the generator with a `1 x 3 x 256 x 256` input tensor using `torchprofile.profile_macs`.

## Included files

```text
SFH-Net-GitHub-ready/
|-- train.py                 # configurable training entry point
|-- test.py                  # GT-mean-corrected RGB PSNR/SSIM evaluation
|-- model.py                 # SFH-Net generator and discriminator
|-- losses.py                # reconstruction, HVI, frequency, and chroma losses
|-- dataloader.py            # paired-image loader
|-- macs.py                  # complexity profiling
|-- model_cidnet.py          # dependency for the controlled CIDNet option
|-- net/                     # CIDNet support modules
|-- weights/
|   `-- sfhnet_lolv1_seed1234.pth
|-- environment.yml
|-- .gitignore
`-- LICENSE
```

Training logs, generated images, caches, intermediate checkpoints, and obsolete implementations are intentionally excluded.
