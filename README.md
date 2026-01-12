# SFH-Net:Enhancing Low-Light Images through HVI Color Space with Spatial-Frequency Domain Integration

This code is the official implementation of the paper submitted to The Visual Computer: "Enhancing Low-Light Images through HVI Color Space with Spatial-Frequency Domain Integration"

## Datasets

Please download the datasets and place them in the `./data` folder.

- **Link:** [Baidu Cloud](https://pan.baidu.com/s/1iu-BCr9XarNMJTMKY1keZw)
- **Access Code:** `dat3`

## Pre-trained Models

You can download the pre-trained models from Baidu Netdisk:
- **Link:** [Baidu Cloud](https://pan.baidu.com/s/1jzKZ5QQc7A4iHOvNj6OIRQ)
- **Access Code:** `pth1`

## Visual Comparisons

### LOL-v1 Dataset
<img width="700" height="313" alt="image" src="https://github.com/user-attachments/assets/801a8d4a-4ace-4edb-acb4-18574af7588e" />


## Installation

Please run the following commands to create the environment and install dependencies:

### 1. Create a conda environment
```bash
conda create -n py38 python=3.8
```
### 2. Activate the environment
```bash
conda activate py38
```
### 3. Install Dependencies
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm torchmetrics pytorch_msssim
```

## Run
### Test
```bash
python test.py
```



