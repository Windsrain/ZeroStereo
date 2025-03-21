# ZeroStereo

This repository contains the source code for our paper: 

[ZeroStereo: Zero-shot Stereo Matching from Single Images](https://arxiv.org/pdf/2501.08654)

Used title: StereoGen: High-quality Stereo Image Generation from a Single Image

![ZeroStereo](ZeroStereo.png)

## Environment

```
conda create -n zerostereo python=3.12
conda activate zerostereo

pip install accelerate hydra-core numpy opencv-python opt_einsum pillow scipy torch torchvision tqdm
```

## Pre-trained Models

You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1UufIY7I3NXiLVm7Hbj3_htJEgttx7-R6?usp=drive_link).

## Evaluation

```
acclerate launch evaluate_stereo.py
```

## Notification

Complete training and generation code will be released soon.
