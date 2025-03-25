# ZeroStereo

This repository contains the source code for our paper: 

[ZeroStereo: Zero-shot Stereo Matching from Single Images](https://arxiv.org/pdf/2501.08654)

Used title: StereoGen: High-quality Stereo Image Generation from a Single Image

![ZeroStereo](ZeroStereo.png)

## Environment

```
conda create -n zerostereo python=3.12
conda activate zerostereo

pip install hydra-core tqdm opt_einsum numpy
pip install opencv-python scipy torch torchvision
pip install accelerate timm==0.5.4
```

## Pre-trained Models

You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1UufIY7I3NXiLVm7Hbj3_htJEgttx7-R6?usp=drive_link).

## Evaluation

To evaluate Zero-RAFT-Stereo, run:

```
accelerate launch evaluate_stereo.py
```

To evaluate Zero-IGEV-Stereo, modify config/evaluate_stereo.yaml or run:

```
accelerate launch evaluate_stereo.py model=igev_stereo checkpoint=checkpoint/igev_stereo/model_700.safetensors  
```

## Notification

Complete training and generation code will be released soon.
