# 🚀 ZeroStereo (ICCV 2025) 🚀

This repository contains the source code for our paper: 

[ZeroStereo: Zero-shot Stereo Matching from Single Images](https://arxiv.org/pdf/2501.08654) <a href="https://arxiv.org/abs/2501.08643"><img src="https://img.shields.io/badge/arXiv-2402.11095-b31b1b?logo=arxiv" alt='arxiv'></a>

Used title: StereoGen: High-quality Stereo Image Generation from a Single Image

![ZeroStereo](ZeroStereo.png)

## ⚙️ Environment

```
conda create -n zerostereo python=3.12
conda activate zerostereo

pip install tqdm numpy wandb opt_einsum hydra-core
pip install scipy torch torchvision diffusers transformers opencv-python matplotlib
pip install xformers accelerate scikit-image timm==0.5.4
```

## ✏️ Required Data

* [DiW](https://wfchen-umich.github.io/wfchen.github.io/depth-in-the-wild/)

* [COCOStuff](https://github.com/nightrome/cocostuff)

* [DIODE](https://diode-dataset.org/)

* [Mapillary](https://www.mapillary.com/dataset/vistas?pKey=1697734990430617)

* [ADE20K](https://ade20k.csail.mit.edu/)

The filepath format should be consistent with the filelist.

## ✈️ Pre-trained Model

You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1UufIY7I3NXiLVm7Hbj3_htJEgttx7-R6?usp=drive_link).

## ✈️ Generation

To generate MfS35K, run:

```
accelerate launch generate_mono.py
accelerate launch generate_stereo.py
```

## ✈️ Evaluation

To evaluate Zero-RAFT-Stereo, run:

```
accelerate launch evaluate_stereo.py
```

To evaluate Zero-IGEV-Stereo, modify config/evaluate_stereo.yaml or run:

```
accelerate launch evaluate_stereo.py model=igev_stereo checkpoint=checkpoint/igev_stereo/model_700.safetensors  
```

## ✈️ Demo

To save disparity, run:

```
accelerate launch save_disparity.py
```

## ✈️ Notification

Complete fine-tuning code will be released soon. The generation code is an initial version, and we will release a final version for better results. We will upload our MfS35K for training directly.

## Acknowledgement

This project is based on [MfS-Stereo](https://github.com/nianticlabs/stereo-from-mono), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Marigold](https://github.com/prs-eth/Marigold), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), and [IGEV-Stereo](https://github.com/gangweix/IGEV). We thank the original authors for their excellent works.