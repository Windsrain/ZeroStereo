# 🏆 [ICCV 2025] ZeroStereo: Zero-shot Stereo Matching from Single Images 🏆

Xianqi Wang, Hao Yang, Gangwei Xu, Junda Cheng, Min Lin, Yong Deng, Jinliang Zang, Yurui Chen, Xin Yang

Huazhong University of Science and Technology, Autel Robotics, Optics Valley Laboratory

<a href="https://arxiv.org/abs/2501.08654"><img src="https://img.shields.io/badge/arXiv-2501.08654-b31b1b?logo=arxiv" alt='arxiv'></a>

Used title: StereoGen: High-quality Stereo Image Generation from a Single Image

![ZeroStereo](ZeroStereo.png)

## 🔄 Update

* **09/16/2025:** Update MfS35K to Hugging Face.
* **09/16/2025:** Update MfS35K for training.
* **07/29/2025:** Update the fine-tuning code for SDv2I.
* **07/14/2025:** Update the generation code to improve the quality of the right image edges.

## ⚙️ Environment

```
conda create -n zerostereo python=3.12
conda activate zerostereo

pip install tqdm numpy wandb opt_einsum hydra-core
pip install scipy torch torchvision diffusers transformers opencv-python matplotlib
pip install xformers accelerate scikit-image timm==0.5.4
```

## 📂 Required Data

**Data for fine-tuning**:

* [TartanAir](https://github.com/castacks/tartanair_tools)

* [CREStereo Dataset](https://github.com/megvii-research/CREStereo)

* [Falling Things](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)

* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

* [VKITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)

**Data for generation**: 

* [DiW](https://wfchen-umich.github.io/wfchen.github.io/depth-in-the-wild/)

* [COCOStuff](https://github.com/nightrome/cocostuff)

* [DIODE](https://diode-dataset.org/)

* [Mapillary](https://www.mapillary.com/dataset/vistas?pKey=1697734990430617)

* [ADE20K](https://ade20k.csail.mit.edu/)

The filepath format should be consistent with the filelist.

**Data for training**:

* [MfS35K on Baidu Netdisk](https://pan.baidu.com/s/1B0_gfJXoX8s8GUXROOjnew?pwd=sidv)

* [MfS35K on Hugging Face](https://huggingface.co/datasets/Windsrain/MfS35K)

We have uploaded MfS35K to Baidu Netdisk, and we will upload it to other websites for downloading.

**Data for evaluation**:

* [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)

* [ETH3D](https://www.eth3d.net/datasets)

## 🎁 Pre-Trained Model

| Model | Link |
| :-: | :-: |
| SDv2I | [Download 🤗](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/tree/main) |
| StereoGen | [Download 🤗](https://huggingface.co/Windsrain/ZeroStereo/tree/main/StereoGen) |
| Zero-RAFT-Stereo | [Download 🤗](https://huggingface.co/Windsrain/ZeroStereo/tree/main/Zero-RAFT-Stereo)|
| Zero-IGEV-Stereo | [Download 🤗](https://huggingface.co/Windsrain/ZeroStereo/tree/main/Zero-IGEV-Stereo)|

## 🚀 Fine-tuning

To fine-tune SDv2I, run:

```
accelerate launch train_stereogen.py
```

## 🛠️ Generation

To generate MfS35K, run:

```
accelerate launch generate_mono.py
accelerate launch generate_stereo.py
```

## 🚀 Training

To train Zero-RAFT-Stereo and Zero-IGEV-Stereo, run:

```
CUDA_VISIBLE_DEVICES='0,1' accelerate launch train_stereo.py
```

## 📊 Evaluation

To evaluate Zero-RAFT-Stereo, run:

```
accelerate launch evaluate_stereo.py
```

To evaluate Zero-IGEV-Stereo, modify config/evaluate_stereo.yaml or run:

```
accelerate launch evaluate_stereo.py model=igev_stereo checkpoint=checkpoint/igev_stereo/model_700.safetensors  
```

## 🎥 Demo

To save disparity, run:

```
accelerate launch save_disparity.py
```

## 🔔 Notification

## 🙏 Acknowledgement

This project is based on [MfS-Stereo](https://github.com/nianticlabs/stereo-from-mono), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Marigold](https://github.com/prs-eth/Marigold), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), and [IGEV-Stereo](https://github.com/gangweix/IGEV). We thank the original authors for their excellent works.