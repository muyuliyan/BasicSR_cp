<p align="center">
  <img src="assets/basicsr_xpixel_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/xinntao/BasicSR/blob/master/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/basicsr)](https://pypi.org/project/basicsr/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/xinntao/BasicSR.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xinntao/BasicSR/context:python)
[![python lint](https://github.com/xinntao/BasicSR/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/BasicSR/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/publish-pip.yml)
[![gitee mirror](https://github.com/xinntao/BasicSR/actions/workflows/gitee-mirror.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/gitee-mirror.yml)

</div>

<div align="center">

⚡[**MRI Quick Start**](MRI_QuickStart.md) **|** 🔧[**Installation**](docs/INSTALL.md) **|** 💻[**Training Commands**](docs/TrainTest.md) **|** 🏥[**MRI Dataset Prep**](docs/MRI_DatasetPreparation.md) **|** 🏰[**Model Zoo**](docs/ModelZoo.md)

📕[**中文解读文档**](https://github.com/XPixelGroup/BasicSR-docs) **|** 📊 [**Plot scripts**](scripts/plot) **|** 📝[Introduction](docs/introduction.md) **|** <a href="https://github.com/XPixelGroup/BasicSR/tree/master/colab"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> **|** ⏳[TODO List](https://github.com/xinntao/BasicSR/projects) **|** ❓[FAQ](docs/FAQ.md)
</div>

🚀 We add [BasicSR-Examples](https://github.com/xinntao/BasicSR-examples), which provides guidance and templates of using BasicSR as a python package. 🚀 <br>
📢 **技术交流QQ群**：**320960100** &emsp; 入群答案：**互帮互助共同进步** <br>
🧭 [入群二维码](#-contact) (QQ、微信) &emsp;&emsp; [入群指南 (腾讯文档)](https://docs.qq.com/doc/DYXBSUmxOT0xBZ05u) <br>

---

BasicSR (**Basic** **S**uper **R**estoration) is an open-source **medical image restoration** toolbox based on PyTorch, specifically optimized for **MRI (Magnetic Resonance Imaging)** super-resolution.<br>
BasicSR (**Basic** **S**uper **R**estoration) 是一个基于 PyTorch 的开源 **医学图像复原工具箱**, 专门针对 **MRI（磁共振成像）超分辨率** 进行优化。

## 🏥 MRI-Specific Features

- **📄 Medical Format Support**: NIfTI (.nii/.nii.gz), NumPy (.npy), MATLAB (.mat)
- **🧠 OASIS Dataset**: Optimized for brain MRI super-resolution
- **💖 MM-WHS Dataset**: Optimized for cardiac MRI super-resolution  
- **🔬 Robust Normalization**: Percentile-based clipping + [0,1] scaling
- **📐 3D to 2D Conversion**: Automatic slice extraction from 3D volumes
- **⚡ Memory Efficient**: Single-channel processing vs multi-spectral imaging

🚩 **MRI Updates & Features**

- ✅ **2024**: Complete transformation to **MRI medical image super-resolution**
- ✅ **OASIS Dataset Support**: Brain MRI super-resolution with optimized normalization
- ✅ **MM-WHS Dataset Support**: Cardiac MRI super-resolution  
- ✅ **Medical Format Support**: NIfTI, NumPy, MATLAB formats
- ✅ **3D Volume Handling**: Automatic 3D→2D slice extraction
- ✅ **Single-Channel Optimization**: Efficient grayscale MRI processing
- ✅ **Robust Normalization**: Percentile-based clipping for medical images

---

## 🚀 MRI Quick Start

### 1. **Installation & Dependencies**
```bash
git clone https://github.com/muyuliyan/BasicSR_cp.git
cd BasicSR_cp
pip install -r requirements.txt

# For medical imaging support
pip install nibabel  # NIfTI format support
```

### 2. **Prepare MRI Dataset**

#### OASIS Brain MRI:
```bash
mkdir -p datasets/OASIS/{train,val,test}/{HR,LR}
# Copy your .nii brain MRI files to HR folders
python scripts/mri_data_preparation.py --input datasets/OASIS/train/HR --output datasets/OASIS/train/LR --scale 4 --extract-2d
```

#### MM-WHS Cardiac MRI:
```bash
mkdir -p datasets/MM-WHS/{train,val,test}/{HR,LR}  
# Copy your cardiac MRI files to HR folders
python scripts/mri_data_preparation.py --input datasets/MM-WHS/train/HR --output datasets/MM-WHS/train/LR --scale 4 --extract-2d
```

### 3. **Train MRI Super-Resolution Model**
```bash
# Train OASIS brain MRI model
PYTHONPATH="./:${PYTHONPATH}" python basicsr/train.py -opt options/train/MRI/train_OASIS_SRResNet_x4.yml --auto_resume

# Train MM-WHS cardiac MRI model  
PYTHONPATH="./:${PYTHONPATH}" python basicsr/train.py -opt options/train/MRI/train_MMWHS_SRResNet_x4.yml --auto_resume
```

### 4. **Test & Evaluate**
```bash
# Test installation
python test_mri_installation.py

# Test OASIS model
PYTHONPATH="./:${PYTHONPATH}" python basicsr/test.py -opt options/test/MRI/test_OASIS_SRResNet_x4.yml

# Test MM-WHS model
PYTHONPATH="./:${PYTHONPATH}" python basicsr/test.py -opt options/test/MRI/test_MMWHS_SRResNet_x4.yml
```

📖 **Full Documentation**: [MRI Quick Start Guide](MRI_QuickStart.md) | [MRI Dataset Preparation](docs/MRI_DatasetPreparation.md)

---

If BasicSR helps your research or work, please help to ⭐ this repo or recommend it to your friends. Thanks😊 <br>
Other recommended projects:<br>
▶️ [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration<br>
▶️ [GFPGAN](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration <br>
▶️ [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions.<br>
▶️ [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison. <br>
▶️ [HandyFigure](https://github.com/xinntao/HandyFigure): Open source of paper figures <br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyCrawler](https://github.com/xinntao/HandyCrawler), [HandyWriting](https://github.com/xinntao/HandyWriting))</sub>

---

## ⚡ HOWTOs

We provide simple pipelines to train/test/inference models for a quick start.
These pipelines/commands cannot cover all the cases and more details are in the following sections.

| GAN                  |                                                |                                                        |          |                                                |                                                        |
| :------------------- | :--------------------------------------------: | :----------------------------------------------------: | :------- | :--------------------------------------------: | :----------------------------------------------------: |
| StyleGAN2            | [Train](docs/HOWTOs.md#How-to-train-StyleGAN2) | [Inference](docs/HOWTOs.md#How-to-inference-StyleGAN2) |          |                                                |                                                        |
| **Face Restoration** |                                                |                                                        |          |                                                |                                                        |
| DFDNet               |                       -                        |  [Inference](docs/HOWTOs.md#How-to-inference-DFDNet)   |          |                                                |                                                        |
| **Super Resolution** |                                                |                                                        |          |                                                |                                                        |
| ESRGAN               |                     *TODO*                     |                         *TODO*                         | SRGAN    |                     *TODO*                     |                         *TODO*                         |
| EDSR                 |                     *TODO*                     |                         *TODO*                         | SRResNet |                     *TODO*                     |                         *TODO*                         |
| RCAN                 |                     *TODO*                     |                         *TODO*                         | SwinIR   | [Train](docs/HOWTOs.md#how-to-train-swinir-sr) | [Inference](docs/HOWTOs.md#how-to-inference-swinir-sr) |
| EDVR                 |                     *TODO*                     |                         *TODO*                         | DUF      |                       -                        |                         *TODO*                         |
| BasicVSR             |                     *TODO*                     |                         *TODO*                         | TOF      |                       -                        |                         *TODO*                         |
| **Deblurring**       |                                                |                                                        |          |                                                |                                                        |
| DeblurGANv2          |                       -                        |                         *TODO*                         |          |                                                |                                                        |
| **Denoise**          |                                                |                                                        |          |                                                |                                                        |
| RIDNet               |                       -                        |                         *TODO*                         | CBDNet   |                       -                        |                         *TODO*                         |

## ✨ **Projects that use BasicSR**

- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration

If you use `BasicSR` in your open-source projects, welcome to contact me (by [email](#-contact) or opening an issue/pull request). I will add your projects to the above list 😊

## 📜 License and Acknowledgement

This project is released under the [Apache 2.0 license](LICENSE.txt).<br>
More details about **license** and **acknowledgement** are in [LICENSE](LICENSE/README.md).

## 🌏 Citations

If BasicSR helps your research or work, please cite BasicSR.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

``` latex
@misc{basicsr,
  author =       {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/XPixelGroup/BasicSR}},
  year =         {2022}
}
```

> Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. <https://github.com/xinntao/BasicSR>, 2022.

## 📧 Contact

If you have any questions, please email `xintao.alpha@gmail.com`, `xintao.wang@outlook.com`.

<br>

- **QQ群**: 扫描左边二维码 或者 搜索QQ群号: 320960100   入群答案：互帮互助共同进步
- **微信群**: 我们的一群已经满500人啦，二群也超过200人了；进群可以添加 Liangbin 的个人微信 (右边二维码)，他会在空闲的时候拉大家入群~

<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/134879983-6f2d663b-16e7-49f2-97e1-7c53c8a5f71a.jpg"  height="300">  &emsp;
  <img src="https://user-images.githubusercontent.com/17445847/139572512-8e192aac-00fa-432b-ac8e-a33026b019df.png"  height="300">
</p>

![visitors](https://visitor-badge.glitch.me/badge?page_id=XPixelGroup/BasicSR) (start from 2022-11-06)
