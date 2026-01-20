# WaveDiffDecloud

<p align="center">
  <h3 align="center">Wavelet-Domain Conditional Diffusion Model for Efficient Cloud Removal</h3>
</p>

<p align="center">
  <em>Official implementation of the paper published in <strong>Computers & Geosciences</strong></em>
</p>

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S009830042600018X">
    <img src="https://img.shields.io/badge/Paper-Elsevier-orange.svg" alt="Paper">
  </a>
  <a href="https://doi.org/10.1016/j.cageo.2026.106121">
    <img src="https://img.shields.io/badge/DOI-10.1016/j.cageo.2026.106121-blue.svg" alt="DOI">
  </a>
  <a href="https://github.com/1390051650/WaveDiffDecloud/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://github.com/1390051650/WaveDiffDecloud">
    <img src="https://img.shields.io/github/stars/1390051650/WaveDiffDecloud?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  <img src="assets/framework.png" width="90%" alt="WaveDiffDecloud Framework">
</p>

---


---

## 📖 Abstract

Cloud cover frequently occludes up to **60%** of optical satellite acquisitions, creating data gaps and radiometric distortions that impede continuous Earth-monitoring applications. While traditional diffusion models show promise, they often suffer from slow inference speeds and texture blurring in high-dimensional pixel space.

**WaveDiffDecloud** addresses these limitations by learning to synthesize the **wavelet coefficients** of cloud-free images rather than generating pixels directly. This design substantially reduces computational complexity while preserving fine structural details.

> 📊 **Key Results:** On the **RICE-I dataset**, our method achieves:
> - **SSIM: 0.957** 
> - **LPIPS: 0.063** 
> - Significantly outperforming existing methods in texture fidelity

---

## ✨ Highlights

| Feature | Description |
|:-------:|:------------|
| 🌊 **Wavelet-Domain Generation** | Synthesizes wavelet coefficients instead of pixels to reduce computational complexity and accelerate inference |
| 🎯 **StruTex-HFR Module** | Structure-and Texture-aware High-Frequency Reconstruction for sharp boundaries and surface textures |
| ⚡ **Physics-Inspired Loss** | Cloud-aware loss function targeting radiometric distortions at cloud edges |
| 🌈 **Spectral Consistency** | Robust across multi-band scenarios from visible to thermal infrared |

---

## 🛠️ Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3

### Setup

```bash
# Clone the repository
git clone https://github.com/1390051650/WaveDiffDecloud.git
cd WaveDiffDecloud

# Create conda environment (recommended)
conda create -n wavediff python=3.8 -y
conda activate wavediff

# Install dependencies
pip install -r requirements.txt



📁 Dataset Preparation

Download and organize the datasets as follows:

datasets/
├── RICE/
│   ├── RICE1/
│   │   ├── cloud/          # Cloudy images
│   │   └── label/          # Ground truth
│   └── RICE2/
│       ├── cloud/
│       └── label/
└── NUAA-CR4L89/
    ├── train/
    ├── val/
    └── test/



🚀 Usage
Training

Training consists of two stages:

# Stage 1: Train Structure-Texture Module (StruTex-HFR)
python train_StruTex.py

# Stage 2: Train Diffusion Model
# For RICE-I
python train_diffusion.py --config "rice1.yml" --resume "Rice1_ddpm.pth.tar"

# For RICE-II
python train_diffusion.py --config "rice2.yml" --resume "Rice2_ddpm.pth.tar"

# For custom dataset
python train_diffusion.py --config "your_config.yml" --resume "your_checkpoint.pth.tar"

Evaluation
# General evaluation
python test_script.py

# Evaluate on RICE-I
python eval_diffusion.py --config "rice1.yml" --resume "Rice1_epoch2000_ddpm.pth.tar"

# Evaluate on RICE-II
python eval_diffusion.py --config "rice2.yml" --resume "Rice2_epoch2000_ddpm.pth.tar"

Configuration

Configuration files are located in configs/:

Config File	Description
rice1.yml	RICE dataset variant 1
rice2.yml	RICE dataset variant 2
🧠 Model Architecture
┌─────────────────────────────────────────────────────────────────────┐
│                     WaveDiffDecloud Framework                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐      ┌─────────────────┐      ┌──────────────────┐   │
│   │  Cloudy  │      │ Wavelet Domain  │      │   Conditional    │   │
│   │  Image   │─────►│   Transform     │─────►│   Diffusion      │   │
│   │          │      │   (DWT)         │      │   Model          │   │
│   └──────────┘      └─────────────────┘      └────────┬─────────┘   │
│                                                       │             │
│                                                       ▼             │
│   ┌──────────┐      ┌─────────────────┐      ┌──────────────────┐   │
│   │  Clean   │      │ Inverse Wavelet │      │  StruTex-HFR     │   │
│   │  Image   │◄─────│   Transform     │◄─────│  Module          │   │
│   │          │      │   (IDWT)        │      │                  │   │
│   └──────────┘      └─────────────────┘      └──────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Key Components
Component	Function
Wavelet Transform	Converts images to wavelet domain (LL, LH, HL, HH subbands)
Conditional Diffusion	Generates cloud-free wavelet coefficients
StruTex-HFR	Enhances high-frequency details and edge sharpness
Multi-scale Loss	Combines pixel-level and perceptual losses
