WaveDiffDecloud: Wavelet-Domain Conditional Diffusion Model for Efficient Cloud Removal
Official implementation code for the paper "WaveDiffDecloud: Wavelet-Domain Conditional Diffusion Model for Efficient Cloud Removal" by Yingjie Huang et al.

📖 Abstract
This repository contains the official implementation of WaveDiffDecloud, a novel approach for cloud removal in satellite imagery using wavelet-domain conditional diffusion models.
Cloud cover frequently occludes up to 60% of optical satellite acquisitions, creating data gaps and radiometric distortions that impede continuous Earth-monitoring applications. While traditional diffusion models show promise, they often suffer from slow inference speeds and texture blurring in high-dimensional pixel space.
WaveDiffDecloud addresses these limitations by learning to synthesize the wavelet coefficients of cloud-free images rather than generating pixels directly. This design substantially reduces computational complexity while preserving fine structural details. Experimental results on RICE and NUAA-CR4L89 benchmarks demonstrate that WaveDiffDecloud achieves state-of-the-art performance. Notably, on the RICE-I dataset, our method achieves the best SSIM of 0.957 and LPIPS of 0.063, significantly outperforming existing methods in texture fidelity while maintaining competitive PSNR.

✨ Key Features
Wavelet-Domain Generative Process: Synthesizes wavelet coefficients instead of pixels to reduce computational complexity and accelerate inference while maintaining high-quality output.
Structure-Aware Reconstruction: Features a specialized Structure-and Texture-aware High-Frequency Reconstruction module that models correlations among high-frequency subbands to recover sharp boundaries and surface textures.
Physics-Inspired Optimization: Utilizes a cloud-aware loss function to explicitly target and resolve radiometric distortions and boundary artifacts at cloud edges.
Spectral Consistency: Demonstrates exceptional robustness across multi-band scenarios, maintaining consistency from visible to thermal infrared wavelengths.
Comprehensive Evaluation: Includes training and testing scripts verified on RICE and NUAA-CR4L89 datasets, with support for extending to other satellite imagery formats.

🛠️ Requirements
code
Bash
pip install -r requirements.txt
🚀 Installation
Clone this repository:
code
Bash
git clone https://github.com/1390051650/WaveDiffDecloud.git
cd WaveDiffDecloud
Install dependencies:
code
Bash
pip install -r requirements.txt
🏃 Usage
1. Training
You can train the WaveDiffDecloud model on different datasets. The training process generally involves the structural/texture module and the diffusion model.
Train on Rice dataset variants:
code
Bash
# Step 1: Train the Structure-Texture module
python train_StruTex.py

# Step 2: Train the Diffusion model
python train_diffusion.py --config "rice1.yml" --resume "Rice1_ddpm.pth.tar"
Train with custom configurations:
code
Bash
python train_StruTex.py
python train.py --config "your_config.yml" --resume "your_checkpoint.pth.tar"
2. Evaluation
Test the trained models to evaluate weather/cloud removal performance.
code
Bash
# General test script
python test_script.py

# Evaluate on Rice datasets
python eval_diffusion.py --config "rice1.yml" --resume "Rice1_epoch2000_ddpm.pth.tar"
python eval_diffusion.py --config "rice2.yml" --resume "Rice1_epoch2000_ddpm.pth.tar"
3. Configuration
The model can be configured using YAML files located in the configs/ directory:
rice1.yml: Configuration for Rice dataset variant 1.
rice2.yml: Configuration for Rice dataset variant 2.
You can create custom .yml files for other datasets.
🧠 Model Architecture
The WaveDiffDecloud framework consists of several key components:
Wavelet Transform Module: Converts images to the wavelet domain for multi-scale processing.
Conditional Diffusion Model: Generates cloud-free images conditioned on input cloudy features.
Modified U-Net: A U-Net architecture optimized for wavelet-aware processing.
Multi-scale Loss Functions: Combines pixel-level losses with perceptual losses for high fidelity.
📊 Results
The model achieves state-of-the-art performance on cloud removal tasks:
High Fidelity: Best SSIM (0.957) and LPIPS (0.063) on RICE-I.
Efficiency: Significantly faster inference via wavelet domain processing.
Robustness: Effective across different cloud types and densities.
🙏 Acknowledgment
This code is heavily based on PatchDM. Many thanks to the authors for their contributions to the community.
🖊️ Citation
If you find this repository or work helpful in your research, please cite our paper:
code
Bibtex
@article{WaveDiffDecloud2026,
  title = {WaveDiffDecloud: Wavelet-domain conditional diffusion model for efficient cloud removal},
  journal = {Computers & Geosciences},
  volume = {209},
  pages = {106121},
  year = {2026},
  issn = {0098-3004},
  doi = {10.1016/j.cageo.2026.106121},
  url = {https://www.sciencedirect.com/science/article/pii/S009830042600018X},
  author = {Huang, Yingjie and Wang, Zewen and Luo, Min and Qiu, Shufang}
}
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact
For questions and support, please contact:
Author: Yingjie Huang
Email: 2020110298@ecut.edu.cn
