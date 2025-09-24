# WaveDiffDecloud: Wavelet-Domain Conditional Diffusion Model for Efficient Cloud Removal

#### This is the official implementation code for the paper "WaveDiffDecloud: Wavelet-Domain Conditional Diffusion Model for Efficient Cloud Removal" by Yingjie Huang.

## Abstract

This repository contains the implementation of WaveDiffDecloud, a novel approach for cloud removal in satellite imagery using wavelet-domain conditional diffusion models. The method leverages the multi-scale representation capabilities of wavelets combined with the powerful generative capabilities of diffusion models to efficiently remove clouds while preserving fine details in satellite images.

## Key Features

- **Wavelet-Domain Processing**: Utilizes wavelet transforms for multi-scale image representation
- **Conditional Diffusion Models**: Employs diffusion models conditioned on cloud-free information
- **Efficient Cloud Removal**: Optimized for both quality and computational efficiency
- **Multiple Dataset Support**: Compatible with various satellite imagery datasets (raindrop, rice1, rice2, etc.)
- **Comprehensive Evaluation**: Includes training and testing scripts for different scenarios
- **Easy-to-use Examples**: Quick start scripts and detailed documentation

## Notes

📰 2024: The official implementation is released. This code provides a complete framework for training and evaluating the WaveDiffDecloud model for cloud removal tasks.


## Requirements

```
pip install -r requirements.txt
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/1390051650/WaveDiffDecloud.git
cd WaveDiffDecloud
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the WaveDiffDecloud model on different datasets:

```bash

# Train on rice dataset variants
python train_diffusion.py --config "rice1.yml" --resume "Rice1_ddpm.pth.tar"

# Train with different configurations
python train_hfrm.py
python train.py --config "rice2.yml" --resume "Rice2_ddpm.pth.tar"
```

### Evaluation

Test the trained models:

```bash
# Test weather/cloud removal performance
python test_weather_script.py

# Evaluate on rice datasets
python eval_diffusion.py --config "rice1.yml" --resume "Rice1_epoch1140_ddpm.pth.tar"
python eval_diffusion.py --config "rice1.yml" --resume "Rice1_epoch390_ddpm.pth.tar"
```

### Configuration

The model can be configured using YAML files in the `configs/` directory:
- `rice1.yml`: Rice dataset variant 1
- `rice2.yml`: Rice dataset variant 2

## Quick Start

For a quick demonstration of cloud removal, use the provided example script:

```bash
python examples/quick_start.py --input path/to/cloudy_image.jpg --output path/to/clean_image.jpg
```

See the [examples/README.md](examples/README.md) for more detailed usage instructions.

## Model Architecture

The WaveDiffDecloud model consists of several key components:

- **Wavelet Transform Module**: Converts images to wavelet domain for multi-scale processing
- **Conditional Diffusion Model**: Generates cloud-free images conditioned on input features
- **U-Net Architecture**: Modified U-Net with wavelet-aware processing
- **Multi-scale Loss Functions**: Combines pixel-level and perceptual losses

## Dataset Support

This implementation supports multiple datasets for cloud removal:

- **Rice Datasets**: Agricultural satellite imagery with cloud coverage
- **Custom Datasets**: Easily extensible to other satellite imagery datasets

## Results

The model achieves state-of-the-art performance on cloud removal tasks with:
- High-quality cloud-free image generation
- Efficient processing in wavelet domain
- Robust performance across different cloud types and densities

## Acknowledgment

This code is heavily based on [PatchDM](https://github.com/IGITUGraz/WeatherDiffusion). Many thanks to the authors!

## Citation

If you find this repository/work helpful in your research, please cite our paper:

```bibtex
@article{huang2024wavediffdecloud,
  title={WaveDiffDecloud: Wavelet-Domain Conditional Diffusion Model for Efficient Cloud Removal},
  author={Huang, Yingjie},
  journal={},
  year={2024},
  publisher={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact:
- Author: Yingjie Huang
- Email: [2020110298@ecut.edu.cn]

