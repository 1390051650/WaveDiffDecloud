'''
import os
os.system("CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 eval_diffusion.py --config raindrop_wavelet.yml --world_size=1 --resume /data1/weather/ckpts/RainDrop_epoch33420_ddpm.pth.tar")
'''
import subprocess

# 运行 train_hfrm.py
subprocess.run(["python", "train_hfrm.py"])

# 设置环境变量并运行 train_diffusion.py
subprocess.run(["CUDA_VISIBLE_DEVICES=0", "python", "train_diffusion.py", "--config", "raindrop_wavelet.yml", "--test_set", "raindrop_wavelet"], shell=True)
