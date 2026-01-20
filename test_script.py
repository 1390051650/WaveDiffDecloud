import subprocess
subprocess.run(["python", "train_StruTex.py"])
subprocess.run(["CUDA_VISIBLE_DEVICES=0", "python", "train_diffusion.py", "--config", "rice1.yml", "--test_set", "rice1_wavelet"], shell=True)
