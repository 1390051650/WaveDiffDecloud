import os
os.system("python train_strutex.py")
os.system("CUDA_VISIBLE_DEVICES=0 python train_diffusion.py --config ricedataset.yml --test_set rice")
