import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image
import pdb
import torchvision.transforms as transforms
from torchvision.utils import save_image
#import torchvision.models.vgg as vgg
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from models.model_dense import *      #unet...................................
from models.arch import HFRM

from datasets.dataset import * 

import torch.nn as nn
import torch.nn.functional as F
import torch

# ---------------------------
# 定义PSNR计算函数
def BatchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean(dim=(1,2,3)).sqrt()
    ps = 20 * torch.log10(1/rmse + 1e-8)
    return ps

# ---------------------------
# 数据预处理函数
def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input - output))
        
# ---------------------------
# TVLoss（总变分损失）定义
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
        
# ---------------------------
# 感知损失网络（使用 VGG19 提取中间特征）
class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",
        }
        
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output

# ---------------------------
# 定义 Haze Line Loss
def haze_line_loss(fake, hazy, eps=1e-6):
    """
    计算 haze line loss：
    对于每个 batch 内图像，根据 hazy（有云图像）估计大气光 A，
    然后对 fake（去云后图像）的每个像素，计算其与 haze line 的距离，
    haze line 定义为经过 A 和原点的直线。为了鼓励去云图像远离 haze line，
    使用 exp(-distance) 当距离较小时产生较大损失。
    """
    # fake, hazy: [B, C, H, W]
    B, C, H, W = hazy.shape
    # 对每张图像，按通道取最大值作为大气光 A
    A = hazy.view(B, C, -1).max(dim=2)[0]  # shape: [B, C]
    # 计算 A 的 L2 范数的平方（避免除 0）
    A_norm_sq = (A ** 2).sum(dim=1, keepdim=True) + eps  # shape: [B, 1]
    # 将 A reshape 成 [B, C, 1, 1]，便于后续广播
    A_reshaped = A.view(B, C, 1, 1)
    # 对 fake 的每个像素，计算与 A 的点积
    dot = (fake * A_reshaped).sum(dim=1, keepdim=True)  # shape: [B, 1, H, W]
    # 计算 fake 在 A 方向上的投影
    proj = (dot / A_norm_sq.view(B, 1, 1, 1)) * A_reshaped  # shape: [B, C, H, W]
    # 计算投影残差
    diff = fake - proj
    # 对每个像素计算 L2 距离
    dist = torch.sqrt((diff ** 2).sum(dim=1) + eps)  # shape: [B, H, W]
    # 使用指数函数，当距离小（靠近 haze line）时，损失较大
    loss = torch.mean(torch.exp(-dist))
    return loss

# ---------------------------
# 定义 SSIM 相关函数
def gaussian_window(window_size, sigma, channel):
    """创建 2D 高斯窗，用于 SSIM 计算"""
    gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)  # shape: [window_size, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # shape: [1, 1, window_size, window_size]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(pred, target, window_size=11, sigma=1.5, size_average=True):
    """
    计算 SSIM 指数。参见论文：
    Wang et al., "Image quality assessment: from error visibility to structural similarity", IEEE TPAMI, 2004.
    """
    (_, channel, _, _) = pred.size()
    window = gaussian_window(window_size, sigma, channel).to(pred.device)
    
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(pred, target):
    """SSIM Loss 定义为 1 - SSIM 指数"""
    return 1 - ssim(pred, target, size_average=True)

# ---------------------------
# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=579, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="rice1", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=6, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')      
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--data_url', type=str, default="", help='name of the dataset')
parser.add_argument('--init_method', type=str, default="", help='name of the dataset')
parser.add_argument('--train_url', type=str, default="", help='name of the dataset')

opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# ---------------------------
# 损失函数
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
# TVLoss 的权重可以单独调节
tvloss = TVLoss(TVLoss_weight=1.0)
lossmse = torch.nn.MSELoss()

# 这里定义各项损失的权重（可根据实际任务调参）
lambda_pixel = 100
lambda_haze_line = 10
lambda_ssim = 5
lambda_tv = 0.1

patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# ---------------------------
# 初始化模型
img_channel = 3
dim = 32
enc_blks = [2, 2, 2, 4]
middle_blk_num = 6
dec_blks = [2, 2, 2, 2]
generator = HFRM(in_channel=img_channel, dim=dim, mid_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
pytorch_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("Total_params_model: {}M".format(pytorch_total_params / 1000000.0))

if cuda:
    generator = generator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_pixelwise = criterion_pixelwise.cuda()
    lossnet = LossNetwork().float().cuda()

if opt.epoch != 0:
    generator.load_state_dict(torch.load('./saved_models/rice1/lastest.pth'), strict=True)
else:
    # 初始化权重
    generator.apply(weights_init_normal)

device = torch.device("cuda:0")

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

mytransform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
    
# 修改为你自己的数据路径
data_root = '../autodl-tmp/RICE1/train'
myfolder = myImageFloder(root=data_root, transform=mytransform, crop=False, resize=False, crop_size=512, resize_size=512)
dataloader = DataLoader(myfolder, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True)
print('Data loader finished!')

def get_mask(dg_img, img):
    mask = np.fabs(dg_img.cpu() - img.cpu())
    mask[mask < (20.0 / 255.0)] = 0.0
    mask = mask.cuda()
    return mask

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(epoch, i, real_A, real_B, fake_B):
    data, pred, label = real_A * 255, fake_B * 255, real_B * 255
    data = data.cpu()
    pred = pred.cpu()
    label = label.cpu()
    pred = torch.clamp(pred.detach(), 0, 255)
    data, pred, label = data.int(), pred.int(), label.int()
    h, w = pred.shape[-2], pred.shape[-1]
    img = np.zeros((h, 3 * w, 3))
    for idx in range(0, 1):
        row = idx * h
        tmplist = [data[idx], pred[idx], label[idx]]
        for k in range(3):
            col = k * w
            tmp = np.transpose(tmplist[k], (1, 2, 0))
            img[row:row+h, col:col+w] = np.array(tmp)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save("./train_result/rice1/%03d_%06d.png" % (epoch, i))
    
# ---------------------------
# 训练循环
prev_time = time.time()
step = 0
best_psnr = 40

for epoch in range(opt.epoch, opt.n_epochs):
    epoch_psnr = []
    for i, batch in enumerate(tqdm(dataloader), 0):
        step += 1
        # 学习率衰减策略（示例）
        current_lr = opt.lr * (1 / 2)**(step / 100000)
        for param_group in optimizer_G.param_groups:
            param_group["lr"] = current_lr
            
        img_train = batch
        # 假设 img_train[0] 是有云图像，img_train[1] 是去云（ground truth）图像
        real_A, real_B = Variable(img_train[0].cuda()), Variable(img_train[1].cuda())
        batch_size = real_B.size(0)
        
        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        
        # 计算各项损失
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        loss_haze = haze_line_loss(fake_B, real_A)
        loss_ssim = ssim_loss(fake_B, real_B)
        loss_tv_val = tvloss(fake_B)
        
        # 总损失：各项损失加权求和
        loss_G = (lambda_pixel * loss_pixel +
                  lambda_haze_line * loss_haze +
                  lambda_ssim * loss_ssim +
                  lambda_tv * loss_tv_val)
        loss_G.backward()
        optimizer_G.step()

        psnr = BatchPSNR(fake_B, real_B)
        epoch_psnr.append(psnr.mean().item())
        print("PSNR this: %f" % psnr.mean().item())

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        if i % 100 == 0:
            print("G loss: %f (pixel: %f, haze: %f, ssim: %f, tv: %f)" %
                  (loss_G.item(), loss_pixel.item(), loss_haze.item(), loss_ssim.item(), loss_tv_val.item()))
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s" %
                             (epoch, opt.n_epochs, i, len(dataloader),
                              loss_G.item(), time_left))
            
        if i % 1000 == 0:
            sample_images(epoch, i, real_A, real_B, fake_B)
            
    print("Epoch %d PSNR: %f, best psnr: %f" % (epoch, np.mean(epoch_psnr), best_psnr))
    if np.mean(epoch_psnr) > best_psnr:
        best_psnr = np.mean(epoch_psnr)
        torch.save(generator.state_dict(), './saved_models/%s/best.pth' % opt.dataset_name)

    torch.save(generator.state_dict(), './saved_models/%s/lastest.pth' % opt.dataset_name)
    if (epoch + 1) % 20 == 0:
        torch.save(generator.state_dict(), './saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
