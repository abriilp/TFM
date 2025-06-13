import argparse
import os, json
import random
import shutil
import time, glob, copy
import os
import time
import torch
import socket
import argparse
import subprocess
import math
from tqdm import tqdm
import warnings
import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse, pdb
import numpy as np
from torch import autograd
from torch.optim import Adam, SGD, AdamW
from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple, Callable
import pdb
#from model_seg import SegNet
import builtins
import torchvision.utils as vutils
from PIL import Image
from diffusion_extractor import StableDiffusion
from PIL import ImageFilter, ImageOps
from dataloader import MIT_Dataset
from pytorch_ssim import SSIM as compute_SSIM_loss
from pytorch_losses import gradient_loss

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Define Hyperparameter
gpu_index = 0
total_folder = 6
precompute_feat_list = True
mit_data_path = '/net/scratch/zhang7/dataset/mit_multiview'
save_precompute_feat_path = 'feat_list'
os.makedirs(save_precompute_feat_path, exist_ok = True)

# Initialize the stable diffusion model
diffusion = StableDiffusion(gpu_index)
transform = transforms.Compose([
        transforms.Resize((512,512), interpolation=3),
        transforms.ToTensor(),
        ])

# Extract the latent features from UNet's decoder and save them to a folder
# Running stable diffusion during training time could be slow.
# It's more efficient to precompute and load later.
if precompute_feat_list:
    dataset = MIT_Dataset(mit_data_path, transform)
    for folder_index in range(total_folder):
        img_list = dataset.get_img_folder_list(folder_index)
        for img_id, img in enumerate(img_list):
            feat_list, img_latent = diffusion.collect_feat(img[None].cuda(gpu_index))
            torch.save(feat_list + [img_latent, img], '{}/{:02d}_{:05d}.pth'.format(save_precompute_feat_path, folder_index, img_id))

# Define the learnable rendering pipeline
class Model(nn.Module):
    def __init__(self, feat = 128):
        super().__init__()
        self.global_fc = nn.Linear(1280 * 2, 2 * feat)
        self.skip1 = nn.Conv2d(1280, feat, kernel_size = 3, padding = 1)
        self.skip2 = nn.Conv2d(1280, feat, kernel_size = 3, padding = 1)
        self.skip3 = nn.Conv2d(640, feat, kernel_size = 3, padding = 1)
        self.skip4 = nn.Conv2d(320, feat, kernel_size = 3, padding = 1)
        self.albedo = nn.Sequential(nn.Conv2d(feat, feat, kernel_size = 3, padding = 1), nn.ReLU(), nn.Conv2d(feat, 4, kernel_size = 3, padding = 1))
        self.surface = nn.Conv2d(feat, feat, kernel_size = 3, padding = 1)
        self.shading = nn.Sequential(nn.Conv2d(feat, feat, kernel_size = 3, padding = 1), nn.ReLU(), nn.Conv2d(feat, 4, kernel_size = 3, padding = 1))

    def forward(self, feat_list, albedo = None, surface = None, scale_shift_light = None):
        global_feat = torch.cat([feat_list[0].mean(dim = [2,3]), feat_list[1].mean(dim = [2,3])], dim = 1)
        if scale_shift_light is None:
            light_scale, light_shift = self.global_fc(global_feat).chunk(2, dim = -1)
        else:
            light_scale, light_shift = scale_shift_light
        spatial_feat = F.interpolate(self.skip1(feat_list[0]), size = (64, 64), mode = 'bilinear') + \
                F.interpolate(self.skip2(feat_list[1]), size = (64, 64), mode = 'bilinear') + \
                F.interpolate(self.skip3(feat_list[2]), size = (64, 64), mode = 'bilinear') + \
                F.interpolate(self.skip4(feat_list[3]), size = (64, 64), mode = 'bilinear')
        if albedo is None:
            albedo = self.albedo(spatial_feat)
        if surface is None:
            surface = self.surface(spatial_feat)
        shading = self.shading(F.relu((light_scale[:,:,None,None] + 1) * surface + light_shift[:,:,None,None]))
        out = albedo * shading
        return albedo, surface, shading, (light_scale, light_shift), out

model = Model()
model.cuda()

optimizer = AdamW(model.parameters(),
            lr=  2e-4, weight_decay = 1e-4)
batch_size = 10
total_train_iter = 3000
# Training pipeline.
for i in range(total_train_iter):
    sid = np.random.randint(total_folder)
    light_id = torch.randperm(25)[:2]
    feat1_list = torch.load('feat_list/{:02d}_{:05d}.pth'.format(sid, light_id[0].item()))
    feat2_list = torch.load('feat_list/{:02d}_{:05d}.pth'.format(sid, light_id[1].item()))
    albedo1, surface1, _, _, out1 = model(feat1_list[:4])
    albedo2, surface2, _, _, out2 = model(feat2_list[:4])
    albedo_loss = F.mse_loss(albedo1, albedo2)
    surface_loss = F.mse_loss(surface1, surface2)
    rec_loss = F.mse_loss(out1, feat1_list[-2]) + F.mse_loss(out2, feat2_list[-2])
    loss = 1.0 / batch_size * (10 * albedo_loss + 10 * surface_loss + rec_loss)
    loss.backward()
    if i % batch_size == 0 and i != 0:
        optimizer.step()
        optimizer.zero_grad()
    print(i, rec_loss.item(), albedo_loss.item(), surface_loss.item())

# save checkpoint
torch.save(model.state_dict(), 'model_dict.pth')

def render_latent(out):
    latents = 1 / 0.18215 * out
    with torch.no_grad():
        image = diffusion.vae.decode(latents).sample
    return ((image.clamp(-1, 1).cpu().data.numpy()[0].transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)

# load checkpoint and render output
model.load_state_dict(torch.load('model_dict.pth'))
model.cuda()
sid = 1
light_id = 2
feat_list1 = torch.load('feat_list/{:02d}_{:05d}.pth'.format(sid, light_id))
albedo1, surface1, shading1, light1, out1 = model(feat_list1[:4])

sid = 2
light_id = 1
feat_list2 = torch.load('feat_list/{:02d}_{:05d}.pth'.format(sid, light_id))
albedo2, surface2, shading2, light2, out2 = model(feat_list2[:4])

out1_light = model(feat_list1[:4], scale_shift_light = light2)[-1]
out1_surface = model(feat_list1[:4], surface = surface2)[-1]
out1_albedo = model(feat_list1[:4], albedo = albedo2)[-1]

rec_img = render_latent(out1)
gt_img = render_latent(feat_list1[4])
img1_light = render_latent(out1_light)
img1_surface = render_latent(out1_surface)
img1_albedo = render_latent(out1_albedo)
all_img1 = np.concatenate([gt_img, rec_img, img1_light, img1_surface, img1_albedo], axis = 1)

out2_light = model(feat_list2[:4], scale_shift_light = light1)[-1]
out2_surface = model(feat_list2[:4], surface = surface1)[-1]
out2_albedo = model(feat_list2[:4], albedo = albedo1)[-1]

rec_img = render_latent(out2)
gt_img = render_latent(feat_list2[4])
img2_light = render_latent(out2_light)
img2_surface = render_latent(out2_surface)
img2_albedo = render_latent(out2_albedo)
all_img2 = np.concatenate([gt_img, rec_img, img2_light, img2_surface, img2_albedo], axis = 1)
all_img = np.concatenate([all_img1, all_img2], axis = 0)
Image.fromarray(all_img).save('img.png')
pdb.set_trace()
