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
from utils import  AverageMeter, ProgressMeter, init_ema_model, update_ema_model
import builtins
import torchvision.utils as vutils
from PIL import Image
import unets, torchvision
import tqdm
from utils import MIT_Dataset_Normal, MIT_Dataset,affine_crop_resize
from skimage.metrics import structural_similarity as ssim

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import lpips
from skimage import color

class DINO_extractor():
    def __init__(self, device):
        from dino import DinoFeaturizer
        self.dino = DinoFeaturizer()
        self.dino = self.dino.to(device)
        self.normalize_mean=torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).to(device)
        self.normalize_std=torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).to(device)

    def extract_feat(self, img):
        img = img * 0.5 + 0.5
        img = (img - self.normalize_mean) / self.normalize_std
        batch_size = 16
        feat_list = []
        for i in range((img.shape[0] + batch_size - 1) // batch_size):
            feat_list.append(F.interpolate(self.dino(img[i * batch_size: (i+1) * batch_size]), size = (128, 128), mode = 'bilinear'))
        return torch.cat(feat_list)

@torch.no_grad()
def extract_feat_hypercolumn(model, all_img, batch_size = 64, target_res = 128):
    all_feat = []
    for i in range((all_img.shape[0] + batch_size - 1) // batch_size):
        img = all_img[i * batch_size: (i + 1) * batch_size]
        P_mean=-0.5
        P_std=1.2
        sigma_data = 0.5
        rnd_normal = torch.randn([img.shape[0], 1, 1, 1], device=img.device)
        sigma = (rnd_normal * P_std + P_mean).exp() * 0 + 0.001
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        noise = torch.randn_like(img)
        noisy_img = img + noise * sigma
        _, _, intrinsic = model(noisy_img, sigma, run_encoder = True)
        out_feat = []
        for feat_intrinsic in intrinsic:
            out_feat.append(F.interpolate(feat_intrinsic, size = target_res, mode = 'bilinear'))
        all_feat.append(torch.cat(out_feat, dim = 1))
    return torch.cat(all_feat)

def get_surface_norma_dataloder(args):
    #crop = affine_crop_resize(size = (256, 256), scale = (0.2, 1.0))
    train_dataset = MIT_Dataset_Normal('/net/projects/willettlab/vdataset/mit_multiview_resize')
    test_dataset = MIT_Dataset_Normal('/net/projects/willettlab/vdataset/mit_multiview_test')
    train_dataset[0]
    #test_dataset = VIDIT_Dataset_val('/net/projects/willettlab/roxie62/dataset/VIDIT', transform_test)
    print('NUM of training images: {}'.format(len(train_dataset)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True, drop_last = True)
        test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last = True, persistent_workers = True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last = False, persistent_workers = False)
    return train_loader, test_loader

def get_eval_relight_dataloder(args, eval_pair_folder_shift = 5, eval_pair_light_shift = 5):
    transform_test = [None, transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])]
    train_dataset = MIT_Dataset('/net/projects/willettlab/vdataset/mit_multiview_test', transform_test, eval_mode = True,
                   )
    #test_dataset = VIDIT_Dataset_val('/net/projects/willettlab/roxie62/dataset/VIDIT', transform_test)
    print('NUM of training images: {}'.format(len(train_dataset)))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last = False, persistent_workers = False)
    return train_loader

def get_eval_relight_dataloder2(args, eval_pair_folder_shift = 5, eval_pair_light_shift = 5):
    transform_test = [
        None, 
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    ]
    from dataset_utils import RSRDataset
    train_dataset = RSRDataset(root_dir=args.data_path, img_transform=transform_test, 
                                is_validation=True, validation_scenes=["scene_01", "scene_03", "scene_05", "scene_06", "scene_07", "scene_08", "scene_10", "scene_21", "scene_22", "scene_23", "scene_24", "scene_26", "scene_27", "scene_29", "scene_30"])
    #test_dataset = VIDIT_Dataset_val('/net/projects/willettlab/roxie62/dataset/VIDIT', transform_test)
    print('NUM of training images: {}'.format(len(train_dataset)))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last = False, persistent_workers = False)
    return train_loader

def eval_surface(args, model):
    train_loader, test_loader = get_surface_norma_dataloder(args)
    # switch to train mode
    t0 = time.time()
    epoch_iter = len(train_loader)
    P_mean=-0.5
    P_std=1.2

    class probing(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.cls = nn.Conv2d(1088, 3, kernel_size = 1, padding = 0)
            #self.cls = nn.Sequential(nn.Conv2d(1088, 512, kernel_size = 1, padding = 0),
            #                    nn.ReLU(),
            #                    nn.Conv2d(512, 3, kernel_size = 1, padding = 0)
            #                    )
        def forward(self, input):
            return self.cls(input)

    prob = probing(1088)
    prob = prob.to(args.gpu)
    prob = torch.nn.parallel.DistributedDataParallel(prob, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    optimizer = AdamW(prob.parameters(),
                lr= 1e-3, weight_decay = 1e-4)

    def visualize(img, pred, target, img_name):
        img = F.interpolate(img, size = (pred.shape[2], pred.shape[3]), mode = 'bilinear', antialias = True)
        n,_,h,w = img.shape
        img = ((img * 0.5 + 0.5) * 255).cpu().data
        pred = ((pred * 0.5 + 0.5) * 255).cpu().data
        target = ((target * 0.5 + 0.5) * 255).cpu().data
        #n * 3 * c * h * w
        img = torch.stack([img, pred, target], dim = 1).permute(0, 2, 3, 1, 4).reshape(n, -1, h, 3 * w)
        img = img[:27].reshape(9,3,-1, h, 3 * w).permute(0, 3, 1, 4, 2).reshape(9 * h, 9 * w, -1)
        Image.fromarray(img.numpy().astype(np.uint8)).save(img_name + '.png')
    @torch.no_grad()
    def eval_result(epoch):
        result = {'idx':[], 'offset':[], 'sim':[], 'out':[]}
        prob.eval()
        for i, (img,normal, folder_idx, folder_offset) in enumerate(test_loader):
            img = img.to(args.gpu)
            normal= normal.to(args.gpu)
            feat = extract_feat_hypercolumn(model, img)
            normal = F.interpolate(normal, size = (feat.shape[2], feat.shape[3]), mode = 'bilinear')
            normal = F.normalize(normal, dim = 1)
            out = prob(feat)
            out = F.normalize(out, dim = 1)
            sim = (out * normal).sum(dim = 1).mean(dim = [1,2])
            result['idx'].append(folder_idx)
            result['offset'].append(folder_offset)
            result['sim'].append(sim)
            result['out'].append(out)
        std = torch.cat(result['out']).reshape(-1,25, 3, out.shape[2], out.shape[3]).std(dim = 1).mean()
        sim = torch.cat(result['sim']).reshape(-1, 25)
        if args.gpu == 0:
            print(f'Epoch:{epoch}, test accuracy, sim:{sim.mean()}, std:{std}')
        if args.gpu == 0:
            visualize(img, out, normal, f'test_normal_{epoch:03d}')
        prob.train()

    for epoch in range(100):
        loss_name = [
                    'sim',
                    'GPU Mem', 'Time']
        moco_loss_meter = [AverageMeter(name, ':6.3f') for name in loss_name]
        progress = ProgressMeter(
            len(train_loader),
            moco_loss_meter,
            prefix="Epoch: [{}]".format(epoch))
        for i, (img,normal, folder_idx, folder_offset) in enumerate(train_loader):
            img = img.to(args.gpu)
            normal= normal.to(args.gpu)
            feat = extract_feat_hypercolumn(model, img)
            normal = F.interpolate(normal, size = (feat.shape[2], feat.shape[3]), mode = 'bilinear')
            normal = F.normalize(normal, dim = 1)
            out = prob(feat)
            l1_loss = (out - normal).abs().mean()
            #print(out.norm(dim = 1).mean().item())
            out = F.normalize(out, dim = 1)
            sim = (out * normal).sum(dim = 1).mean()
            optimizer.zero_grad()
            (-1 * sim + 10 * l1_loss).backward()
            optimizer.step()
            t1 = time.time()
            for val_id, val in enumerate([sim,
                            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), t1 - t0,
                        ]):
                if not isinstance(val, float) and not isinstance(val, int):
                    val = val.item()
                moco_loss_meter[val_id].update(val)
            progress.display(i)
            t0 = time.time()
            #model.module._dequeue_and_enqueue(norm_extrinsic1)
            torch.cuda.reset_peak_memory_stats()
        if args.gpu == 0:
            visualize(img, out, normal, f'normal_{epoch:03d}')
        eval_result(epoch)
    torch.distributed.barrier()
    pdb.set_trace()

def eval_ssl_surface(args):
    train_loader, test_loader = get_surface_norma_dataloder(args)

    model = DINO_extractor(args.gpu)
    # switch to train mode
    t0 = time.time()
    epoch_iter = len(train_loader)
    P_mean=-0.5
    P_std=1.2

    class probing(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.cls = nn.Conv2d(dim, 3, kernel_size = 1, padding = 0)
        def forward(self, input):
            return self.cls(input)

    prob = probing(384)
    prob = prob.to(args.gpu)
    prob = torch.nn.parallel.DistributedDataParallel(prob, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    optimizer = AdamW(prob.parameters(),
                lr= 1e-3, weight_decay = 1e-4)

    @torch.no_grad()
    def eval_result(epoch):
        result = {'idx':[], 'offset':[], 'sim':[], 'out':[]}
        for i, (img,normal, folder_idx, folder_offset) in enumerate(test_loader):
            img = img.to(args.gpu)
            normal= normal.to(args.gpu)
            feat = model.extract_feat(img)
            normal = F.interpolate(normal, size = (feat.shape[2], feat.shape[3]), mode = 'bilinear')
            normal = F.normalize(normal, dim = 1)
            out = prob(feat)
            out = F.normalize(out, dim = 1)
            sim = (out * normal).sum(dim = 1).mean(dim = [1,2])
            result['idx'].append(folder_idx)
            result['offset'].append(folder_offset)
            result['sim'].append(sim)
            result['out'].append(out)
        std = torch.cat(result['out']).reshape(-1,25, 3, out.shape[2], out.shape[3]).std(dim = 1).mean()
        sim = torch.cat(result['sim']).reshape(-1, 25)
        if args.gpu == 0:
            print(f'Epoch:{epoch}, test accuracy, sim:{sim.mean()}, std:{std}')

    def unnormalize_img(img):
        return ((img[0].permute(1,2,0) * 0.5 + 0.5) * 255).cpu().data.numpy().astype(np.uint8)

    def unnormalize_normal(normal):
        return ((img[0].permute(1,2,0) * 0.5 + 0.5) * 255).cpu().data.numpy().astype(np.uint8)

    for epoch in range(100):
        loss_name = [
                    'sim',
                    'GPU Mem', 'Time']
        moco_loss_meter = [AverageMeter(name, ':6.3f') for name in loss_name]
        progress = ProgressMeter(
            len(train_loader),
            moco_loss_meter,
            prefix="Epoch: [{}]".format(epoch))
        for i, (img,normal, folder_idx, folder_offset) in enumerate(train_loader):
            img = img.to(args.gpu)
            normal= normal.to(args.gpu)
            #feat = extract_feat_hypercolumn(model, img)
            feat = model.extract_feat(img)
            normal = F.interpolate(normal, size = (feat.shape[2], feat.shape[3]), mode = 'bilinear')
            normal = F.normalize(normal, dim = 1)
            out = prob(feat)
            l1_loss = (out - normal).abs().mean()
            out = F.normalize(out, dim = 1)
            sim = (out * normal).sum(dim = 1).mean()
            optimizer.zero_grad()
            (-1 * sim + 10 * l1_loss).backward()
            optimizer.step()
            t1 = time.time()
            for val_id, val in enumerate([sim,
                            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), t1 - t0,
                        ]):
                if not isinstance(val, float) and not isinstance(val, int):
                    val = val.item()
                moco_loss_meter[val_id].update(val)
            progress.display(i)
            t0 = time.time()
            #model.module._dequeue_and_enqueue(norm_extrinsic1)
            torch.cuda.reset_peak_memory_stats()
        eval_result(epoch)
    torch.distributed.barrier()
    pdb.set_trace()


@torch.no_grad()
def eval_relight(args, epoch, model, eval_pair_folder_shift = 5, eval_pair_light_shift = 5):
    torch.manual_seed(args.gpu)
    train_loader = get_eval_relight_dataloder2(args, eval_pair_folder_shift, eval_pair_light_shift)
    t0 = time.time()
    epoch_iter = len(train_loader)
    P_mean=-0.5
    P_std=1.2
    model.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.gpu)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.gpu)
    lpips_metric = lpips.LPIPS(net='alex').to(args.gpu)
    delta_e_metric = deltaE00()

    # Metrics accumulation
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    delta_e_scores = []
            
    # Lists to store all metrics for wandb table
    all_metrics = []
    
    for i, (img1, img2, img3) in enumerate(tqdm.tqdm(train_loader)):
        img1 = img1.to(args.gpu)
        img2 = img2.to(args.gpu)
        img3 = img3.to(args.gpu)

        rnd_normal = torch.randn([img1.shape[0], 1, 1, 1], device=img1.device)
        sigma = (rnd_normal * P_std + P_mean).exp().to(img1.device) * 0 + 0.001
        noise = torch.randn_like(img1)
        noisy_img1 = img1 + noise * sigma
        noisy_img2 = img2 + noise * sigma

        intrinsic1, extrinsic1 = model(img1, run_encoder = True)
        intrinsic2, extrinsic2 = model(img2, run_encoder = True)
        relight_img2 = model([intrinsic1, extrinsic2], run_encoder = False).clamp(-1,1)
  
        shift = (relight_img2 - img2).mean(dim = [2,3], keepdim = True)
        correct_relight_img2 = relight_img2 - shift

        # Compute metrics
        ssim_score = ssim_metric(correct_relight_img2, img3)
        psnr_score = psnr_metric(correct_relight_img2, img3)
        lpips_score = lpips_metric(correct_relight_img2, img3)
        delta_e_score = delta_e_metric(correct_relight_img2, img3)
                
        # Accumulate scores
        ssim_scores.append(ssim_score.item())
        psnr_scores.append(psnr_score.item())
        lpips_scores.append(lpips_score.item())
        delta_e_scores.append(delta_e_score)

    # Calculate average metrics
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_lpips = np.mean(lpips_scores)
    avg_delta_e = np.mean(delta_e_scores)

    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  PSNR: {avg_psnr:.4f}")
    print(f"  LPIPS: {avg_lpips:.4f}")
    print(f"  Delta E: {avg_delta_e:.4f}")
    print(f"  Samples: {len(train_loader)}")
        
    return


class deltaE00():
    def __init__(self, color_chart_area=0):
        super().__init__()
        self.color_chart_area = color_chart_area
        self.kl = 1
        self.kc = 1
        self.kh = 1

    def __call__(self, img1, img2):
        """ Compute the deltaE00 between two numpy RGB images """
        
        if type(img1) == torch.Tensor:
            assert img1.shape[0] == 1
            img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()

        if type(img2) == torch.Tensor:
            assert img2.shape[0] == 1
            img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE00
        Lstd = np.transpose(img1[:, 0])
        astd = np.transpose(img1[:, 1])
        bstd = np.transpose(img1[:, 2])
        Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
        Lsample = np.transpose(img2[:, 0])
        asample = np.transpose(img2[:, 1])
        bsample = np.transpose(img2[:, 2])
        Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
        Cabarithmean = (Cabstd + Cabsample) / 2
        G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
            Cabarithmean, 7) + np.power(25, 7))))
        apstd = (1 + G) * astd
        apsample = (1 + G) * asample
        Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
        Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
        Cpprod = (Cpsample * Cpstd)
        zcidx = np.argwhere(Cpprod == 0)
        hpstd = np.arctan2(bstd, apstd)
        hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
        hpsample = np.arctan2(bsample, apsample)
        hpsample = hpsample + 2 * np.pi * (hpsample < 0)
        hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
        dL = (Lsample - Lstd)
        dC = (Cpsample - Cpstd)
        dhp = (hpsample - hpstd)
        dhp = dhp - 2 * np.pi * (dhp > np.pi)
        dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
        dhp[zcidx] = 0
        dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
        Lp = (Lsample + Lstd) / 2
        Cp = (Cpstd + Cpsample) / 2
        hp = (hpstd + hpsample) / 2
        hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
        hp = hp + (hp < 0) * 2 * np.pi
        hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
        Lpm502 = np.power((Lp - 50), 2)
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
            0.32 * np.cos(3 * hp + np.pi / 30) \
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
        Sh = 1 + 0.015 * Cp * T
        delthetarad = (30 * np.pi / 180) * np.exp(
            - np.power((180 / np.pi * hp - 275) / 25, 2))
        Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
        RT = - np.sin(2 * delthetarad) * Rc
        klSl = self.kl * Sl
        kcSc = self.kc * Sc
        khSh = self.kh * Sh
        de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                       np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))

        return np.sum(de00) / (np.shape(de00)[0] - self.color_chart_area)
