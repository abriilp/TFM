import argparse
import os, json
import random
import shutil
import time, glob, copy
import os
import time
import torch
import argparse
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
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse, pdb
import numpy as np
from torch import autograd
from torch.optim import AdamW
from typing import Union, List, Optional, Callable
import pdb
from utils import  AverageMeter, ProgressMeter, init_ema_model, update_ema_model
import builtins
from PIL import Image
import torchvision
import tqdm
from utils import MIT_Dataset, affine_crop_resize, multi_affine_crop_resize, IIW,apply_affine
from unets import UNet
import copy
from pytorch_ssim import SSIM as compute_SSIM_loss
from pytorch_losses import gradient_loss
from model_utils import plot_relight_img_train, compute_logdet_loss, intrinsic_loss,save_checkpoint

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--img_size', default=224, type=int,
                    help='img size')
parser.add_argument('--affine_scale', default=5e-3, type=float)
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=5, type=int,
                     help='print frequency (default: 10)')
parser.add_argument('--load_ckpt', default= '', type = str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--weight_decay", type=float, default=0)
# training params
parser.add_argument("--gpus", type=int, default=1)
# datamodule params
parser.add_argument("--data_path", type=str, default=".")

args = parser.parse_args()


def init_model(args):
    model = UNet(img_resolution = 256, in_channels = 3, out_channels = 3,
                     num_blocks_list = [1, 2, 2, 4, 4, 4], attn_resolutions = [0], model_channels = 32,
                     channel_mult = [1, 2, 4, 4, 8, 16], affine_scale = float(args.affine_scale))
    model.cuda(args.gpu)
    ema_model = copy.deepcopy(model)
    ema_model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    init_ema_model(model, ema_model)
    optimizer = AdamW(model.parameters(),
                lr= args.learning_rate, weight_decay = args.weight_decay)
    return model, ema_model, optimizer

def main():
    torch.manual_seed(2)
    import os
    #torch.backends.cudnn.benchmark=False
    cudnn.deterministic = True
    args = parser.parse_args()
    #assert args.batch_size % args.batch_iter == 0
    if not os.path.exists('visualize'):
        os.system('mkdir visualize')
    if not os.path.exists('checkpoint'):
        os.system('mkdir checkpoint')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('start')
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.gpu = args.gpu % torch.cuda.device_count()
    print('world_size', args.world_size)
    print('rank', args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args = copy.deepcopy(args)
    args.cos = True
    args.is_master = args.rank == 0

    model, ema_model, optimizer = init_model(args)
    torch.cuda.set_device(args.gpu)

    optimizer = AdamW(model.parameters(),
                lr= args.learning_rate, weight_decay = args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    if os.path.isfile(args.load_ckpt):
        print("=> loading checkpoint '{}'".format(args.load_ckpt))
        if args.gpu is None:
            checkpoint = torch.load(args.load_ckpt)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.load_ckpt, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_ckpt, checkpoint['epoch']))
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'!!!!!!!".format(args.load_ckpt))

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])

    iiw_dataset = IIW(args.data_path, img_transform = img_transform, split = 'val')

    if args.distributed:
        img_sampler = torch.utils.data.distributed.DistributedSampler(iiw_dataset, shuffle = False, drop_last = False)
    else:
        img_sampler = None

    img_loader = torch.utils.data.DataLoader(
        iiw_dataset, batch_size=1, shuffle=(img_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=img_sampler, drop_last = False, persistent_workers = True)
    infer_albedo(img_loader, ema_model, args.start_epoch, args)

def eval_iiw(data_folder, img_folder):
    from whdr import load_image, compute_whdr
    img_list = glob.glob(img_folder + '/*.png')
    img_list.sort()
    whdr_list = []
    x_list = []
    for threshold in np.linspace(0, 0.9, 10):
        whdr = []
        print(threshold)
        for img_path in tqdm.tqdm(img_list):
            reflectance = load_image(filename=img_path, is_srgb=False)
            img_id = str(int(img_path.split('/')[-1].split('.')[0]))
            judgements = json.load(open(data_folder + img_id + '.json'))
            whdr.append(compute_whdr(reflectance, judgements, threshold))
        whdr_list.append(np.array(whdr).mean())
        print(whdr_list)
        x_list.append(threshold)
    plt.plot(np.array(x_list), np.array(whdr_list))
    plt.savefig('img2.png')
    print(np.array(whdr_list))

def create_gaussian_mask(patch_size, std):
    center_patch = 0.5 * (patch_size - 1)
    coor = torch.arange(patch_size) - center_patch
    grid = torch.stack(torch.meshgrid(coor, coor))
    grid = grid / grid.max()
    gaussian = torch.exp(-1 * ((grid ** 2).sum(dim = 0) / std ** 2))
    return gaussian

@torch.no_grad()
def infer_albedo(img_loader, model, epoch, args):
    model.module.update_affine_scale(0)
    from utils import multi_affine_crop_resize
    data_aug = multi_affine_crop_resize(scale = [0.2, 1.0], size = (256, 256))
    patch_minibatch = 10
    save_folder_path = f'iiw_out_{img_loader.dataset.split}'
    if not os.path.exists(save_folder_path):
        os.system('mkdir -p {}'.format(save_folder_path))
    for i, (raw_img, ori_img_shape, img_index) in tqdm.tqdm(enumerate(img_loader)):
        raw_img = raw_img.to(args.gpu)
        img = raw_img
        img_unaffine_list = []
        mask_unaffine_list = []
        for slide_patch in [256, min(img.shape[2], img.shape[3])]:
            slide_stride = slide_patch // 8
            affine = data_aug.get_patch_affine(img, slide_patch, slide_stride)
            img_affine = apply_affine(img.expand(affine.shape[0],-1,-1,-1), affine[:,:2].to(args.gpu), out_size = (256,256))
            mask = create_gaussian_mask(256, 0.5)[None,None,...].expand(img_affine.shape[0],-1,-1,-1).to(args.gpu)
            inverse_affine = torch.inverse(affine)
            mask_unaffine = apply_affine(mask, inverse_affine[:,:2].to(args.gpu), out_size = (img.shape[2],img.shape[3]))
            with torch.cuda.amp.autocast():
                rec_img_list = []
                for idx in range((img_affine.shape[0] + patch_minibatch - 1) // patch_minibatch):
                    intrinsic, extrinsic = model(img_affine[idx * patch_minibatch:(idx + 1)* patch_minibatch], run_encoder = True)
                    rec_img_list.append(model([intrinsic, extrinsic], run_encoder = False).float().clamp(min = -1,max = 1))
                rec_img = torch.cat(rec_img_list)
                assert rec_img.shape[0] == affine.shape[0]
            img_unaffine = apply_affine(rec_img, inverse_affine[:,:2].to(args.gpu), out_size = (img.shape[2],img.shape[3]))
            img_unaffine_list.append(img_unaffine)
            mask_unaffine_list.append(mask_unaffine)
        img_unaffine = torch.cat(img_unaffine_list)
        mask_unaffine = torch.cat(mask_unaffine_list)
        rec_img = (img_unaffine * mask_unaffine).sum(dim = 0) / mask_unaffine.sum(dim = 0)
        rec_img = ((rec_img.clamp(-1,1) * 0.5 + 0.5) * 255).cpu().data.numpy().transpose(1,2,0).astype(np.uint8)
        raw_img = ((raw_img[0].permute(1,2,0) * 0.5 + 0.5) * 255).cpu().data.numpy().astype(np.uint8)
        pil_img = Image.fromarray(rec_img)
        pil_img.save(f'{save_folder_path}/{int(img_index[0]):06d}.png')
    torch.distributed.barrier()
    if args.gpu == 0:
        eval_iiw(args.data_path, save_folder_path)
    torch.distributed.barrier()

if __name__ == '__main__':
    main()
