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
from utils import MIT_Dataset, affine_crop_resize, multi_affine_crop_resize, MIT_Dataset_PreLoad
from unets import UNet
import copy
from pytorch_ssim import SSIM as compute_SSIM_loss
from pytorch_losses import gradient_loss
from model_utils import plot_relight_img_train, compute_logdet_loss, intrinsic_loss,save_checkpoint

#from sklearn.metrics import average_precision_score
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
parser.add_argument('--resume', action = 'store_true',
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
parser.add_argument("--load_ckpt", type=str, default=".")

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
    save_folder_path = 'relight_result' 
    args.save_folder_path = save_folder_path
    if not os.path.exists(save_folder_path):
        os.system('mkdir -p {}'.format(save_folder_path))

    args.is_master = args.rank == 0

    model, ema_model, optimizer = init_model(args)
    torch.cuda.set_device(args.gpu)

    args.start_epoch = 0
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
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_ckpt, checkpoint['epoch']))
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'!!!!!!!".format(args.load_ckpt))

    from eval_utils import eval_relight
    for i in range(12):
        eval_relight(args, 100, model)
    exit()

if __name__ == '__main__':
    main()
