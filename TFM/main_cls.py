# Standard library imports
import argparse
import builtins
import copy
import glob
import json
import math
import os
import pdb
import random
import shutil
import time
from typing import Union, List, Optional, Callable

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from PIL import Image
from torch import autograd
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Third-party loss/metrics
from pytorch_losses import gradient_loss
from pytorch_ssim import SSIM as compute_SSIM_loss

# wandb import
import wandb

# Local application imports
from utils import (
    AverageMeter, ProgressMeter, init_ema_model, update_ema_model,
    MIT_Dataset, affine_crop_resize, multi_affine_crop_resize, MIT_Dataset_PreLoad, IIW2
)
from unets import UNet
from model_utils import (
    plot_relight_img_train, compute_logdet_loss, intrinsic_loss, save_checkpoint
)

# - visualization imports
from visu_utils import *
from eval_utils import get_eval_relight_dataloder

# - dataset imports
from dataset_utils import *



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
parser.add_argument("--intrinsics_loss_weight", type=float, default=1e-1)
parser.add_argument("--reg_weight", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--weight_decay", type=float, default=0)
# training params
parser.add_argument("--gpus", type=int, default=1)
# datamodule params
parser.add_argument("--data_path", type=str, default=".")


# WandB arguments
parser.add_argument("--wandb_project", type=str, default="latent-intrinsics-relight", 
                    help="WandB project name")
parser.add_argument("--wandb_run_name", type=str, default=None,
                    help="WandB run name (if None, will be auto-generated)")
parser.add_argument("--disable_wandb", action='store_true',
                    help="Disable wandb logging")

parser.add_argument('--log_freq', default=10, type=int,
                    help='Frequency of logging to wandb (default: 10)')

# Path to save visu
parser.add_argument("--visu_path", type=str, default=".", help="Path to save visualization images")

# Depth params
parser.add_argument("--depth_loss_weight", type=float, default=1e-2,
                    help="Weight for depth consistency loss")
parser.add_argument("--depth_model_name", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="Depth Anything model variant")
parser.add_argument("--enable_depth_loss", action='store_true',
                    help="Enable depth consistency loss")

# Dataset params
parser.add_argument('--dataset', default='mit', choices=['mit', 'rsr_256', 'iiw'], 
                    help='Dataset type to use for training')

# arguments for RelightingDataset
parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of scenes to use for testing (default: 0.2)')
parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible train/test splits (default: 42)')


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
    save_folder_path = '''checkpoint/intrinsics_loss_weight_{}_reg_weight_{}_lr_{}_batch_size_{}_weight_decay_{}_affine_scale_{}'''.replace('\n',' ').replace(' ','').format(
                        args.intrinsics_loss_weight, args.reg_weight, args.learning_rate, args.batch_size, args.weight_decay, args.affine_scale)
    args.save_folder_path = save_folder_path
    args.is_master = args.rank == 0

    # Apply visualization configuration
    from visu_utils import apply_visualization_config
    args = apply_visualization_config(args, getattr(args, 'viz_config', 'standard'))

    # Initialize WandB with depth loss config
    if args.is_master and not args.disable_wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"il{args.intrinsics_loss_weight}_rw{args.reg_weight}_lr{args.learning_rate}_bs{args.batch_size}_wd{args.weight_decay}_as{args.affine_scale}"
            if args.enable_depth_loss:
                args.wandb_run_name += f"_dl{args.depth_loss_weight}"
        
        config = {
            "learning_rate": args.learning_rate,
            "intrinsics_loss_weight": args.intrinsics_loss_weight,
            "reg_weight": args.reg_weight,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "affine_scale": args.affine_scale,
            "img_size": args.img_size,
            "epochs": args.epochs,
            "architecture": "UNet",
            "model_channels": 32,
            "channel_mult": [1, 2, 4, 4, 8, 16],
            "num_blocks_list": [1, 2, 2, 4, 4, 4],
            "enable_depth_loss": args.enable_depth_loss,
            # Add visualization config to wandb
            "viz_log_freq": getattr(args, 'viz_log_freq', 50),
            "max_viz_images": getattr(args, 'max_viz_images', 4),
        }

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            resume="allow" if args.resume else None
        )

    
    model, ema_model, optimizer = init_model(args)
    torch.cuda.set_device(args.gpu)

    optimizer = AdamW(model.parameters(),
                lr= args.learning_rate, weight_decay = args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    args.start_epoch = 0
    if args.resume:
        args.resume = '{}/last.pth.tar'.format(save_folder_path)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    transform_train = [affine_crop_resize(size = (256, 256), scale = (0.2, 1.0)),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])]

    # Validation transform (no augmentation)
    transform_val = [None, transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])]

    # UPDATED: Dataset initialization section
    if args.dataset == 'mit':
        #train_dataset = MIT_Dataset(args.data_path, transform_train)
        train_dataset = MIT_Dataset_PreLoad(args.data_path, transform_train, total_split = args.world_size, split_id = args.rank)
        # Create validation dataset
        val_dataset = MIT_Dataset('/home/apinyol/TFM/Data/multi_illumination_test_mip2_jpg', transform_val, eval_mode = True,)

    elif args.dataset == 'iiw':
        # Updated to use the new RelightingDataset with proper parameters
        train_dataset = IIW2(
            root=args.data_path,  # Path to IIW images
            img_transform=transform_train,
            split='train',  # Training split
            return_two_images=True
        )
        # Validation dataset
        val_dataset = IIW2(
            root=args.data_path,
            img_transform=transform_val,
            split='val',  # Validation split
            return_two_images=True
        )

    elif args.dataset == 'rsr_256':
        # Updated to use the new RelightingDataset with proper parameters
        train_dataset = RSRDataset(args.data_path, transform_train, total_split = args.world_size, split_id = args.rank)
        # For RSR, you might want to create a separate validation split or use a subset
        val_dataset = RSRDataset(args.data_path, transform_val, total_split = args.world_size, split_id = args.rank)

    print('NUM of training images: {}'.format(len(train_dataset)))
    print('NUM of validation images: {}'.format(len(val_dataset)))
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True, drop_last = True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle = False, drop_last = False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_sampler = None  # You had this line, keeping it
    val_sampler = None    # Corresponding line for validation
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last = True, persistent_workers = True)
    
    # Validation loader (typically smaller batch size)
    val_batch_size = getattr(args, 'val_batch_size', args.batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last = False, persistent_workers = True)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.is_master):
        if not os.path.exists(save_folder_path):
            os.system('mkdir -p {}'.format(save_folder_path))

    # Training loop with validation
    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Training
        last_training_step = train_D(train_loader, model, scaler, optimizer, ema_model, epoch, args)
        
        # Validation (configurable frequency)
        val_freq = getattr(args, 'validation_freq', 1)  # Default: validate every 5 epochs
        if epoch % val_freq == 0 or epoch == args.start_epoch:
            print(f"Running validation at epoch {epoch}")
            val_metrics = validate_D(val_loader, model, epoch, args, current_step=last_training_step)
            
            # Log validation summary
            if args.is_master and not args.disable_wandb and val_metrics:
                print(f"Validation metrics at epoch {epoch}: {val_metrics}")
        
        # Checkpointing
        if args.is_master:
            if epoch % 2 == 0 and epoch != 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'scaler': scaler.state_dict(),
                }, False, filename = '{}/last.pth.tar'.format(save_folder_path))
            if epoch % 20 == 0 and epoch != 0:
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'scaler': scaler.state_dict(),
                },False , filename = '{}/{}.pth.tar'.format(save_folder_path, epoch))

    # Finish wandb run
    if args.is_master and not args.disable_wandb:
        wandb.finish()

def print_gradients(model):
    max_grad = 0
    max_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.norm())
            max_norm = max(max_norm, p.data.norm(2))
    return max_grad, max_norm

def train_D(train_loader, model, scaler, optimizer, ema_model, epoch, args):
    loss_name = [
                'loss','logdet', 'light_logdet', 'intrinsic_sim',
                'GPU Mem', 'Time', 'pe', 'ge']
    moco_loss_meter = [AverageMeter(name, ':6.3f') for name in loss_name]
    progress = ProgressMeter(
        len(train_loader),
        moco_loss_meter,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    t0 = time.time()
    P_mean=-1.2
    P_std=1.2
    sigma_data = 0.5

    logdet_loss = compute_logdet_loss()
    ssim_loss = compute_SSIM_loss()

    # Import visualization utilities
    from visu_utils import log_training_visualizations

    # Global step counter for wandb logging
    global_step = epoch * len(train_loader)

    for i, (input_img, ref_img) in enumerate(train_loader):
        current_step = global_step + i

        input_img = input_img.to(args.gpu)
        ref_img = ref_img.to(args.gpu)

        rnd_normal = torch.randn([input_img.shape[0], 1, 1, 1], device=input_img.device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        if epoch >= 60:
            sigma = sigma * 0

        noisy_input_img = input_img + torch.randn_like(input_img) * sigma
        noisy_ref_img = ref_img + torch.randn_like(ref_img) * sigma

        with torch.cuda.amp.autocast():
            intrinsic_input, extrinsic_input = model(noisy_input_img, run_encoder = True)
            intrinsic_ref, extrinsic_ref = model(noisy_ref_img, run_encoder = True)

            mask = (torch.rand(input_img.shape[0]) > 0.9).float().to(args.gpu).reshape(-1,1,1,1).float()
            intrinsic = [i_input * mask + i_ref * (1 - mask) for i_input, i_ref in zip(intrinsic_input, intrinsic_ref)]

            relit_img = model([intrinsic_input, extrinsic_ref], run_encoder = False).float()

        
        logdet_pred, logdet_target = logdet_loss(intrinsic_input)
        logdet_pred_ext, logdet_target_ext = logdet_loss([extrinsic_input])
        sim_intrinsic = intrinsic_loss(intrinsic_input, intrinsic_ref)
        rec_loss = nn.MSELoss()(relit_img,input_img)
        ssim_component = 0.1 * (1 - ssim_loss(relit_img,input_img))
        grad_component = gradient_loss(relit_img,input_img)

        rec_loss_total = 10 * rec_loss + ssim_component + grad_component
        logdet_reg_loss = args.reg_weight * ((logdet_pred - logdet_target) ** 2).mean()
        logdet_ext_reg_loss = args.reg_weight * ((logdet_pred_ext - logdet_target_ext) ** 2).mean()
        intrinsic_loss_component = - args.intrinsics_loss_weight * sim_intrinsic

        loss = rec_loss_total + logdet_reg_loss + logdet_ext_reg_loss + intrinsic_loss_component

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        ge, pe = print_gradients(model)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        update_ema_model(model, ema_model, 0.999)

        t1 = time.time()
        # Collect metrics including depth loss
        metrics = {
            'total_loss': loss.item(),
            'reconstruction_loss': rec_loss.item(),
            'rec_loss_mse': rec_loss.item(),
            'rec_loss_ssim': ssim_component.item() if isinstance(ssim_component, torch.Tensor) else ssim_component,
            'rec_loss_gradient': grad_component.item() if isinstance(grad_component, torch.Tensor) else grad_component,
            'logdet_loss': logdet_pred[:-1].mean().item(),
            'light_logdet_loss': logdet_pred[-1].item(),
            'logdet_reg_loss': logdet_reg_loss.item(),
            'logdet_ext_reg_loss': logdet_ext_reg_loss.item(),
            'intrinsic_similarity': sim_intrinsic.item(),
            'intrinsic_loss_component': intrinsic_loss_component.item(),
            'sigma_noise': sigma.mean().item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_norm': ge.item() if isinstance(ge, torch.Tensor) else ge,
            'parameter_norm': pe.item() if isinstance(pe, torch.Tensor) else pe,
            'gpu_memory_mb': torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            'step_time': t1 - t0,
            'epoch': epoch,
        }

        # Log training visualizations (configurable frequency)
        if args.is_master and not args.disable_wandb:
            if i % args.log_freq == 0 or i == 0:
                wandb.log(metrics, step=current_step)
            
            # Log visualizations less frequently to avoid overwhelming wandb
            viz_freq = getattr(args, 'viz_log_freq', args.log_freq * 10)  # Default: 10x less frequent than metrics
            if i % viz_freq == 0 or i == 0:
                log_training_visualizations(
                    input_img=input_img,
                    ref_img=ref_img,
                    relit_img=relit_img,
                    noisy_input_img=noisy_input_img,
                    noisy_ref_img=noisy_ref_img,
                    step=current_step,
                    mode='train',
                    max_images=getattr(args, 'max_viz_images', 4)  # Configurable number of images to show
                )
            
        for val_id, val in enumerate([rec_loss, logdet_pred[:-1].mean(), logdet_pred[-1], sim_intrinsic,
                        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), t1 - t0, pe, ge
                    ]):
            if not isinstance(val, float) and not isinstance(val, int):
                val = val.item()
            moco_loss_meter[val_id].update(val)
        progress.display(i)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    if args.gpu == 0 and epoch % 5 == 0:
        target_img = ref_img[torch.randperm(input_img.shape[0]).to(args.gpu)]
        plot_relight_img_train(model, input_img, ref_img, target_img, args.save_folder_path + '/{:05d}_{:05d}_gen'.format(epoch + 1, i))

    torch.distributed.barrier()
    return current_step


def validate_D(val_loader, model, epoch, args, current_step=None):
    """Validation function with visualizations"""
    from visu_utils import log_relighting_results
    
    model.eval()
    val_metrics = {}
    
    # Global step counter for wandb logging
    if current_step is None:
        current_step = epoch * len(val_loader)

    with torch.no_grad():
        for i, (input_img, ref_img, gt_img) in enumerate(val_loader):
            val_step = current_step + i
            if i >= getattr(args, 'max_val_batches', 5):  # Limit validation batches for speed
                break
                
            input_img = input_img.to(args.gpu)
            ref_img = ref_img.to(args.gpu)
            gt_img = gt_img.to(args.gpu)
            
            # Forward pass without noise for validation
            intrinsic_input, extrinsic_input = model(input_img, run_encoder=True)
            intrinsic_ref, extrinsic_ref = model(ref_img, run_encoder=True)
            
            # Simple relighting for validation
            relit_img = model([intrinsic_input, extrinsic_input], run_encoder=False).float()
            
            # Compute validation metrics
            val_loss = nn.MSELoss()(relit_img, input_img)
            
            if i == 0:  # Log metrics from first batch
                val_metrics = {
                    'val_reconstruction_loss': val_loss.item(),
                    'val_epoch': epoch,
                }
            
            # Log validation visualizations
            if args.is_master and not args.disable_wandb and i in [0, 1]:  # Only first and second batch 
                log_relighting_results(
                    #model,
                    input_img=input_img,
                    ref_img=ref_img,
                    relit_img=relit_img,
                    gt_img=gt_img,
                    step=val_step,
                    mode='validation'
                )
                
                if val_metrics:
                    wandb.log(val_metrics, step=val_step)
    
    model.train()
    return val_metrics

if __name__ == '__main__':
    main()
