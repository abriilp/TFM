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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from utils import *
#(
#    AverageMeter, ProgressMeter, init_ema_model, update_ema_model,
#    MIT_Dataset, affine_crop_resize, multi_affine_crop_resize, MIT_Dataset_PreLoad, IIW2
#)
from unets import UNet
from model_utils import (
    plot_relight_img_train, compute_logdet_loss, intrinsic_loss, save_checkpoint
)

# - visualization imports
from visu_utils import *
from eval_utils import get_eval_relight_dataloder

# - dataset imports
from dataset_utils import *

# model visualization
import torchvision.models as models
from torchviz import make_dot

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

# Dataset params
parser.add_argument('--dataset', default='mit', choices=['mit', 'rsr_256', 'iiw'], 
                    help='Dataset type to use for training')

# arguments for RelightingDataset
parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of scenes to use for testing (default: 0.2)')
parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible train/test splits (default: 42)')

# Multi-dataset validation arguments
#parser.add_argument('--validation_datasets', nargs='+', default=['mit', 'rsr_256'],
#                    choices=['mit', 'rsr_256', 'iiw'],
#                    help='List of datasets to validate on (default: mit rsr_256)')

# arguments for better checkpoint management
parser.add_argument("--experiment_name", type=str, default="intrinsics_experiment",
                    help="Base name for the experiment")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                    help="Base directory for saving checkpoints")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Specific checkpoint path to resume from (overrides auto-resume)")
parser.add_argument("--auto_resume", action='store_true',
                    help="Automatically resume from latest checkpoint in experiment")
parser.add_argument("--save_every", type=int, default=5,
                    help="Save checkpoint every N epochs")
parser.add_argument("--keep_last", type=int, default=3,
                    help="Keep only the last N checkpoints (0 to keep all)")

# Depth params
parser.add_argument("--depth_loss_weight", type=float, default=1e-2,
                    help="Weight for depth consistency loss")
parser.add_argument("--depth_model_name", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="Depth Anything model variant")
parser.add_argument("--enable_depth_loss", action='store_true',
                    help="Enable depth consistency loss")

args = parser.parse_args()


def init_model(args):
    model = UNet(img_resolution = 256, in_channels = 3, out_channels = 3,
                     num_blocks_list = [1, 2, 2, 4, 4, 4], attn_resolutions = [0], model_channels = 32,
                     channel_mult = [1, 2, 4, 4, 8, 16], affine_scale = float(args.affine_scale))
    print("Model architecture:", model)
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
    
    depth_loss_fn = None
    if args.enable_depth_loss:
        depth_loss_fn = DepthConsistencyLoss(
            model_name=args.depth_model_name, 
            device=args.gpu
        )

    return model, ema_model, optimizer, depth_loss_fn

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

    # Create or find experiment folder
    if args.resume or args.auto_resume or args.resume_from:
        experiment_path, checkpoint_path = find_experiment_folder(args)
        if experiment_path is None:
            if args.is_master:
                print("No checkpoint found for resuming, starting new experiment")
                experiment_path = create_experiment_folder(args)
                checkpoint_path = None
    else:
        if args.is_master:
            experiment_path = create_experiment_folder(args)
            checkpoint_path = None

    args.save_folder_path = experiment_path
    
    # Apply visualization configuration
    from visu_utils import apply_visualization_config
    args = apply_visualization_config(args, getattr(args, 'viz_config', 'standard'))

    # Initialize WandB
    if args.is_master and not args.disable_wandb:
        if args.wandb_run_name is None:
            experiment_name = os.path.basename(experiment_path)
            args.wandb_run_name = experiment_name
            if args.enable_depth_loss:
                args.wandb_run_name += f"_dl{args.depth_loss_weight}"
        
        config = {
            "experiment_path": experiment_path,
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
            "viz_log_freq": getattr(args, 'viz_log_freq', 50),
            "max_viz_images": getattr(args, 'max_viz_images', 4),
            "validation_datasets": getattr(args, 'validation_datasets', ['mit', 'rsr_256']),
            "training_dataset": args.dataset,
            "enable_depth_loss": args.enable_depth_loss,
        }

        if args.enable_depth_loss:
            config["depth_loss_weight"] = args.depth_loss_weight
            config["depth_model_name"] = args.depth_model_name

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            resume="allow" if checkpoint_path else None
        )

    model, ema_model, optimizer, depth_loss_fn = init_model(args)
    torch.cuda.set_device(args.gpu)

    optimizer = AdamW(model.parameters(),
                lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if available
    args.start_epoch = 0
    if checkpoint_path:
        args.start_epoch = load_checkpoint_improved(
            checkpoint_path, model, ema_model, optimizer, scaler, args
        )

    # Initialize datasets and loaders (keeping your existing code)
    transform_train = [affine_crop_resize(size=(256, 256), scale=(0.2, 1.0)),
                       transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                       ])]

    transform_val = [None, transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])]

    # Dataset initialization (keeping your existing logic)
    if args.dataset == 'mit':
        train_dataset = MIT_Dataset_PreLoad(args.data_path, transform_train, 
                                           total_split=args.world_size, split_id=args.rank)
        val_dataset = MIT_Dataset('/home/apinyol/TFM/Data/multi_illumination_test_mip2_jpg', 
                                 transform_val, eval_mode=True)
    elif args.dataset == 'iiw':
        train_dataset = IIW2(root=args.data_path, img_transform=transform_train, 
                           split='train', return_two_images=True)
        val_dataset = IIW2(root=args.data_path, img_transform=transform_val, 
                          split='val', return_two_images=True)
    elif args.dataset == 'rsr_256':
        train_dataset = RSRDataset(root_dir=args.data_path, img_transform=transform_train, 
                                 is_validation=False)
        val_dataset = RSRDataset(root_dir=args.data_path, img_transform=transform_val, 
                                is_validation=True)

    print(f'NUM of training images: {len(train_dataset)}')
    print(f'NUM of validation images: {len(val_dataset)}')
    
    validation_datasets = create_validation_datasets(args, transform_val)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_sampler = None
    val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, 
        drop_last=True, persistent_workers=True)
    
    validation_loaders = create_validation_loaders(validation_datasets, args, val_sampler)

    global_step =  args.start_epoch * len(train_loader)

    # Training loop with improved checkpointing
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Training
        last_training_step, global_step = train_D(train_loader, model, scaler, optimizer, ema_model, epoch, args, global_step, depth_loss_fn)
        
        # Validation
        val_freq = getattr(args, 'validation_freq', 1)
        if epoch % val_freq == 0 or epoch == args.start_epoch:
            print(f"Running multi-dataset validation at epoch {epoch}")
            all_val_metrics, global_step = validate_multi_datasets(
                validation_loaders, model, epoch, args, current_step=global_step)
            
            if args.is_master and not args.disable_wandb and all_val_metrics:
                print(f"Multi-dataset validation metrics at epoch {epoch}:")
                for dataset_name, metrics in all_val_metrics.items():
                    print(f"  {dataset_name}: {metrics}")
        
        # Improved checkpointing
        if args.is_master:
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'scaler': scaler.state_dict(),
                'args': args,
                'global_step': global_step,
            }
            
            # Save checkpoint based on save_every parameter
            if epoch % args.save_every == 0 or epoch == args.epochs + args.start_epoch - 1:
                print(f"Saving checkpoint at epoch {epoch + 1} to {experiment_path}")
                save_checkpoint_improved(
                    checkpoint_state, 
                    experiment_path, 
                    epoch + 1, 
                    is_best=False,  # You can add logic to determine if it's the best
                    keep_last=args.keep_last
                )

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

def train_D(train_loader, model, scaler, optimizer, ema_model, epoch, args, global_step, depth_loss_fn=None):
    """FIXED: Modified to accept and return global_step"""
    # FIXED: Added depth_loss to the loss_name list
    loss_name = [
        'loss', 'logdet', 'light_logdet', 'intrinsic_sim', 'depth_loss',
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
    logdet_loss_ext = compute_logdet_loss()
    ssim_loss = compute_SSIM_loss()

    # Import visualization utilities
    from visu_utils import log_training_visualizations

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

            """mask = (torch.rand(input_img.shape[0]) > 0.9).float().to(args.gpu).reshape(-1,1,1,1).float()
            intrinsic = [i_input * mask + i_ref * (1 - mask) for i_input, i_ref in zip(intrinsic_input, intrinsic_ref)]"""

            recon_img_rr = model([intrinsic_ref, extrinsic_ref], run_encoder = False).float()
            recon_img_ir = model([intrinsic_input, extrinsic_ref], run_encoder = False).float()
            recon_img_ri = model([intrinsic_ref, extrinsic_input], run_encoder = False).float()
            recon_img_ii = model([intrinsic_input, extrinsic_input], run_encoder = False).float()


        
        logdet_pred, logdet_target = logdet_loss(intrinsic_input)
       
        logdet_pred_ext, logdet_target_ext = logdet_loss([extrinsic_input])


        sim_intrinsic = intrinsic_loss(intrinsic_input, intrinsic_ref)

        rec_loss_rr = nn.MSELoss()(recon_img_rr, ref_img)
        ssim_component_rr = 0.1 * (1 - ssim_loss(recon_img_rr, ref_img))
        grad_component_rr = gradient_loss(recon_img_rr, ref_img)

        rec_loss_ri = nn.MSELoss()(recon_img_ri, input_img)
        ssim_component_ri = 0.1 * (1 - ssim_loss(recon_img_ri, input_img))
        grad_component_ri = gradient_loss(recon_img_ri ,input_img)

        rec_loss_ir = nn.MSELoss()(recon_img_ir,ref_img)
        ssim_component_ir = 0.1 * (1 - ssim_loss(recon_img_ir,ref_img))
        grad_component_ir= gradient_loss(recon_img_ir,ref_img)

        rec_loss_ii = nn.MSELoss()(recon_img_ii,input_img)
        ssim_component_ii = 0.1 * (1 - ssim_loss(recon_img_ii,input_img))
        grad_component_ii = gradient_loss(recon_img_ii,input_img)

        # Compute depth consistency loss
        depth_loss_value = torch.tensor(0.0, device=input_img.device)  # FIXED: Initialize as tensor
        depth_input_norm = None
        depth_recon_norm = None
        
        if args.enable_depth_loss and depth_loss_fn is not None:
            # Only compute depth loss every few iterations to save computation
            if i % 5 == 0:  # Compute every 5 iterations
                try:
                    depth_loss_value, depth_input_norm, depth_recon_norm = depth_loss_fn(input_img, recon_img_ri)
                    # FIXED: Ensure depth_loss_value is a tensor
                    if not isinstance(depth_loss_value, torch.Tensor):
                        depth_loss_value = torch.tensor(depth_loss_value, device=input_img.device)
                except Exception as e:
                    print(f"Warning: Depth loss computation failed: {e}")
                    depth_loss_value = torch.tensor(0.0, device=input_img.device)

        rec_loss_total_rr = 10 * rec_loss_rr + ssim_component_rr + grad_component_rr
        rec_loss_total_ri = 10 * rec_loss_ri + ssim_component_ri + grad_component_ri
        rec_loss_total_ir = 10 * rec_loss_ir + ssim_component_ir + grad_component_ir
        rec_loss_total_ii = 10 * rec_loss_ii + ssim_component_ii + grad_component_ii
        rec_loss_total = (rec_loss_total_rr + rec_loss_total_ri + rec_loss_total_ir + rec_loss_total_ii) / 4.0

        logdet_reg_loss = args.reg_weight * ((logdet_pred - logdet_target) ** 2).mean()
        logdet_ext_reg_loss = args.reg_weight * ((logdet_pred_ext - logdet_target_ext) ** 2).mean()
        intrinsic_loss_component = args.intrinsics_loss_weight * sim_intrinsic
        
        # FIXED: Ensure depth_loss_weight exists and handle tensor conversion
        depth_loss_weight = getattr(args, 'depth_loss_weight', 0.0)
        depth_loss_component = depth_loss_weight * depth_loss_value

        loss = rec_loss_total + logdet_reg_loss + intrinsic_loss_component + depth_loss_component + logdet_ext_reg_loss

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
            'rec_loss_total1': rec_loss_total.item(),
            'logdet_reg_loss2': logdet_reg_loss.item(),
            'logdet_ext_reg_loss3': logdet_ext_reg_loss.item(),
            'intrinsic_loss_component4': intrinsic_loss_component.item(),
            'depth_loss_component5': depth_loss_component.item() if isinstance(depth_loss_component, torch.Tensor) else depth_loss_component,
            'intrinsic_similarity': sim_intrinsic.item(),
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
            """if i % viz_freq == 0 or i == 0:
                log_training_visualizations(
                    input_img=input_img,
                    ref_img=ref_img,
                    recon_img=recon_img_ri,
                    noisy_input_img=noisy_input_img,
                    noisy_ref_img=noisy_ref_img,
                    step=current_step,
                    mode='train',
                    max_images=getattr(args, 'max_viz_images', 4)  # Configurable number of images to show
                )"""

            """# Log depth maps occasionally for visualization
            if args.enable_depth_loss and depth_input_norm is not None and i % 50 == 0:
                depth_vis = {
                    "depth_input": wandb.Image(depth_input_norm[0, 0].cpu().numpy(), caption="Input Depth"),
                    "depth_recon": wandb.Image(depth_recon_norm[0, 0].cpu().numpy(), caption="Reconstructed Depth"),
                }
                wandb.log(depth_vis, step=current_step)"""
            
        # FIXED: Update the moco_loss_meter with the correct number of values matching loss_name
        loss_values = [
            rec_loss_total.item(), 
            logdet_pred[:-1].mean(), 
            logdet_pred[-1], 
            sim_intrinsic, 
            depth_loss_value,  # Now included in loss_name
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 
            t1 - t0, 
            pe, 
            ge
        ]
        
        for val_id, val in enumerate(loss_values):
            if not isinstance(val, float) and not isinstance(val, int):
                val = val.item()
            moco_loss_meter[val_id].update(val)
        progress.display(i)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    """if args.gpu == 0 and epoch % 5 == 0:
        target_img = ref_img[torch.randperm(input_img.shape[0]).to(args.gpu)]
        plot_relight_img_train(model, input_img, ref_img, target_img, args.save_folder_path + '/{:05d}_{:05d}_gen'.format(epoch + 1, i))
    ARREGLAR,RuntimeError: shape '[2, 2, 3, 256, 256]' is invalid for input of size 3145728 """
    torch.distributed.barrier()
    
    # FIXED: Return the updated global step
    final_step = global_step + len(train_loader)
    return current_step, final_step


def validate_D(val_loader, model, epoch, args, current_step=None, dataset_name='validation', depth_loss_fn=None):
    """Validation function with simplified depth visualizations"""
    from visu_utils import log_relighting_results
    
    model.eval()
    val_metrics = {}
    
    if current_step is None:
        current_step = 0

    with torch.no_grad():
        print(f"Starting validation on {dataset_name}...")
        print(f"Len Validation loader {len(val_loader)}")
        
        for i, batch in enumerate(val_loader):
            print(f"Validation batch {i}/{len(val_loader)} for {dataset_name}")
            val_step = current_step + i
            
            if i >= getattr(args, 'max_val_batches', 2):
                break
            
            # Handle 3-image validation batch (input, ref, gt)
            if len(batch) == 3:
                input_img, ref_img, gt_img = batch
            else:
                input_img, ref_img = batch
                gt_img = ref_img
                
            input_img = input_img.to(args.gpu)
            ref_img = ref_img.to(args.gpu)
            gt_img = gt_img.to(args.gpu)
            
            # Forward pass without noise for validation
            intrinsic_input, extrinsic_input = model(input_img, run_encoder=True)
            intrinsic_ref, extrinsic_ref = model(ref_img, run_encoder=True)
            
            # Proper relighting for validation
            relit_img = model([intrinsic_input, extrinsic_ref], run_encoder=False).float()

            # Bias correction
            shift = (relit_img.clamp(-1,1) - gt_img).mean(dim=[2, 3], keepdim=True)
            correct_relit_img = relit_img.clamp(-1,1) - shift
            
            # Compute validation metrics against ground truth
            val_reconstruction_loss = nn.MSELoss()(relit_img, gt_img)
            
            if i == 0:
                val_metrics = {
                    'val_reconstruction_loss_vs_gt': val_reconstruction_loss.item(),
                    'val_epoch': epoch,
                }
            
            # Log visualizations for first batch only
            if args.is_master and not args.disable_wandb and i == 0:
                # Standard relighting visualization
                log_relighting_results(
                    input_img=input_img,
                    ref_img=ref_img,
                    relit_img=relit_img,
                    gt_img=gt_img,
                    step=val_step,
                    mode=f'{dataset_name}_validation'
                )

                # Depth visualization if enabled
                if args.enable_depth_loss and depth_loss_fn is not None:
                    try:
                        # Get depth maps
                        depth_loss_value, input_depth, relit_depth = depth_loss_fn(input_img, correct_relit_img)
                        
                        # Create simple visualization
                        depth_vis = depth_loss_fn.create_simple_visualization(
                            input_img, correct_relit_img, input_depth, relit_depth, 
                            epoch, dataset_name
                        )
                        
                        if depth_vis is not None:
                            wandb.log({
                                f'{dataset_name}_depth_comparison': wandb.Image(
                                    depth_vis, 
                                    caption=f"Depth Comparison - {dataset_name} Epoch {epoch}"
                                )
                            }, step=val_step)
                        
                        # Add depth loss to metrics
                        val_metrics[f'val_depth_loss'] = depth_loss_value.item()
                        
                    except Exception as e:
                        print(f"Warning: Depth analysis failed for {dataset_name}: {e}")
                
                # Log all metrics
                if val_metrics:
                    wandb.log(val_metrics, step=val_step)
    
    model.train()
    
    next_step = current_step + min(len(val_loader), getattr(args, 'max_val_batches', 2))
    return val_metrics, next_step


def validate_multi_datasets(validation_loaders, model, epoch, args, current_step=None):
    """Multi-dataset validation with proper step tracking"""
    all_val_metrics = {}
    
    if current_step is None:
        current_step = 0
    
    # Initialize depth loss function if needed
    depth_loss_fn = None
    if args.enable_depth_loss:
        try:
            depth_loss_fn = DepthConsistencyLoss(
                model_name=getattr(args, 'depth_model_name', "depth-anything/Depth-Anything-V2-Small-hf"),
                device=args.gpu
            )
        except Exception as e:
            print(f"Warning: Could not initialize depth loss function: {e}")
    
    for dataset_name, val_loader in validation_loaders.items():
        print(f"Validating on {dataset_name} dataset...")
        
        val_metrics, current_step = validate_D(
            val_loader, model, epoch, args, 
            current_step=current_step, 
            dataset_name=dataset_name,
            depth_loss_fn=depth_loss_fn
        )
        
        # Add dataset prefix to metrics
        prefixed_metrics = {f"{dataset_name}_{k}": v for k, v in val_metrics.items()}
        all_val_metrics[dataset_name] = prefixed_metrics
        
        # Log dataset-specific metrics
        if args.is_master and not args.disable_wandb:
            wandb.log(prefixed_metrics, step=current_step-1)
    
    return all_val_metrics, current_step


if __name__ == '__main__':
    main()