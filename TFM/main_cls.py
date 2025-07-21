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
parser.add_argument("--depth_loss_weight", type=float, default=1e-1,
                    help="Weight for depth consistency loss")
parser.add_argument("--depth_model_name", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="Depth Anything model variant")
parser.add_argument("--enable_depth_loss", action='store_true',
                    help="Enable depth consistency loss")

# Normals params
parser.add_argument('--enable_normal_loss', action='store_true', 
                       help='Enable normal consistency loss')
parser.add_argument('--normal_loss_weight', type=float, default=0.1,
                       help='Weight for normal consistency loss')
parser.add_argument('--normals_model_name', type=str, 
                       default='GonzaloMG/marigold-e2e-ft-normals',
                       help='Hugging Face model name for normal estimation')

# Normals decoder params
parser.add_argument('--enable_normals_decoder_loss', action='store_true', help='Enable normals decoder loss')
parser.add_argument('--normals_decoder_loss_weight', type=float, default=1.0, help='Weight for normals decoder loss')


args = parser.parse_args()


def init_model(args):
    model = UNet(img_resolution=256, in_channels=3, out_channels=3,
                 num_blocks_list=[1, 2, 2, 4, 4, 4], attn_resolutions=[0], model_channels=32,
                 channel_mult=[1, 2, 4, 4, 8, 16], affine_scale=float(args.affine_scale))
    """if args.is_master:
        wandb.watch(model, log='all', log_freq=100)"""
    model.cuda(args.gpu)
    ema_model = copy.deepcopy(model)
    ema_model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                      find_unused_parameters=True, broadcast_buffers=False)
    ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[args.gpu], 
                                                          find_unused_parameters=True, broadcast_buffers=False)
    init_ema_model(model, ema_model)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Enhanced depth and normal loss
    depth_normal_loss_fn = None
    if getattr(args, 'enable_depth_loss', False) or getattr(args, 'enable_normal_loss', False) or getattr(args, 'enable_normals_decoder_loss', False):
        depth_normal_loss_fn = DepthNormalConsistencyLoss(
            depth_model_name=getattr(args, 'depth_model_name', 'depth-anything/Depth-Anything-V2-Small-hf'),
            normals_model_name=getattr(args, 'normals_model_name', 'GonzaloMG/marigold-e2e-ft-normals'),
            device=args.gpu,
            enable_depth=getattr(args, 'enable_depth_loss', False),
            enable_normals=(
            getattr(args, 'enable_normal_loss', False) or 
            getattr(args, 'enable_normals_decoder_loss', False)
        )
        )

    return model, ema_model, optimizer, depth_normal_loss_fn


def main():
    torch.manual_seed(2)
    import os
    cudnn.deterministic = True
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('visualize', exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    
    # Set random seeds if specified
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    # Setup distributed training
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('Starting training...')
    if args.distributed:
        if args.local_rank != -1:  # torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # SLURM scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    args.gpu = args.gpu % torch.cuda.device_count()
    print('World size:', args.world_size)
    print('Rank:', args.rank)
    
    # Suppress printing if not master GPU
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args = copy.deepcopy(args)
    args.cos = True
    args.is_master = args.rank == 0

    # Handle resuming/finetuning
    is_finetuning = False
    checkpoint_path = None
    loaded_epoch = 0
    
    if args.resume or args.auto_resume or args.resume_from:
        experiment_path, checkpoint_path = find_experiment_folder(args)
        if experiment_path is None:
            if args.is_master:
                print("No checkpoint found for resuming, starting new experiment")
                experiment_path = create_experiment_folder(args)
        else:
            is_finetuning = True
            if args.is_master:
                print(f"Found checkpoint for finetuning: {checkpoint_path}")
    else:
        if args.is_master:
            experiment_path = create_experiment_folder(args)

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
        
        # Add finetuning indicator to run name
        if is_finetuning:
            args.wandb_run_name += "_finetuned"
        
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
            "enable_normal_loss": args.enable_normal_loss,
            "viz_log_freq": getattr(args, 'viz_log_freq', 50),
            "max_viz_images": getattr(args, 'max_viz_images', 4),
            "validation_datasets": getattr(args, 'validation_datasets', ['mit', 'rsr_256']),
            "training_dataset": args.dataset,
            "is_finetuning": is_finetuning,
        }

        if args.enable_depth_loss:
            config["depth_loss_weight"] = args.depth_loss_weight
            config["depth_model_name"] = args.depth_model_name
        
        if args.enable_normal_loss:
            config["normal_loss_weight"] = args.normal_loss_weight
            config["normals_model_name"] = args.normals_model_name

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            resume="allow" if checkpoint_path else None
        )

    # Initialize model and optimizer
    model, ema_model, optimizer, depth_normal_loss_fn = init_model(args)
    torch.cuda.set_device(args.gpu)
    
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if available (but reset counters)
    if checkpoint_path:
        loaded_epoch = load_checkpoint_improved(
            checkpoint_path, model, ema_model, optimizer, scaler, args
        )
        if args.is_master:
            print(f"Loaded checkpoint from epoch {loaded_epoch}, but starting fresh at epoch 0")
            if is_finetuning:
                print("Starting from epoch 0 with loaded weights (finetuning)")

    # Initialize datasets and loaders
    train_dataset, val_dataset = setup_datasets(args)
    print(f'Training images: {len(train_dataset)}')
    print(f'Validation images: {len(val_dataset)}')
    
    # Create data loaders
    train_loader, validation_loaders = setup_data_loaders(train_dataset, val_dataset, args)
    
    # Training loop - always start from 0
    global_step = 0
    for epoch in range(args.epochs):
        if args.distributed and hasattr(train_loader, 'sampler') and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        
        # Training
        global_step = train_epoch(train_loader, model, scaler, optimizer, ema_model, 
                                  epoch, args, global_step, depth_normal_loss_fn, is_finetuning)
        
        # Validation every "x" epochs
        if epoch % 1 == 0 or epoch == 0:
            print(f"Running validation at epoch {epoch}")
            validate_multi_datasets(validation_loaders, model, epoch, args, global_step, depth_normal_loss_fn=depth_normal_loss_fn)
        
        # Save checkpoint
        if args.is_master and (epoch % args.save_every == 0 or epoch == args.epochs - 1):
            save_checkpoint_improved(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'scaler': scaler.state_dict(),
                    'args': args,
                    'global_step': global_step,
                    'original_loaded_epoch': loaded_epoch,  # Keep track of original epoch
                },
                experiment_path, 
                epoch + 1, 
                is_best=False,
                keep_last=args.keep_last
            )

    # Finish wandb
    if args.is_master and not args.disable_wandb:
        wandb.finish()


def setup_datasets(args):
    """Setup training and validation datasets"""
    transform_train = [
        affine_crop_resize(size=(256, 256), scale=(0.2, 1.0)),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    ]

    transform_val = [
        None, 
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    ]

    # Dataset initialization
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
                                 is_validation=False)#, validation_scenes=["scene_01", "scene_03", "scene_05", "scene_06", "scene_07", "scene_08", "scene_10", "scene_21", "scene_22", "scene_23", "scene_24", "scene_26", "scene_27", "scene_29", "scene_30"])
        val_dataset = RSRDataset(root_dir=args.data_path, img_transform=transform_val, 
                                is_validation=True)#, validation_scenes=["scene_01", "scene_03", "scene_05", "scene_06", "scene_07", "scene_08", "scene_10", "scene_21", "scene_22", "scene_23", "scene_24", "scene_26", "scene_27", "scene_29", "scene_30"])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_dataset, val_dataset


def setup_data_loaders(train_dataset, val_dataset, args):
    """Setup data loaders for training and validation"""
    # Setup samplers
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Create train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=args.workers, 
        pin_memory=False, 
        sampler=train_sampler, 
        drop_last=True, 
        persistent_workers=True
    )
    
    # Create validation loaders
    validation_datasets = create_validation_datasets(args, 
                                                   [None, transforms.Compose([
                                                       transforms.Resize((256, 256)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                   ])])
    validation_loaders = create_validation_loaders(validation_datasets, args, val_sampler)
    
    return train_loader, validation_loaders


import time

def train_epoch(train_loader, model, scaler, optimizer, ema_model, epoch, args, global_step, depth_normal_loss_fn, is_finetuning):
    """Train for one epoch"""
    loss_metrics = ['loss', 'reconstruction', 'logdet', 'light_logdet', 'intrinsic_sim', 'depth_loss', 'normal_loss', 'normals_decoder_loss']
    meters = [AverageMeter(name, ':6.3f') for name in loss_metrics]
    progress = ProgressMeter(len(train_loader), meters, prefix=f"Epoch: [{epoch}]")

    model.train()
    
    # Training parameters
    P_mean, P_std = -1.2, 1.2
    sigma_data = 0.5
    
    # Loss functions
    logdet_loss = compute_logdet_loss()
    logdet_loss_ext = compute_logdet_loss()
    ssim_loss = compute_SSIM_loss()

    for i, (input_img, ref_img) in enumerate(train_loader):
        """if i>=2:
            break  # Limit to first 5 batches for debugging"""
        current_step = global_step + i
        
        # Start timing the step
        step_start_time = time.time()
        
        input_img = input_img.to(args.gpu)
        ref_img = ref_img.to(args.gpu)

        # Add noise
        rnd_normal = torch.randn([input_img.shape[0], 1, 1, 1], device=input_img.device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        if epoch >= 60:
            sigma = sigma * 0

        noisy_input_img = input_img + torch.randn_like(input_img) * sigma
        noisy_ref_img = ref_img + torch.randn_like(ref_img) * sigma

        # Forward pass timing
        forward_start_time = time.time()
        with torch.cuda.amp.autocast():
            intrinsic_input, extrinsic_input = model(noisy_input_img, run_encoder=True)
            intrinsic_ref, extrinsic_ref = model(noisy_ref_img, run_encoder=True)

            # Reconstruct images
            recon_img_rr, normals_ref = model([intrinsic_ref, extrinsic_ref], run_encoder=False, decode_mode='both')
            recon_img_ir, normals_input = model([intrinsic_input, extrinsic_ref], run_encoder=False, decode_mode='both')
            recon_img_ri, _ = model([intrinsic_ref, extrinsic_input], run_encoder=False, decode_mode='both')
            recon_img_ii, _ = model([intrinsic_input, extrinsic_input], run_encoder=False, decode_mode='both')
            
            recon_img_rr, recon_img_ir, recon_img_ri, recon_img_ii, normals_input, normals_ref= recon_img_ii.float(), recon_img_ir.float(), recon_img_ri.float(), recon_img_ii.float(), normals_input.float(), normals_ref.float()
            # Generate normals from the new decoder (using intrinsics only)
            #normals_input = model(intrinsic_input, run_encoder=False, decode_mode='normal').float()
            #normals_ref = model(intrinsic_ref, run_encoder=False, decode_mode='normal').float()
        forward_time = time.time() - forward_start_time
        if i < 5 and args.is_master and not args.disable_wandb:
            try:
                # Standard relighting visualization
                log_relighting_results(
                    input_img=input_img,
                    ref_img=ref_img,
                    relit_img=recon_img_ir,
                    gt_img=ref_img,
                    step=current_step,
                    )
            except:
                print("Warning: Failed to log relighting results, skipping visualization in training loop")

        # Loss computation timing
        loss_start_time = time.time()
        # Compute losses
        rec_loss_total = compute_reconstruction_loss(
            [(recon_img_rr, ref_img), (recon_img_ir, ref_img), 
             (recon_img_ri, input_img), (recon_img_ii, input_img)], 
            ssim_loss
        )
        
        logdet_pred, logdet_target = logdet_loss(intrinsic_input)
        logdet_pred_ext, logdet_target_ext = logdet_loss_ext([extrinsic_input])
        
        logdet_reg_loss = args.reg_weight * ((logdet_pred - logdet_target) ** 2).mean()
        logdet_ext_reg_loss = args.reg_weight * ((logdet_pred_ext - logdet_target_ext) ** 2).mean()
        
        sim_intrinsic = intrinsic_loss(intrinsic_input, intrinsic_ref)
        intrinsic_loss_component = args.intrinsics_loss_weight * sim_intrinsic
        
        # Enhanced depth and normal loss
        depth_loss_value = torch.tensor(0.0, device=input_img.device)
        normal_loss_value = torch.tensor(0.0, device=input_img.device)
        
        if depth_normal_loss_fn is not None:
            try:
                #total_geom_loss, depth_loss_component, normal_loss_component = depth_normal_loss_fn(input_img, recon_img_ri)
                
                # You can separate depth and normal losses if needed by calling them individually
                if getattr(args, 'enable_depth_loss', False) and getattr(args, 'enable_normal_loss', False):
                    # Both enabled
                    total_geom_loss, depth_loss_value, normal_loss_value, _, _, _, _ = depth_normal_loss_fn(input_img, recon_img_ri)
                elif getattr(args, 'enable_depth_loss', False):
                    total_geom_loss, depth_loss_value, _, _, _, _, _ = depth_normal_loss_fn(input_img, recon_img_ri)
                elif getattr(args, 'enable_normal_loss', False):
                    total_geom_loss, _, normal_loss_value, _, _, _, _ = depth_normal_loss_fn(input_img, recon_img_ri)
                    
            except Exception as e:
                print(f"Warning: Depth/Normal loss computation failed: {e}")
                total_geom_loss = torch.tensor(0.0, device=input_img.device)

        # Combine geometric losses
        depth_loss_component = getattr(args, 'depth_loss_weight', 0.0) * depth_loss_value
        normal_loss_component = getattr(args, 'normal_loss_weight', 0.0) * normal_loss_value
        
        # normals decoder loss
        normals_decoder_loss_value = torch.tensor(0.0, device=input_img.device)
        if getattr(args, 'enable_normals_decoder_loss', False) and depth_normal_loss_fn is not None:
            try:
                # Extract ground truth normals from depth_normal_loss_fn
                # We'll use the input image to get ground truth normals
                
                gt_normals_input = depth_normal_loss_fn.get_normal_map(input_img)
                gt_normals_ref = depth_normal_loss_fn.get_normal_map(ref_img)
                
                if gt_normals_input is not None:
                    _, _, normals_input_loss_value, _, _, _, _ = depth_normal_loss_fn(gt_normals_input, normals_input)
                    normals_decoder_loss_value += normals_input_loss_value
                    
                # Also add consistency loss between input and reference normals when they should be similar
                if gt_normals_ref is not None:
                    _, _, normals_ref_loss_value , _, _, _, _ = depth_normal_loss_fn(gt_normals_ref, normals_ref)
                    normals_decoder_loss_value += normals_ref_loss_value
                    
                        
            except Exception as e:
                print(f"Warning: Normals decoder loss computation failed: {e}")
                normals_decoder_loss_value = torch.tensor(0.0, device=input_img.device)
        
        normals_decoder_loss_component = getattr(args, 'normals_decoder_loss_weight', 0.0) * normals_decoder_loss_value
       
        # Total loss
        total_loss = (rec_loss_total + logdet_reg_loss + intrinsic_loss_component + 
                     depth_loss_component + normal_loss_component + logdet_ext_reg_loss +
                     normals_decoder_loss_component )
        print(f"Total loss: {total_loss.item()} | Rec: {rec_loss_total.item()} | Logdet: {logdet_reg_loss.item()} | Intrinsic: {intrinsic_loss_component.item()} | Depth: {depth_loss_component.item()} | Normal: {normal_loss_component.item()} | Logdet Ext: {logdet_ext_reg_loss.item()} | Normals Decoder: {normals_decoder_loss_component.item()}")
        loss_time = time.time() - loss_start_time
        # Backward pass timing
        backward_start_time = time.time()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        update_ema_model(model, ema_model, 0.999)
        backward_time = time.time() - backward_start_time
        
        # Calculate total step time
        step_time = time.time() - step_start_time

        # Log metrics
        if args.is_master and not args.disable_wandb and i % args.log_freq == 0:
            metrics = {
                'total_loss': total_loss.item(),
                'reconstruction_loss': rec_loss_total.item(),
                'logdet_loss': logdet_reg_loss.item(),
                'intrinsic_loss': intrinsic_loss_component.item(),
                'logdet_ext_loss': logdet_ext_reg_loss.item(),
                'depth_loss': depth_loss_component.item(),
                'normal_loss': normal_loss_component.item(),
                'normals_decoder_loss': normals_decoder_loss_component.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'is_finetuning': is_finetuning,
                # Timing metrics
                'step_time': step_time,
                'forward_time': forward_time,
                'loss_time': loss_time,
                'backward_time': backward_time,
                'steps_per_second': 1.0 / step_time if step_time > 0 else 0.0,
            }
            wandb.log(metrics, step=current_step)

        # Update progress meters
        meters[0].update(total_loss.item())
        meters[1].update(rec_loss_total.item())
        meters[2].update(logdet_reg_loss.item())
        meters[3].update(logdet_ext_reg_loss.item())
        meters[4].update(intrinsic_loss_component.item())
        meters[5].update(depth_loss_component.item())
        meters[6].update(normal_loss_component.item())
        meters[7].update(normals_decoder_loss_component.item())
        
        if i % 50 == 0:
            progress.display(i)
        

    torch.distributed.barrier()
    return global_step + len(train_loader)



def compute_reconstruction_loss(recon_pairs, ssim_loss):
    """Compute reconstruction loss for multiple image pairs"""
    total_loss = 0
    for recon_img, target_img in recon_pairs:
        mse_loss = nn.MSELoss()(recon_img, target_img)
        ssim_component = 0.1 * (1 - ssim_loss(recon_img, target_img))
        grad_component = gradient_loss(recon_img, target_img)
        total_loss += 10 * mse_loss + ssim_component + grad_component
    return total_loss / len(recon_pairs)


def validate_multi_datasets(validation_loaders, model, epoch, args, current_step, depth_normal_loss_fn=None):
    """Validation using the same losses as training"""
    model.eval()
    
    # Loss functions
    logdet_loss = compute_logdet_loss()
    logdet_loss_ext = compute_logdet_loss()
    ssim_loss = compute_SSIM_loss()
    
    with torch.no_grad():
        for dataset_name, val_loader in validation_loaders.items():
            print(f"Validating on {dataset_name}...")
            
            val_metrics = {}
            total_loss = 0
            total_rec_loss = 0
            total_logdet_int_loss = 0
            total_logdet_ext_loss = 0
            total_intrinsic_loss = 0
            total_depth_loss = 0
            total_normal_loss = 0
            total_normals_decoder_loss = 0
            num_batches = 0

            total_val_batches = len(val_loader)
            last_batch_idx = total_val_batches - 1

            for i, batch in enumerate(val_loader):
                if i != 0 and i != last_batch_idx:
                    continue
                    
                if len(batch) == 3:
                    input_img, ref_img, gt_img = batch
                else:
                    input_img, ref_img = batch
                    gt_img = ref_img
                
                input_img = input_img.to(args.gpu)
                ref_img = ref_img.to(args.gpu)
                gt_img = gt_img.to(args.gpu)
                
                # Forward pass (no noise for validation)
                intrinsic_input, extrinsic_input = model(input_img, run_encoder=True)
                intrinsic_ref, extrinsic_ref = model(ref_img, run_encoder=True)
                
                # Reconstruct images
                recon_img_rr, normals_ref = model([intrinsic_ref, extrinsic_ref], run_encoder=False, decode_mode='both')
                recon_img_ir, normals_input = model([intrinsic_input, extrinsic_ref], run_encoder=False, decode_mode='both')
                recon_img_ii, _ = model([intrinsic_input, extrinsic_input], run_encoder=False, decode_mode='both')
            
                recon_img_rr, recon_img_ir, recon_img_ii, normals_input, normals_ref= recon_img_ii.float(), recon_img_ir.float(), recon_img_ii.float(), normals_input.float(), normals_ref.float()
           
                
                # Compute same losses as training
                rec_loss_total = compute_reconstruction_loss(
                    [(recon_img_rr, ref_img), (recon_img_ir, gt_img), (recon_img_ii, input_img)], 
                    ssim_loss
                )
                
                logdet_pred, logdet_target = logdet_loss(intrinsic_input)
                logdet_pred_ext, logdet_target_ext = logdet_loss_ext([extrinsic_input])
                
                logdet_reg_loss = args.reg_weight * ((logdet_pred - logdet_target) ** 2).mean()
                logdet_ext_reg_loss = args.reg_weight * ((logdet_pred_ext - logdet_target_ext) ** 2).mean()
                
                sim_intrinsic = intrinsic_loss(intrinsic_input, intrinsic_ref)
                intrinsic_loss_component = args.intrinsics_loss_weight * sim_intrinsic

                # Depth and normal loss for validation
                depth_loss_value = torch.tensor(0.0, device=input_img.device)
                normal_loss_value = torch.tensor(0.0, device=input_img.device)
                input_depth, relit_depth = None, None
                input_normals, relit_normals = None, None
                
                if depth_normal_loss_fn is not None:
                    try:
                        #total_geom_loss, depth_loss_component, normal_loss_component = depth_normal_loss_fn(input_img, recon_img_ri)
                        
                        # You can separate depth and normal losses if needed by calling them individually
                        if getattr(args, 'enable_depth_loss', False) and getattr(args, 'enable_normal_loss', False):
                            # Both enabled
                            total_geom_loss, depth_loss_value, normal_loss_value, input_depth, relit_depth, input_normals, relit_normals = depth_normal_loss_fn(input_img, recon_img_ir)
                        elif getattr(args, 'enable_depth_loss', False):
                            total_geom_loss, depth_loss_value, _, input_depth, relit_depth, _, _= depth_normal_loss_fn(input_img, recon_img_ir)
                        elif getattr(args, 'enable_normal_loss', False):
                            total_geom_loss, _, normal_loss_value, _, _, input_normals, relit_normals = depth_normal_loss_fn(input_img, recon_img_ir)
                            
                    except Exception as e:
                        print(f"Warning: Depth/Normal loss computation failed: {e}")
                        total_geom_loss = torch.tensor(0.0, device=input_img.device)
                
                depth_loss_component = getattr(args, 'depth_loss_weight', 0.0) * depth_loss_value
                normal_loss_component = getattr(args, 'normal_loss_weight', 0.0) * normal_loss_value

                normals_decoder_loss_value = torch.tensor(0.0, device=input_img.device)
                if getattr(args, 'enable_normals_decoder_loss', False) and depth_normal_loss_fn is not None:
                    try:
                        # Extract ground truth normals from depth_normal_loss_fn
                        # We'll use the input image to get ground truth normals
                        
                        gt_normals_input = depth_normal_loss_fn.get_normal_map(input_img)
                        gt_normals_ref = depth_normal_loss_fn.get_normal_map(ref_img)
                        
                        if gt_normals_input is not None:
                            _, _, normals_input_loss_value, _, _, _, _ = depth_normal_loss_fn(gt_normals_input, normals_input)
                            normals_decoder_loss_value += normals_input_loss_value
                            
                        # Also add consistency loss between input and reference normals when they should be similar
                        if gt_normals_ref is not None:
                            _, _, normals_ref_loss_value , _, _, _, _ = depth_normal_loss_fn(gt_normals_ref, normals_ref)
                            normals_decoder_loss_value += normals_ref_loss_value
                            
                                
                    except Exception as e:
                        print(f"Warning: Normals decoder loss computation failed: {e}")
                        normals_decoder_loss_value = torch.tensor(0.0, device=input_img.device)
                
                normals_decoder_loss_component = getattr(args, 'normals_decoder_loss_weight', 0.0) * normals_decoder_loss_value
                
                batch_total_loss = (rec_loss_total + logdet_reg_loss + intrinsic_loss_component + 
                                  logdet_ext_reg_loss + depth_loss_component + normal_loss_component + normals_decoder_loss_component)
                
                total_loss += batch_total_loss.item()
                total_rec_loss += rec_loss_total.item()
                total_logdet_int_loss += logdet_reg_loss.item()
                total_logdet_ext_loss += logdet_ext_reg_loss.item()
                total_intrinsic_loss += intrinsic_loss_component.item()
                total_depth_loss += depth_loss_component.item()
                total_normal_loss += normal_loss_component.item()
                total_normals_decoder_loss = normals_decoder_loss_component.item()
                num_batches += 1
                
                # Log visualizations for first batch only
                if i == 0 and args.is_master and not args.disable_wandb:
                    try:
                        # Standard relighting visualization
                        log_relighting_results(
                            input_img=input_img,
                            ref_img=ref_img,
                            relit_img=recon_img_ir,
                            gt_img=gt_img,
                            step=current_step,
                            mode=f'{dataset_name}_validation'
                        )
                        
                        # Depth and normal visualization
                        if depth_normal_loss_fn is not None and (input_depth is not None or input_normals is not None):
                            vis_image = depth_normal_loss_fn.create_comprehensive_visualization(
                                input_img=input_img,
                                relit_img=recon_img_ir,
                                gt_img=gt_img,
                                input_depth=input_depth,
                                relit_depth=relit_depth,
                                gt_depth=depth_normal_loss_fn.get_depth_map(gt_img),
                                input_normals=input_normals,
                                relit_normals=relit_normals,
                                gt_normals=depth_normal_loss_fn.get_normal_map(gt_img),
                                epoch=epoch,
                                dataset_name=f'{dataset_name}_validation'
                            )
                            
                            if vis_image is not None:
                                wandb.log({
                                    f'{dataset_name}_depth_normal_comparison': wandb.Image(
                                        vis_image, 
                                        caption=f"{dataset_name} Validation Depth/Normal Comparison - Epoch {epoch}"
                                    )
                                }, step=current_step)
                        
                        print(f"Successfully logged images for {dataset_name}")
                    except Exception as e:
                        print(f"Failed to log images for {dataset_name}: {e}")
            
            if num_batches > 0:
                val_metrics = {
                    f'val_{dataset_name}_total_loss': total_loss / num_batches,
                    f'val_{dataset_name}_reconstruction_loss': total_rec_loss / num_batches,
                    f'val_{dataset_name}_logdet_loss': total_logdet_int_loss / num_batches,
                    f'val_{dataset_name}_intrinsic_loss': total_intrinsic_loss / num_batches,
                    f'val_{dataset_name}_logdet_ext_loss': total_logdet_ext_loss / num_batches,
                    f'val_{dataset_name}_depth_loss': total_depth_loss / num_batches,
                    f'val_{dataset_name}_normal_loss': total_normal_loss / num_batches,
                    f'val_{dataset_name}_normals_decoder_loss': total_normals_decoder_loss / num_batches,
                    f'val_epoch': epoch,
                }
                
                if args.is_master and not args.disable_wandb:
                    wandb.log(val_metrics, step=current_step)
                    print(f"Validation metrics for {dataset_name}: {val_metrics}")
    
    model.train()


if __name__ == '__main__':
    main()