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
parser.add_argument('--dataset', default='mit', choices=['mit', 'rsr_256', 'iiw', 'rlsid'], 
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
        if epoch % 3 == 0 or epoch == 0:
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
    elif args.dataset == 'rlsid':
        train_dataset = RLSIDDataset(
            root_dir=args.data_path,
            validation_scenes=['00012', '00014', '00020', '00022', '00029', '00030', '00036', '00042', '00043', '00049', '00054', '00058', '00060', '00070', '00076', '00081', '00087', '00098', '00103', '00112', '00113', '00115', '00121', '00124', '00131', '00137', '00138', '00145', '00156', '00157', '00164', '00175', '00176', '00177', '00178', '00179', '00181', '00190', '00198', '00200', '00205', '00206', '00209', '00217', '00219', '00230', '00240', '00244', '00246', '00247', '00255', '00260', '00262', '00276', '00281', '00282', '00283', '00286', '00291', '00306', '00312', '00317', '00320', '00321', '00325', '00332', '00334', '00341', '00343', '00344', '00346', '00349', '00352', '00356', '00361', '00364', '00369', '00371', '00372', '00373', '00380', '00384', '00385', '00392', '00400', '00402', '00411', '00413', '00414', '00419', '00422', '00424', '00428', '00431', '00435', '00437', '00442', '00443', '00448', '00457', '00459', '00461', '00466', '00476', '00477', '00491', '00494', '00495', '00498', '00505', '00506', '00515', '00527', '00529', '00533', '00542', '00552', '00556', '00558', '00566', '00569', '00573', '00576', '00577', '00581', '00586', '00587', '00594', '00600', '00603', '00604', '00605', '00608', '00619', '00636', '00648', '00663', '00669', '00673', '00678', '00687', '00697', '00702', '00704', '00707', '00713', '00726', '00727', '00731', '00736', '00743', '00744', '00749', '00760', '00768', '00772', '00776', '00777', '00785', '00788', '00790', '00792', '00794', '00796', '00802', '00806', '00820', '00821', '00824', '00828', '00832', '00847', '00851', '00852', '00855', '00857', '00867', '00871', '00879', '00880', '00883', '00900', '00902', '00904', '00905', '00909', '00921', '00923', '00931', '00935', '00942', '00946', '00961', '00971', '00981', '00986', '00988', '01000', '01006', '01009', '01014', '01018', '01020', '01024', '01031', '01035', '01036', '01038', '01040', '01041', '01044', '01059', '01060', '01064', '01065', '01066', '01067', '01069', '01076', '01085', '01086', '01088', '01090', '01095', '01097', '01113', '01115', '01124', '01139', '01148', '01150', '01152', '01154', '01155', '01158', '01163', '01166', '01168', '01178', '01182', '01191', '01197', '01200', '01201', '01207', '01210', '01211', '01213', '01214', '01218', '01221', '01223', '01229', '01233', '01236', '01244', '01250', '01253', '01258', '01263', '01270', '01278', '01293', '01295', '01301', '01311', '01314', '01323', '01329', '01338', '01339', '01342', '01343', '01348', '01356', '01363', '01367', '01370', '01384', '01393', '01398', '01404', '01405', '01410', '01416', '01419', '01423', '01424', '01426', '01431', '01433', '01438', '01444', '01451', '01460', '01463', '01466', '01468', '01469', '01479', '01481', '01485', '01487', '01492', '01498', '01505', '01519', '01520', '01525', '01526', '01537', '01547', '01561', '01566', '01570', '01577', '01579', '01582', '01583', '01585', '01588', '01589', '01593', '01594', '01599', '01603', '01604', '01605', '01606', '01610', '01611', '01620', '01622', '01627', '01637', '01646', '01649', '01650', '01652', '01657', '01658', '01668', '01670', '01672', '01678', '01680', '01681', '01686', '01694', '01710', '01711', '01716', '01721', '01726', '01732', '01733', '01735', '01737', '01740', '01744', '01748', '01754', '01766', '01767', '01773', '01779', '01781', '01786', '01790', '01793', '01796', '01804', '01806', '01807', '01809', '01811', '01821', '01831', '01836', '01842', '01845', '01851', '01852', '01858', '01865', '01877', '01887', '01895', '01896', '01901', '01903', '01906', '01912', '01922', '01924', '01929', '01931', '01932', '01935', '01940', '01941', '01946', '01964', '01966', '01969', '01976', '01981', '01985', '01987', '01990', '01993', '01995', '02007', '02047', '02051', '02052', '02057', '02062', '02069', '02073', '02076', '02081', '02083', '02100', '02101', '02102', '02110', '02119', '02125', '02129', '02130', '02131', '02135', '02136', '02145', '02150', '02163', '02167', '02189', '02202', '02216', '02219', '02229', '02230', '02233', '02234', '02235', '02236', '02247', '02256', '02258', '02261', '02263', '02265', '02266', '02267', '02273', '02278', '02287', '02288', '02291', '02307', '02311', '02322', '02335', '02343', '02352', '02355', '02356', '02359', '02373', '02374', '02380', '02385', '02387', '02392', '02396', '02400', '02407', '02416', '02417', '02418', '02424', '02425', '02429', '02441', '02452', '02456', '02491', '02501', '02503', '02513', '02515', '02517', '02518', '02520', '02526', '02530', '02537', '02539', '02542', '02543', '02548', '02550', '02561', '02562', '02569', '02580', '02581', '02584', '02591', '02594', '02604', '02605', '02607', '02624', '02631', '02632', '02634', '02642', '02645', '02653', '02660', '02661', '02669', '02670', '02672', '02676', '02678', '02684', '02694', '02696', '02697', '02701', '02709', '02711', '02713', '02734', '02736', '02737', '02743', '02752', '02767', '02770', '02776', '02779', '02783', '02784', '02790', '02791', '02793', '02797', '02808', '02810', '02814', '02820', '02826', '02828', '02829', '02833', '02834', '02839', '02852', '02854', '02856', '02858', '02859', '02860', '02863', '02865', '02870', '02874', '02886', '02888', '02894', '02900', '02905', '02906', '02911', '02913', '02917', '02922', '02928', '02942', '02945', '02946', '02952', '02958', '02973', '02975', '02976', '02977', '02982', '02995', '02996', '02997', '03001', '03006', '03013', '03015', '03018', '03024', '03038', '03048', '03056', '03057', '03061', '03071', '03072', '03078', '03084', '03087', '03099', '03101', '03116', '03118', '03120', '03128', '03133', '03143', '03145', '03149', '03150', '03152', '03154', '03157', '03158', '03161', '03162', '03164', '03169', '03170', '03172', '03184', '03191', '03192', '03207', '03208', '03213', '03219', '03220', '03237', '03241', '03247', '03251', '03256', '03258', '03271', '03273', '03283', '03288', '03290', '03297', '03302', '03307', '03308', '03333', '03340', '03343', '03345', '03352', '03362', '03366', '03368', '03379', '03391', '03395', '03397', '03398', '03416', '03419', '03430', '03431', '03440', '03447', '03450', '03459', '03467', '03469', '03489', '03491', '03492', '03501', '03503', '03507', '03511', '03518', '03519', '03520', '03539', '03547', '03556', '03569', '03571', '03574', '03595', '03604', '03617', '03621', '03631', '03638', '03640', '03647', '03648', '03656', '03662', '03668', '03670', '03672', '03675', '03680', '03682', '03690', '03691', '03693', '03694', '03706', '03713', '03716', '03717', '03724', '03731', '03733', '03734', '03747', '03758', '03781', '03782', '03791', '03794', '03795', '03801', '03802', '03813', '03815', '03817', '03821', '03826', '03830', '03832', '03856', '03861', '03864', '03867', '03873', '03877', '03883', '03889', '03891', '03892', '03895', '03896', '03900', '03901', '03904', '03909', '03912', '03915', '03920', '03925', '03944', '03948', '03951', '03956', '03960', '03967', '03968', '03970', '03981', '03986', '03997', '03998', '04002', '04004', '04016', '04019', '04022', '04023', '04027', '04032', '04041', '04046', '04049', '04050', '04051', '04054', '04055', '04059', '04060', '04064', '04072', '04074', '04075', '04077', '04082', '04087', '04096', '04097', '04104', '04105', '04115', '04116', '04117', '04120', '04122', '04128', '04130', '04140', '04142', '04146', '04149', '04152', '04156', '04165', '04170', '04175', '04180', '04191', '04192', '04202', '04212', '04213', '04222', '04226', '04232', '04235', '04237', '04254', '04258', '04259', '04267', '04279', '04283', '04287', '04289', '04290', '04291', '04295', '04298', '04313', '04317', '04319', '04326', '04328', '04329', '04363', '04364', '04365', '04368', '04372', '04373', '04376', '04386', '04387', '04389', '04390', '04396', '04398', '04402', '04405', '04408', '04417', '04438', '04439', '04446', '04456', '04460', '04464', '04466', '04467', '04474', '04476', '04489', '04490', '04494', '04498', '04507', '04518', '04519', '04520', '04522', '04536', '04546', '04550', '04560', '04581', '04586', '04589', '04597', '04598', '04601', '04604', '04607', '04608', '04616', '04622', '04624', '04626', '04637', '04641', '04644', '04645', '04650', '04656', '04660', '04669', '04672', '04676', '04678', '04681', '04685', '04689', '04690', '04693', '04694', '04698', '04706', '04707', '04715', '04728', '04729', '04742', '04744', '04751', '04752', '04753', '04756', '04758', '04760', '04767', '04794', '04798', '04800', '04801', '04803', '04813', '04815', '04818', '04825', '04827', '04832', '04842', '04861', '04862', '04875', '04881', '04889', '04890', '04895', '04902', '04923', '04934', '04943', '04950', '04955', '04961', '04972', '04977', '04978', '04980', '04987', '04995', '04998', '05005', '05006', '05013', '05016', '05019', '05025', '05027', '05029', '05036', '05037', '05042', '05046', '05056', '05062', '05063', '05066', '05068', '05081', '05089', '05091', '05092', '05093', '05095', '05097', '05099', '05103', '05108', '05111', '05112', '05116', '05117', '05124', '05125', '05127', '05128', '05135', '05136', '05138', '05139', '05144', '05147', '05148', '17219', '17235', '17239', '17245', '17250', '17258', '17263', '17275', '17276', '17278', '17286', '17287', '17291', '17293', '17294', '17297', '17307', '17309', '17313', '17318', '17324', '17329', '17333', '17334', '17341', '17347', '17359', '17362', '17364', '17367', '17371', '17378', '17385', '17386', '17389', '17402', '17417', '17418', '17421', '17425', '17439', '17441', '17446', '17447', '17455', '17457', '17462', '17466', '17475', '17477', '17483', '17485', '17497', '17506', '17514', '17518', '17521', '17527', '17528', '17538', '17539', '17541', '17546', '17547', '17550', '17554', '17562', '17564', '17565', '17566', '17580', '17583', '17597', '17599', '17601', '17602', '17603', '17606', '17611', '17612', '17616', '17621', '17623', '17635', '17637', '17656', '17662', '17665', '17666', '17674', '17681', '17682', '17683', '17690', '17693', '17696', '17704', '17705', '17719', '17722', '17727', '17729', '17730', '17741', '17745', '17746', '17747', '17751', '17753', '17756', '17757', '17761', '17763', '17765', '17768', '17770', '17771', '17784', '17786', '17788', '17800', '17801', '17805', '17806', '17807', '17809', '17815', '17821', '17828', '17829', '17831', '17836', '17837', '17851', '17855', '17863', '17865', '17873', '17876', '17877', '17883', '17884', '17893', '17894', '17897', '17906', '17908', '17911', '17919', '17932', '17933', '17937', '17946', '17947', '17949', '17953', '17957', '17962', '17968', '17974', '17987', '17990', '17994', '17995', '18002', '18007', '18008', '18011', '18018', '18022', '18026', '18028', '18033', '18037', '18041', '18043', '18045', '18048', '18055', '18063', '18075', '18078', '18080', '18082', '18092', '18094', '18097', '18104', '18106', '18107', '18110', '18111', '18112', '18126', '18128', '18152', '18159', '18162', '18171', '18172', '18178', '18179', '18180', '18181', '18195', '18197', '18206', '18208', '18209', '18211', '18221', '18228', '18232', '18237', '18238', '18239', '18241', '18253', '18270', '18280', '18288', '18292', '18294', '18299', '18301', '18302', '18304', '18312', '18319', '18320', '18323', '18327', '18335', '18337', '18342', '18353', '18354', '18357', '18358', '18360', '18364', '18365', '18375', '18377', '18381', '18382', '18383', '18393', '18394', '18401', '18405', '18414', '18417', '18419', '18431', '18434', '18436', '18441', '18446', '18447', '18451', '18453', '18459', '18460', '18462', '18466', '18468', '18477', '18482', '18490', '18492', '18493', '18496', '18499', '18501', '18511', '18518', '18532', '18542', '18543', '18546', '18551', '18556', '18561', '18566', '18570', '18573', '18582', '18587', '18596', '18601', '18602', '18616', '18617', '18620', '18632', '18648', '18653', '18655', '18660', '18661', '18665', '18676', '18680', '18681', '18683', '18684', '18692', '18694', '18698', '18702', '18710', '18712', '18716', '18718', '18725', '18734', '18739', '18741', '18743', '18754', '18756', '18759', '18761', '18771', '18777', '18778', '18780', '18783', '18786', '18796', '18797', '18802', '18813', '18814', '18815', '18831', '18835', '18836', '18841', '18849', '18850', '18853', '18857', '18859', '18865', '18872', '18878', '18881', '18887', '18892', '18900', '18911', '18912', '18916', '18922', '18928', '18940', '18949', '18959', '18960', '18962', '18981', '18986', '18988', '18989', '18994', '19001', '19008', '19017', '19026', '19034', '19038', '19040', '19045', '19046', '19047', '19050', '19052', '19054', '19058', '19059', '19060', '19066', '19071', '19074', '19077', '19079', '19088', '19096', '19098', '19101', '19102', '19108', '19111', '19114', '19120', '19121', '19125', '19128', '19130', '19136', '19147', '19151', '19154', '19155', '19165', '19168', '19170', '19181', '19189', '19194', '19196', '19202', '19204', '19207', '19211', '19214', '19220', '19225', '19229', '19235', '19237', '19240', '19251', '19257', '19259', '19266', '19267', '19268', '19274', '19277', '19285', '19297', '19298', '19304', '19309', '19313', '19320', '19324', '19326', '19327', '19331', '19335', '19345', '19347', '19348', '19357', '19359', '19364', '19370', '19373', '19374', '19375', '19377', '19383', '19384', '19386', '19388', '19398', '19400', '19405', '19408', '19409', '19419', '19421', '19423', '19426', '19428', '19431', '19432', '19437', '19440', '19441', '19443', '19448', '19451', '19452', '19455', '19459', '19460', '19464', '19467', '19475', '19480', '19483', '19487', '19496', '19498', '19512', '19520', '19528', '19532', '19534', '19535', '19542', '19550', '19555', '19561', '19572', '19573', '19579', '19581', '19586', '19587', '19591', '19604', '19606', '19636', '19650', '19652', '19654', '19655', '19658', '19660', '19665', '19673', '19675', '19690', '19708', '19711', '19718', '19729', '19733', '19745', '19749', '19755', '19756', '19757', '19764', '19772', '19776', '19778', '19786', '19787', '19789', '19790', '19792', '19793', '19796', '19800', '19801', '19805', '19820', '19823', '19829', '19840', '19845', '19857', '19866', '19871', '19879', '19881', '19890', '19904', '19913', '19918', '19923', '19931', '19932', '19936', '19937', '19941', '19945', '19956', '19962', '19965', '19966', '19976', '19978', '19979', '19984', '19989', '19993', '19994', '19998', '20002', '20003', '20007', '20011', '20012', '20014', '20018', '20019', '20020', '20032', '20034', '20037', '20039', '20040', '20041', '20052', '20065', '20074', '20084', '20085', '20086', '20087', '20088', '20098', '20100', '20117', '20118', '20129', '20130', '20144', '20161', '20166', '20170', '20179', '20183', '20186', '20188', '20189', '20215', '20237', '20251', '20253', '20255', '20257', '20265', '20266', '20269', '20274', '20275', '20276', '20280', '20283', '20284', '20285', '20286', '20289', '20290', '20303', '20307', '20313', '20329', '20339', '20342', '20348', '20352', '20366', '20367', '20374', '20375', '20376', '20382', '20389', '20391', '20393', '20396', '20399', '20402', '20403', '20405', '20414', '20416', '20424', '20435', '20438', '20441', '20449', '20452', '20461', '20463', '20465', '20468', '20472', '20477', '20483', '20490', '20494', '20495', '20504', '20511', '20515', '20516', '20518', '20520', '20523', '20526', '20529', '20530', '20535', '20541', '20561', '20566', '20569', '20573', '20579', '20595', '20599', '20605', '20606', '20608', '20614', '20618', '20621', '20629', '20631', '20643', '20644', '20662', '20669', '20677', '20679', '20682', '20688', '20691', '20692', '20701', '20709', '20711', '20713', '20721', '20724', '20729', '20737', '20739', '20740', '20741', '20743', '20755', '20768', '20776', '20782', '20787', '20793', '20800', '20803', '20804', '20813', '20816', '20817', '20820', '20822', '20829', '20835', '20837', '20838', '20846', '20849', '20858', '20866', '20874', '20888', '20889', '20892', '20898', '20919', '20921', '20923', '20929', '20933', '20936', '20943', '20944', '20946', '20951', '20958', '20962', '20964', '20966', '20976', '20977', '20988', '20996', '21010', '21011', '21012', '21026', '21036', '21042', '21051', '21053', '21054', '21060', '21065', '21068', '21078', '21080', '21082', '21083', '21096', '21102', '21106', '21112', '21115', '21116', '21131', '21133', '21138', '21149', '21155', '21158', '21162', '21164', '21172', '21175', '21193', '21202', '21203', '21213', '21214', '21218', '21219', '21223', '21224', '21233', '21235', '21239', '21244', '21247', '21251', '21253', '21256', '21260', '21265', '21270', '21273', '21276', '21277', '21284', '21286', '21289', '21293', '21294', '21304', '21310', '21313', '21314', '21318', '21333', '21340', '21348', '21353', '21354', '21363', '21369', '21373', '21380', '21381', '21383', '21387', '21392', '21394', '21403', '21406', '21407', '21411', '21412', '21425', '21426', '21428', '21435', '21439', '21442', '21444', '21447', '21449', '21455', '21457', '21468', '21471', '21483', '21485', '21488', '21489', '21495', '21496', '21497', '21504', '21505', '21506', '21507', '21510', '21512', '21516', '21518', '21519', '21521', '21530', '21537', '21540', '21542', '21544', '21550', '21561', '21565', '21570', '21572', '21577', '21579', '21590', '21594', '21595', '21597', '21603', '21605', '21607', '21613', '21614', '21623', '21626', '21632', '21639', '21640', '21644', '21645', '21648', '21649', '21652', '21655', '21659', '21663', '21664', '21666', '21673', '21674', '21683', '21686', '21690', '21695', '21702', '21703', '21705', '21709', '21713', '21717', '21724', '21728', '21742', '21743', '21752', '21758', '21770', '21772', '21783', '21784', '21793', '21805', '21809', '21821', '21827', '21838', '21843', '21844', '21846', '21851', '21872', '21875', '21877', '21878', '21887', '21891', '21896', '21903', '21907', '21915', '21916', '21919', '21920', '21922', '21929', '21936', '21937', '21946', '21947', '21949', '21951', '21954', '21955', '21957', '21958', '21960', '21964', '21968', '21971', '21973', '21979', '21981', '21986', '21988', '21989', '21993', '21998', '22004', '22005', '22006', '22010', '22045', '22048', '22056', '22062', '22063', '22069', '22074', '22079', '22095', '22097', '22100', '22105', '22106', '22111', '22122', '22123', '22124', '22127', '22128', '22129'],
            img_transform=transform_train,
            is_validation=False,
            max_training_images=30000,
        )
        val_dataset = RLSIDDataset(
            root_dir='/data/storage/datasets/RLSID',
            validation_scenes=['00012', '00014', '00020', '00022', '00029', '00030', '00036', '00042', '00043', '00049', '00054', '00058', '00060', '00070', '00076', '00081', '00087', '00098', '00103', '00112', '00113', '00115', '00121', '00124', '00131', '00137', '00138', '00145', '00156', '00157', '00164', '00175', '00176', '00177', '00178', '00179', '00181', '00190', '00198', '00200', '00205', '00206', '00209', '00217', '00219', '00230', '00240', '00244', '00246', '00247', '00255', '00260', '00262', '00276', '00281', '00282', '00283', '00286', '00291', '00306', '00312', '00317', '00320', '00321', '00325', '00332', '00334', '00341', '00343', '00344', '00346', '00349', '00352', '00356', '00361', '00364', '00369', '00371', '00372', '00373', '00380', '00384', '00385', '00392', '00400', '00402', '00411', '00413', '00414', '00419', '00422', '00424', '00428', '00431', '00435', '00437', '00442', '00443', '00448', '00457', '00459', '00461', '00466', '00476', '00477', '00491', '00494', '00495', '00498', '00505', '00506', '00515', '00527', '00529', '00533', '00542', '00552', '00556', '00558', '00566', '00569', '00573', '00576', '00577', '00581', '00586', '00587', '00594', '00600', '00603', '00604', '00605', '00608', '00619', '00636', '00648', '00663', '00669', '00673', '00678', '00687', '00697', '00702', '00704', '00707', '00713', '00726', '00727', '00731', '00736', '00743', '00744', '00749', '00760', '00768', '00772', '00776', '00777', '00785', '00788', '00790', '00792', '00794', '00796', '00802', '00806', '00820', '00821', '00824', '00828', '00832', '00847', '00851', '00852', '00855', '00857', '00867', '00871', '00879', '00880', '00883', '00900', '00902', '00904', '00905', '00909', '00921', '00923', '00931', '00935', '00942', '00946', '00961', '00971', '00981', '00986', '00988', '01000', '01006', '01009', '01014', '01018', '01020', '01024', '01031', '01035', '01036', '01038', '01040', '01041', '01044', '01059', '01060', '01064', '01065', '01066', '01067', '01069', '01076', '01085', '01086', '01088', '01090', '01095', '01097', '01113', '01115', '01124', '01139', '01148', '01150', '01152', '01154', '01155', '01158', '01163', '01166', '01168', '01178', '01182', '01191', '01197', '01200', '01201', '01207', '01210', '01211', '01213', '01214', '01218', '01221', '01223', '01229', '01233', '01236', '01244', '01250', '01253', '01258', '01263', '01270', '01278', '01293', '01295', '01301', '01311', '01314', '01323', '01329', '01338', '01339', '01342', '01343', '01348', '01356', '01363', '01367', '01370', '01384', '01393', '01398', '01404', '01405', '01410', '01416', '01419', '01423', '01424', '01426', '01431', '01433', '01438', '01444', '01451', '01460', '01463', '01466', '01468', '01469', '01479', '01481', '01485', '01487', '01492', '01498', '01505', '01519', '01520', '01525', '01526', '01537', '01547', '01561', '01566', '01570', '01577', '01579', '01582', '01583', '01585', '01588', '01589', '01593', '01594', '01599', '01603', '01604', '01605', '01606', '01610', '01611', '01620', '01622', '01627', '01637', '01646', '01649', '01650', '01652', '01657', '01658', '01668', '01670', '01672', '01678', '01680', '01681', '01686', '01694', '01710', '01711', '01716', '01721', '01726', '01732', '01733', '01735', '01737', '01740', '01744', '01748', '01754', '01766', '01767', '01773', '01779', '01781', '01786', '01790', '01793', '01796', '01804', '01806', '01807', '01809', '01811', '01821', '01831', '01836', '01842', '01845', '01851', '01852', '01858', '01865', '01877', '01887', '01895', '01896', '01901', '01903', '01906', '01912', '01922', '01924', '01929', '01931', '01932', '01935', '01940', '01941', '01946', '01964', '01966', '01969', '01976', '01981', '01985', '01987', '01990', '01993', '01995', '02007', '02047', '02051', '02052', '02057', '02062', '02069', '02073', '02076', '02081', '02083', '02100', '02101', '02102', '02110', '02119', '02125', '02129', '02130', '02131', '02135', '02136', '02145', '02150', '02163', '02167', '02189', '02202', '02216', '02219', '02229', '02230', '02233', '02234', '02235', '02236', '02247', '02256', '02258', '02261', '02263', '02265', '02266', '02267', '02273', '02278', '02287', '02288', '02291', '02307', '02311', '02322', '02335', '02343', '02352', '02355', '02356', '02359', '02373', '02374', '02380', '02385', '02387', '02392', '02396', '02400', '02407', '02416', '02417', '02418', '02424', '02425', '02429', '02441', '02452', '02456', '02491', '02501', '02503', '02513', '02515', '02517', '02518', '02520', '02526', '02530', '02537', '02539', '02542', '02543', '02548', '02550', '02561', '02562', '02569', '02580', '02581', '02584', '02591', '02594', '02604', '02605', '02607', '02624', '02631', '02632', '02634', '02642', '02645', '02653', '02660', '02661', '02669', '02670', '02672', '02676', '02678', '02684', '02694', '02696', '02697', '02701', '02709', '02711', '02713', '02734', '02736', '02737', '02743', '02752', '02767', '02770', '02776', '02779', '02783', '02784', '02790', '02791', '02793', '02797', '02808', '02810', '02814', '02820', '02826', '02828', '02829', '02833', '02834', '02839', '02852', '02854', '02856', '02858', '02859', '02860', '02863', '02865', '02870', '02874', '02886', '02888', '02894', '02900', '02905', '02906', '02911', '02913', '02917', '02922', '02928', '02942', '02945', '02946', '02952', '02958', '02973', '02975', '02976', '02977', '02982', '02995', '02996', '02997', '03001', '03006', '03013', '03015', '03018', '03024', '03038', '03048', '03056', '03057', '03061', '03071', '03072', '03078', '03084', '03087', '03099', '03101', '03116', '03118', '03120', '03128', '03133', '03143', '03145', '03149', '03150', '03152', '03154', '03157', '03158', '03161', '03162', '03164', '03169', '03170', '03172', '03184', '03191', '03192', '03207', '03208', '03213', '03219', '03220', '03237', '03241', '03247', '03251', '03256', '03258', '03271', '03273', '03283', '03288', '03290', '03297', '03302', '03307', '03308', '03333', '03340', '03343', '03345', '03352', '03362', '03366', '03368', '03379', '03391', '03395', '03397', '03398', '03416', '03419', '03430', '03431', '03440', '03447', '03450', '03459', '03467', '03469', '03489', '03491', '03492', '03501', '03503', '03507', '03511', '03518', '03519', '03520', '03539', '03547', '03556', '03569', '03571', '03574', '03595', '03604', '03617', '03621', '03631', '03638', '03640', '03647', '03648', '03656', '03662', '03668', '03670', '03672', '03675', '03680', '03682', '03690', '03691', '03693', '03694', '03706', '03713', '03716', '03717', '03724', '03731', '03733', '03734', '03747', '03758', '03781', '03782', '03791', '03794', '03795', '03801', '03802', '03813', '03815', '03817', '03821', '03826', '03830', '03832', '03856', '03861', '03864', '03867', '03873', '03877', '03883', '03889', '03891', '03892', '03895', '03896', '03900', '03901', '03904', '03909', '03912', '03915', '03920', '03925', '03944', '03948', '03951', '03956', '03960', '03967', '03968', '03970', '03981', '03986', '03997', '03998', '04002', '04004', '04016', '04019', '04022', '04023', '04027', '04032', '04041', '04046', '04049', '04050', '04051', '04054', '04055', '04059', '04060', '04064', '04072', '04074', '04075', '04077', '04082', '04087', '04096', '04097', '04104', '04105', '04115', '04116', '04117', '04120', '04122', '04128', '04130', '04140', '04142', '04146', '04149', '04152', '04156', '04165', '04170', '04175', '04180', '04191', '04192', '04202', '04212', '04213', '04222', '04226', '04232', '04235', '04237', '04254', '04258', '04259', '04267', '04279', '04283', '04287', '04289', '04290', '04291', '04295', '04298', '04313', '04317', '04319', '04326', '04328', '04329', '04363', '04364', '04365', '04368', '04372', '04373', '04376', '04386', '04387', '04389', '04390', '04396', '04398', '04402', '04405', '04408', '04417', '04438', '04439', '04446', '04456', '04460', '04464', '04466', '04467', '04474', '04476', '04489', '04490', '04494', '04498', '04507', '04518', '04519', '04520', '04522', '04536', '04546', '04550', '04560', '04581', '04586', '04589', '04597', '04598', '04601', '04604', '04607', '04608', '04616', '04622', '04624', '04626', '04637', '04641', '04644', '04645', '04650', '04656', '04660', '04669', '04672', '04676', '04678', '04681', '04685', '04689', '04690', '04693', '04694', '04698', '04706', '04707', '04715', '04728', '04729', '04742', '04744', '04751', '04752', '04753', '04756', '04758', '04760', '04767', '04794', '04798', '04800', '04801', '04803', '04813', '04815', '04818', '04825', '04827', '04832', '04842', '04861', '04862', '04875', '04881', '04889', '04890', '04895', '04902', '04923', '04934', '04943', '04950', '04955', '04961', '04972', '04977', '04978', '04980', '04987', '04995', '04998', '05005', '05006', '05013', '05016', '05019', '05025', '05027', '05029', '05036', '05037', '05042', '05046', '05056', '05062', '05063', '05066', '05068', '05081', '05089', '05091', '05092', '05093', '05095', '05097', '05099', '05103', '05108', '05111', '05112', '05116', '05117', '05124', '05125', '05127', '05128', '05135', '05136', '05138', '05139', '05144', '05147', '05148', '17219', '17235', '17239', '17245', '17250', '17258', '17263', '17275', '17276', '17278', '17286', '17287', '17291', '17293', '17294', '17297', '17307', '17309', '17313', '17318', '17324', '17329', '17333', '17334', '17341', '17347', '17359', '17362', '17364', '17367', '17371', '17378', '17385', '17386', '17389', '17402', '17417', '17418', '17421', '17425', '17439', '17441', '17446', '17447', '17455', '17457', '17462', '17466', '17475', '17477', '17483', '17485', '17497', '17506', '17514', '17518', '17521', '17527', '17528', '17538', '17539', '17541', '17546', '17547', '17550', '17554', '17562', '17564', '17565', '17566', '17580', '17583', '17597', '17599', '17601', '17602', '17603', '17606', '17611', '17612', '17616', '17621', '17623', '17635', '17637', '17656', '17662', '17665', '17666', '17674', '17681', '17682', '17683', '17690', '17693', '17696', '17704', '17705', '17719', '17722', '17727', '17729', '17730', '17741', '17745', '17746', '17747', '17751', '17753', '17756', '17757', '17761', '17763', '17765', '17768', '17770', '17771', '17784', '17786', '17788', '17800', '17801', '17805', '17806', '17807', '17809', '17815', '17821', '17828', '17829', '17831', '17836', '17837', '17851', '17855', '17863', '17865', '17873', '17876', '17877', '17883', '17884', '17893', '17894', '17897', '17906', '17908', '17911', '17919', '17932', '17933', '17937', '17946', '17947', '17949', '17953', '17957', '17962', '17968', '17974', '17987', '17990', '17994', '17995', '18002', '18007', '18008', '18011', '18018', '18022', '18026', '18028', '18033', '18037', '18041', '18043', '18045', '18048', '18055', '18063', '18075', '18078', '18080', '18082', '18092', '18094', '18097', '18104', '18106', '18107', '18110', '18111', '18112', '18126', '18128', '18152', '18159', '18162', '18171', '18172', '18178', '18179', '18180', '18181', '18195', '18197', '18206', '18208', '18209', '18211', '18221', '18228', '18232', '18237', '18238', '18239', '18241', '18253', '18270', '18280', '18288', '18292', '18294', '18299', '18301', '18302', '18304', '18312', '18319', '18320', '18323', '18327', '18335', '18337', '18342', '18353', '18354', '18357', '18358', '18360', '18364', '18365', '18375', '18377', '18381', '18382', '18383', '18393', '18394', '18401', '18405', '18414', '18417', '18419', '18431', '18434', '18436', '18441', '18446', '18447', '18451', '18453', '18459', '18460', '18462', '18466', '18468', '18477', '18482', '18490', '18492', '18493', '18496', '18499', '18501', '18511', '18518', '18532', '18542', '18543', '18546', '18551', '18556', '18561', '18566', '18570', '18573', '18582', '18587', '18596', '18601', '18602', '18616', '18617', '18620', '18632', '18648', '18653', '18655', '18660', '18661', '18665', '18676', '18680', '18681', '18683', '18684', '18692', '18694', '18698', '18702', '18710', '18712', '18716', '18718', '18725', '18734', '18739', '18741', '18743', '18754', '18756', '18759', '18761', '18771', '18777', '18778', '18780', '18783', '18786', '18796', '18797', '18802', '18813', '18814', '18815', '18831', '18835', '18836', '18841', '18849', '18850', '18853', '18857', '18859', '18865', '18872', '18878', '18881', '18887', '18892', '18900', '18911', '18912', '18916', '18922', '18928', '18940', '18949', '18959', '18960', '18962', '18981', '18986', '18988', '18989', '18994', '19001', '19008', '19017', '19026', '19034', '19038', '19040', '19045', '19046', '19047', '19050', '19052', '19054', '19058', '19059', '19060', '19066', '19071', '19074', '19077', '19079', '19088', '19096', '19098', '19101', '19102', '19108', '19111', '19114', '19120', '19121', '19125', '19128', '19130', '19136', '19147', '19151', '19154', '19155', '19165', '19168', '19170', '19181', '19189', '19194', '19196', '19202', '19204', '19207', '19211', '19214', '19220', '19225', '19229', '19235', '19237', '19240', '19251', '19257', '19259', '19266', '19267', '19268', '19274', '19277', '19285', '19297', '19298', '19304', '19309', '19313', '19320', '19324', '19326', '19327', '19331', '19335', '19345', '19347', '19348', '19357', '19359', '19364', '19370', '19373', '19374', '19375', '19377', '19383', '19384', '19386', '19388', '19398', '19400', '19405', '19408', '19409', '19419', '19421', '19423', '19426', '19428', '19431', '19432', '19437', '19440', '19441', '19443', '19448', '19451', '19452', '19455', '19459', '19460', '19464', '19467', '19475', '19480', '19483', '19487', '19496', '19498', '19512', '19520', '19528', '19532', '19534', '19535', '19542', '19550', '19555', '19561', '19572', '19573', '19579', '19581', '19586', '19587', '19591', '19604', '19606', '19636', '19650', '19652', '19654', '19655', '19658', '19660', '19665', '19673', '19675', '19690', '19708', '19711', '19718', '19729', '19733', '19745', '19749', '19755', '19756', '19757', '19764', '19772', '19776', '19778', '19786', '19787', '19789', '19790', '19792', '19793', '19796', '19800', '19801', '19805', '19820', '19823', '19829', '19840', '19845', '19857', '19866', '19871', '19879', '19881', '19890', '19904', '19913', '19918', '19923', '19931', '19932', '19936', '19937', '19941', '19945', '19956', '19962', '19965', '19966', '19976', '19978', '19979', '19984', '19989', '19993', '19994', '19998', '20002', '20003', '20007', '20011', '20012', '20014', '20018', '20019', '20020', '20032', '20034', '20037', '20039', '20040', '20041', '20052', '20065', '20074', '20084', '20085', '20086', '20087', '20088', '20098', '20100', '20117', '20118', '20129', '20130', '20144', '20161', '20166', '20170', '20179', '20183', '20186', '20188', '20189', '20215', '20237', '20251', '20253', '20255', '20257', '20265', '20266', '20269', '20274', '20275', '20276', '20280', '20283', '20284', '20285', '20286', '20289', '20290', '20303', '20307', '20313', '20329', '20339', '20342', '20348', '20352', '20366', '20367', '20374', '20375', '20376', '20382', '20389', '20391', '20393', '20396', '20399', '20402', '20403', '20405', '20414', '20416', '20424', '20435', '20438', '20441', '20449', '20452', '20461', '20463', '20465', '20468', '20472', '20477', '20483', '20490', '20494', '20495', '20504', '20511', '20515', '20516', '20518', '20520', '20523', '20526', '20529', '20530', '20535', '20541', '20561', '20566', '20569', '20573', '20579', '20595', '20599', '20605', '20606', '20608', '20614', '20618', '20621', '20629', '20631', '20643', '20644', '20662', '20669', '20677', '20679', '20682', '20688', '20691', '20692', '20701', '20709', '20711', '20713', '20721', '20724', '20729', '20737', '20739', '20740', '20741', '20743', '20755', '20768', '20776', '20782', '20787', '20793', '20800', '20803', '20804', '20813', '20816', '20817', '20820', '20822', '20829', '20835', '20837', '20838', '20846', '20849', '20858', '20866', '20874', '20888', '20889', '20892', '20898', '20919', '20921', '20923', '20929', '20933', '20936', '20943', '20944', '20946', '20951', '20958', '20962', '20964', '20966', '20976', '20977', '20988', '20996', '21010', '21011', '21012', '21026', '21036', '21042', '21051', '21053', '21054', '21060', '21065', '21068', '21078', '21080', '21082', '21083', '21096', '21102', '21106', '21112', '21115', '21116', '21131', '21133', '21138', '21149', '21155', '21158', '21162', '21164', '21172', '21175', '21193', '21202', '21203', '21213', '21214', '21218', '21219', '21223', '21224', '21233', '21235', '21239', '21244', '21247', '21251', '21253', '21256', '21260', '21265', '21270', '21273', '21276', '21277', '21284', '21286', '21289', '21293', '21294', '21304', '21310', '21313', '21314', '21318', '21333', '21340', '21348', '21353', '21354', '21363', '21369', '21373', '21380', '21381', '21383', '21387', '21392', '21394', '21403', '21406', '21407', '21411', '21412', '21425', '21426', '21428', '21435', '21439', '21442', '21444', '21447', '21449', '21455', '21457', '21468', '21471', '21483', '21485', '21488', '21489', '21495', '21496', '21497', '21504', '21505', '21506', '21507', '21510', '21512', '21516', '21518', '21519', '21521', '21530', '21537', '21540', '21542', '21544', '21550', '21561', '21565', '21570', '21572', '21577', '21579', '21590', '21594', '21595', '21597', '21603', '21605', '21607', '21613', '21614', '21623', '21626', '21632', '21639', '21640', '21644', '21645', '21648', '21649', '21652', '21655', '21659', '21663', '21664', '21666', '21673', '21674', '21683', '21686', '21690', '21695', '21702', '21703', '21705', '21709', '21713', '21717', '21724', '21728', '21742', '21743', '21752', '21758', '21770', '21772', '21783', '21784', '21793', '21805', '21809', '21821', '21827', '21838', '21843', '21844', '21846', '21851', '21872', '21875', '21877', '21878', '21887', '21891', '21896', '21903', '21907', '21915', '21916', '21919', '21920', '21922', '21929', '21936', '21937', '21946', '21947', '21949', '21951', '21954', '21955', '21957', '21958', '21960', '21964', '21968', '21971', '21973', '21979', '21981', '21986', '21988', '21989', '21993', '21998', '22004', '22005', '22006', '22010', '22045', '22048', '22056', '22062', '22063', '22069', '22074', '22079', '22095', '22097', '22100', '22105', '22106', '22111', '22122', '22123', '22124', '22127', '22128', '22129'],
            img_transform=transform_val,
            is_validation=True,
        )
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

    #for i, (input_img, ref_img, gt_normals_input) in enumerate(train_loader):
    for i, (input_img, ref_img) in enumerate(train_loader):
        """if i>=2:
            break  # Limit to first 5 batches for debugging"""
        current_step = global_step + i
        #gt_normals_ref = gt_normals_input.clone()
        
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
        if i%100 == 0 or i==0 and args.is_master and not args.disable_wandb:
            try:
                print(f"Training image visualization at step {current_step}")
                # Standard relighting visualization
                log_relighting_results(
                    input_img=input_img,
                    ref_img=ref_img,
                    relit_img=recon_img_ir,
                    gt_img=ref_img,
                    step=current_step
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
                gt_normals_input, gt_normals_ref = depth_normal_loss_fn.get_normal_map(input_img), depth_normal_loss_fn.get_normal_map(ref_img)
                if gt_normals_input is not None:
                    _, _, normals_input_loss_value, _, _, _, _ = depth_normal_loss_fn(gt_normals_input.to(args.gpu), normals_input.to(args.gpu))
                    normals_decoder_loss_value += normals_input_loss_value
                    
                # Also add consistency loss between input and reference normals when they should be similar
                if gt_normals_ref is not None:
                    _, _, normals_ref_loss_value , _, _, _, _ = depth_normal_loss_fn(gt_normals_ref.to(args.gpu), normals_ref.to(args.gpu))
                    normals_decoder_loss_value += normals_ref_loss_value

                if i%100 == 0 or i==0 and args.is_master and not args.disable_wandb: # Log normals decoder results every 100 steps
                    try:
                        def normalize_normals_for_logging(normals: torch.Tensor) -> torch.Tensor:
                            """
                            Convert normalized surface normals (range [-1, 1]) to uint8 image format (range [0, 255]).
                            
                            Args:
                                normals (torch.Tensor): shape [B, 3, H, W], values in [-1, 1]

                            Returns:
                                torch.Tensor: shape [B, 3, H, W], values in [0, 255], dtype=torch.uint8
                            """
                            normals = torch.clamp(normals, -1, 1)         # Ensure range is bounded
                            img = (normals + 1.0) * 127.5                 # Map from [-1,1] → [0,255]
                            return img#.to(dtype=torch.uint8)

                        log_relighting_results(
                            input_img=normalize_normals_for_logging(gt_normals_input.to(args.gpu)),
                            ref_img=normalize_normals_for_logging(normals_input.to(args.gpu)),
                            relit_img=normalize_normals_for_logging(gt_normals_ref.to(args.gpu)),
                            gt_img=normalize_normals_for_logging(normals_ref.to(args.gpu)),
                            step=current_step,
                            mode='normals_decoder'
)
                    except Exception as e:
                        print(f"Warning: Failed to log normals decoder results: {e}")
                    
                        
            except Exception as e:
                print(f"Warning: Normals decoder loss computation failed: {e}")
                normals_decoder_loss_value = torch.tensor(0.0, device=input_img.device)
        
        normals_decoder_loss_component = getattr(args, 'normals_decoder_loss_weight', 0.0) * normals_decoder_loss_value
       
        # Total loss
        total_loss = (rec_loss_total + logdet_reg_loss + intrinsic_loss_component + 
                     depth_loss_component + normal_loss_component + logdet_ext_reg_loss +
                     normals_decoder_loss_component )
        #print(f"Total loss: {total_loss.item()} | Rec: {rec_loss_total.item()} | Logdet: {logdet_reg_loss.item()} | Intrinsic: {intrinsic_loss_component.item()} | Depth: {depth_loss_component.item()} | Normal: {normal_loss_component.item()} | Logdet Ext: {logdet_ext_reg_loss.item()} | Normals Decoder: {normals_decoder_loss_component.item()}")
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
                
                if len(batch) > 3:
                    input_img, ref_img, gt_img, gt_normals_input, gt_normals_ref = batch
                elif len(batch) == 3:
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
                        
                        
                        
                        if gt_normals_input is not None:
                            _, _, normals_input_loss_value, _, _, _, _ = depth_normal_loss_fn(gt_normals_input.to(args.gpu), normals_input.to(args.gpu))
                            normals_decoder_loss_value += normals_input_loss_value
                            
                        # Also add consistency loss between input and reference normals when they should be similar
                        if gt_normals_ref is not None:
                            _, _, normals_ref_loss_value , _, _, _, _ = depth_normal_loss_fn(gt_normals_ref.to(args.gpu), normals_ref.to(args.gpu))
                            normals_decoder_loss_value += normals_ref_loss_value
                        
                        # Log visualizations for first batch only
                        if i == 0 and args.is_master and not args.disable_wandb:
                            try:
                                def normalize_normals_for_logging(normals):
                                    return (normals + 1) / 2  # maps [-1, 1] to [0, 1]
                                # Standard relighting visualization
                                log_relighting_results(
                                    input_img=normalize_normals_for_logging(gt_normals_input.to(args.gpu)),
                                    ref_img=normalize_normals_for_logging(normals_input.to(args.gpu)),
                                    relit_img=normalize_normals_for_logging(gt_normals_ref.to(args.gpu)),
                                    gt_img=normalize_normals_for_logging(normals_ref.to(args.gpu)),
                                    step=current_step,
                                    mode=f'{dataset_name}_validation'
                                    )
                            except:
                                print(f"Warning: Failed to log normalsj {dataset_name} validation")
                            
                                
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