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


from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import lpips
from skimage import color
import wandb

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
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=0)
# training params
parser.add_argument("--gpus", type=int, default=1)
# datamodule params
parser.add_argument("--data_path", type=str, default="/home/apinyol/TFM/Data/RSR_256")
parser.add_argument("--load_ckpt", type=str, default="/home/apinyol/TFM/Models/last.pth.tar")

args = parser.parse_args()


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
    val_dataset = RSRDataset(root_dir=args.data_path, img_transform=transform_test, 
                                is_validation=True, validation_scenes=["scene_01", "scene_03", "scene_05", "scene_06", "scene_07", "scene_08", "scene_10", "scene_21", "scene_22", "scene_23", "scene_24", "scene_26", "scene_27", "scene_29", "scene_30"])
    val_dataset2 = MIT_Dataset('/home/apinyol/TFM/Data/multi_illumination_test_mip2_jpg', 
                                 transform_test, eval_mode=True)
    #test_dataset = VIDIT_Dataset_val('/net/projects/willettlab/roxie62/dataset/VIDIT', transform_test)
    print('NUM of training images: {}'.format(len(val_dataset)))
    val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last = False, persistent_workers = False)
    val_loader2 = torch.utils.data.DataLoader(
        val_dataset2, batch_size=1, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last = False, persistent_workers = False)
    return val_loader, val_loader2



@torch.no_grad()
def eval_relight(args, epoch, model, eval_pair_folder_shift = 5, eval_pair_light_shift = 5):
    torch.manual_seed(args.gpu)
    val_loader1, val_loader2 = get_eval_relight_dataloder2(args, eval_pair_folder_shift, eval_pair_light_shift)
    t0 = time.time()
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
    wandb_images = []
    for val_loader in [val_loader2]: #val_loader1,
        for i, (img1, img2, img3) in enumerate(tqdm.tqdm(val_loader)):
            img1 = img1.to(args.gpu)  # Input image
            img2 = img2.to(args.gpu)  # Reference image (different lighting)
            img3 = img3.to(args.gpu)  # Ground truth (target lighting)

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
            ssim_score = ssim_metric(relight_img2, img3)
            psnr_score = psnr_metric(relight_img2, img3)
            lpips_score = lpips_metric(relight_img2, img3)
            delta_e_score = delta_e_metric(relight_img2, img3)
            """
            #identitat
            ssim_score = ssim_metric(img1, img3)
            psnr_score = psnr_metric(img1, img3)
            lpips_score = lpips_metric(img1, img3)
            delta_e_score = delta_e_metric(img1, img3)
            """

            # Accumulate scores
            ssim_scores.append(ssim_score.item())
            psnr_scores.append(psnr_score.item())
            lpips_scores.append(lpips_score.item())
            delta_e_scores.append(delta_e_score)

            # Prepare images for wandb logging (convert from [-1, 1] to [0, 1])
            def normalize_for_display(img_tensor):
                return (img_tensor + 1.0) / 2.0
            
            #print(f"Batch {i} | SSIM: {ssim_score.item():.3f}, PSNR: {psnr_score.item():.2f}, LPIPS: {lpips_score.item():.3f}, ΔE: {delta_e_score:.3f}")
            batch_size = img1.shape[0]
            for j in range(min(batch_size, 4)):  # Max 4 images per batch
                # Create a grid of images: [Input, Reference, Relit, Ground Truth]
                input_img = normalize_for_display(img1[j]).cpu()
                ref_img = normalize_for_display(img2[j]).cpu()
                relit_img = normalize_for_display(relight_img2[j]).cpu()
                gt_img = normalize_for_display(img3[j]).cpu()
                    
                # Create a horizontal grid
                grid_img = torch.cat([input_img, ref_img, relit_img, gt_img], dim=2)
                    
                if i==0 or i%10 == 0:
                    wandb_images.append(wandb.Image(
                            grid_img,
                            caption=f"Batch {i}, Sample {j} | SSIM: {ssim_score.item():.3f}, PSNR: {psnr_score.item():.2f}, LPIPS: {lpips_score.item():.3f}, ΔE: {delta_e_score:.3f}"
                        ))
                    
                # Store metrics for table
                all_metrics.append({
                        "batch_idx": i,
                        "sample_idx": j,
                        "ssim": ssim_score.item(),
                        "psnr": psnr_score.item(),
                        "lpips": lpips_score.item(),
                        "delta_e": delta_e_score,
                        "input_reference_relit_gt": wandb.Image(grid_img, caption="Input | Reference | Relit | Ground Truth")
                    })
                wandb.log({
                    "eval/ssim": ssim_score,
                    "eval/psnr": psnr_score,
                    "eval/lpips": lpips_score,
                    "eval/delta_e": delta_e_score,
                    "eval/num_samples": len(val_loader),
                    "eval/step": i
                    })

        # Calculate average metrics
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)
        avg_lpips = np.mean(lpips_scores)
        avg_delta_e = np.mean(delta_e_scores)

        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  PSNR: {avg_psnr:.4f}")
        print(f"  LPIPS: {avg_lpips:.4f}")
        print(f"  Delta E: {avg_delta_e:.4f}")
        print(f"  Samples: {len(val_loader)}")
        
        wandb.log({
                "eval/avg_ssim": avg_ssim,
                "eval/avg_psnr": avg_psnr,
                "eval/avg_lpips": avg_lpips,
                "eval/avg_delta_e": avg_delta_e,
                "eval/num_samples": len(val_loader),
                "eval/epoch": epoch
            })
        
        # Log image grids
        wandb.log({
                "eval/relight_results": wandb_images
            })
            
        # Create and log detailed results table
        table = wandb.Table(columns=["batch_idx", "sample_idx", "ssim", "psnr", "lpips", "delta_e", "images"])
        for metric_row in all_metrics:
                table.add_data(
                    metric_row["batch_idx"],
                    metric_row["sample_idx"],
                    metric_row["ssim"],
                    metric_row["psnr"],
                    metric_row["lpips"],
                    metric_row["delta_e"],
                    metric_row["input_reference_relit_gt"]
                )
            
        wandb.log({"eval/detailed_results": table})
            
        """# Log metric distributions
        wandb.log({
                "eval/ssim_distribution": wandb.Histogram(ssim_scores),
                "eval/psnr_distribution": wandb.Histogram(psnr_scores),
                "eval/lpips_distribution": wandb.Histogram(lpips_scores),
                "eval/delta_e_distribution": wandb.Histogram(delta_e_scores)
            })"""
        
    return

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

    # Initialize wandb (only on master process)
    if args.rank == 0:
        wandb.init(
            project="relight_evaluation",
            config=vars(args),
            tags=["evaluation", "relight"]
        )

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

    #from eval_utils import eval_relight
    for i in range(1):
        eval_relight(args, 100, model)
    
    wandb.finish()
    
    exit()

if __name__ == '__main__':
    main()