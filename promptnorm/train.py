import subprocess
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from utils.aln_dataset import ALNDatasetGeom
from model import PromptNorm
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.val_utils import compute_psnr_ssim
from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class PromptNormModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.net = PromptIR(decoder=True)
        self.net = PromptNorm(decoder=True)
        self.l1_loss  = nn.L1Loss()
        self.lpips_loss = LPIPS(net="vgg").requires_grad_(False)
        self.ssim_loss = SSIM()
        self.lpips_lambda = 0.1
        self.ssim_lambda = 0.2
        
        # Visualization settings
        self.val_vis_frequency = 50  # Log images every N validation steps
        self.train_vis_frequency = 500  # Log images every N training steps
        self.save_vis_locally = False  # Save visualizations locally as well
        
        # Add step counter for validation to ensure proper WandB logging
        self.val_step_counter = 0
        
    def forward(self,x):
        return self.net(x)
    
    def denormalize_image(self, tensor):
        """Convert tensor to displayable format (0-1 range)"""
        # Clamp to [0, 1] range
        return torch.clamp(tensor, 0, 1)
    
    def create_comparison_grid(self, degraded, depth, restored, target, max_images=4):
        """Create a grid comparing input, depth, restored, and target images"""
        batch_size = min(degraded.size(0), max_images)
        
        # Take only the first few images from batch
        degraded = degraded[:batch_size][:, :3, :, :]
        depth = depth[:batch_size]
        restored = restored[:batch_size]
        target = target[:batch_size]
        
        # Denormalize images
        degraded = self.denormalize_image(degraded)
        restored = self.denormalize_image(restored)
        target = self.denormalize_image(target)
       
        # Handle depth visualization (convert to 3-channel if needed)
        if depth.size(1) == 1:
            depth = depth.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        depth = self.denormalize_image(depth)
        
        # Create grid: [degraded, depth, restored, target] for each image
        comparison_images = []
        for i in range(batch_size):
            comparison_images.extend([degraded[i], depth[i], restored[i], target[i]])
        
        # Stack all images
        grid = torch.stack(comparison_images)
        
        # Create grid with 4 columns (degraded, depth, restored, target)
        grid = vutils.make_grid(grid, nrow=4, padding=2, normalize=False, pad_value=1.0)
        
        return grid
    
    def save_visualization_locally(self, grid, step, phase="train"):
        """Save visualization grid locally"""
        if self.save_vis_locally:
            os.makedirs(f"visualizations/{phase}", exist_ok=True)
            
            # Convert to numpy and save
            grid_np = grid.cpu().numpy().transpose(1, 2, 0)
            plt.figure(figsize=(16, 4 * (grid_np.shape[0] // 4)))
            plt.imshow(grid_np)
            plt.axis('off')
            plt.title(f'{phase.capitalize()} - Step {step}\nColumns: Degraded | Depth/Normals | Restored | Target')
            plt.tight_layout()
            plt.savefig(f"visualizations/{phase}/step_{step}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, depth_patch, clean_patch) = batch
        restored = self.net(degrad_patch, depth_patch)

        # Compute losses
        l1_loss = self.l1_loss(restored,clean_patch)
        lpips_loss = self.lpips_loss(restored,clean_patch)
        ssim_loss = 1 - self.ssim_loss(restored,clean_patch)
        total_loss = l1_loss + self.lpips_lambda * lpips_loss + self.ssim_lambda * ssim_loss

        # Logging to TensorBoard (if installed) by default
        self.log("l1_loss", l1_loss, sync_dist=True)
        self.log("lpips_loss", lpips_loss, sync_dist=True)
        self.log("ssim_loss", ssim_loss, sync_dist=True)
        self.log("total_loss", total_loss, sync_dist=True)
        
        # Visualize training progress periodically
        if batch_idx % self.train_vis_frequency == 0 and self.global_rank == 0:
            with torch.no_grad():
                grid = self.create_comparison_grid(degrad_patch, depth_patch, restored, clean_patch)
                
                # Log to tensorboard/wandb
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_image(
                        'train/comparison', grid, self.global_step
                    )
                elif isinstance(self.logger, WandbLogger):
                    import wandb
                    # Use global_step for consistency in training
                    self.logger.experiment.log({
                        'train/comparison': wandb.Image(grid.cpu().numpy().transpose(1, 2, 0))
                    }, step=self.global_step)
                
                # Save locally
                self.save_visualization_locally(grid, self.global_step, "train")

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        ([clean_name, de_id], degrad_patch, depth_patch, target_patch) = batch

        with torch.no_grad():
            restored = self.net(degrad_patch, depth_patch)

        psnr_i, ssim_i, _ = compute_psnr_ssim(restored, target_patch)

        self.log("val_psnr", psnr_i, sync_dist=True)
        self.log("val_ssim", ssim_i, sync_dist=True)
        
        # Visualize validation results periodically
        if batch_idx % self.val_vis_frequency == 0 and self.global_rank == 0:
            grid = self.create_comparison_grid(degrad_patch, depth_patch, restored, target_patch)
            
            # Log to tensorboard/wandb
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_image(
                    f'val/comparison_batch_{batch_idx}', grid, self.current_epoch
                )
            elif isinstance(self.logger, WandbLogger):
                import wandb
                # Use global_step to maintain consistency with training logs
                # This ensures WandB doesn't complain about step ordering
                self.logger.experiment.log({
                    f'val/comparison_batch_{batch_idx}': wandb.Image(grid.cpu().numpy().transpose(1, 2, 0))
                }, step=self.global_step)
            
            # Save locally
            self.save_visualization_locally(grid, f"{self.current_epoch:03d}_batch_{batch_idx:03d}", "val")
            
            # Log metrics for this specific batch
            self.log(f"val_psnr_batch_{batch_idx}", psnr_i, sync_dist=True)
            self.log(f"val_ssim_batch_{batch_idx}", ssim_i, sync_dist=True)
        
        return {"val_psnr": psnr_i, "val_ssim": ssim_i}
    
    def on_validation_epoch_start(self):
        """Reset validation step counter at the start of each validation epoch"""
        self.val_step_counter = 0
    
    def on_validation_epoch_end(self):
        """Create a summary visualization at the end of each validation epoch"""
        if self.global_rank == 0:
            print(f"Validation epoch {self.current_epoch} completed.")
            print(f"Visualizations saved in: visualizations/val/")

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)

    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptNorm-Train-Viz")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)

    trainset = ALNDatasetGeom(input_folder="/home/apinyol/TFM/Data/prova_promptnorm2/train/input",#opt.train_input_dir,
                               geom_folder="/home/apinyol/TFM/Data/prova_promptnorm2/train/normals",#opt.train_depth_dir,
                               target_folder="/home/apinyol/TFM/Data/prova_promptnorm2/train/target",#opt.train_target_dir,
                               reference_folder="/home/apinyol/TFM/Data/prova_promptnorm2/train/target",#opt.train_target_dir,
                               resize_width_to=opt.resize_width,
                               patch_size=opt.patch_size)

    testset = ALNDatasetGeom(input_folder="/home/apinyol/TFM/Data/prova_promptnorm2/test/input", 
                             geom_folder="/home/apinyol/TFM/Data/prova_promptnorm2/test/normals", 
                             target_folder="/home/apinyol/TFM/Data/prova_promptnorm2/test/target", 
                             reference_folder="/home/apinyol/TFM/Data/prova_promptnorm2/test/reference",
                             resize_width_to=opt.resize_width, 
                             patch_size=opt.patch_size)
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)
    
    model = PromptNormModel()
    
    print(f"Training will save visualizations every {model.train_vis_frequency} training steps")
    print(f"Validation will save visualizations every {model.val_vis_frequency} validation steps")
    print("Visualizations will be saved locally in './visualizations/' directory")
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == '__main__':
    main()