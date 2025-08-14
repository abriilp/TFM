import subprocess
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils.dataset_utils import PromptTrainDataset, ALNDatasetRLSID
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Removed the problematic import
# from utils.val_utils import compute_psnr_ssim
from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = LPIPS(net="vgg").requires_grad_(False)
        self.ssim_loss = SSIM()
        self.lpips_lambda = 0.1
        self.ssim_lambda = 0.2
        
        # Visualization settings
        self.val_vis_frequency = 50  # Log images every N validation steps
        self.train_vis_frequency = 500  # Log images every N training steps
        self.save_vis_locally = True  # Save visualizations locally as well
        
        # Add step counter for validation to ensure proper WandB logging
        self.val_step_counter = 0
    
    def forward(self, x):
        return self.net(x)
    
    def denormalize_image(self, tensor):
        """Convert tensor to displayable format (0-1 range)"""
        # Clamp to [0, 1] range
        return torch.clamp(tensor, 0, 1)
    
    def create_comparison_grid(self, concatenated_input, restored, target, max_images=4):
        """Create a grid comparing input, reference, restored, and target images"""
        batch_size = min(concatenated_input.size(0), max_images)
        
        # Extract input and reference from concatenated input
        # concatenated_input has 6 channels: first 3 are input, last 3 are reference
        input_img = concatenated_input[:batch_size, :3, :, :]  # First 3 channels
        reference_img = concatenated_input[:batch_size, 3:6, :, :]  # Last 3 channels
        
        # Take only the first few images from batch
        restored = restored[:batch_size]
        target = target[:batch_size]
        
        # Denormalize images
        input_img = self.denormalize_image(input_img)
        reference_img = self.denormalize_image(reference_img)
        restored = self.denormalize_image(restored)
        target = self.denormalize_image(target)
        
        # Create grid: [input, reference, restored, target] for each image
        comparison_images = []
        for i in range(batch_size):
            comparison_images.extend([input_img[i], reference_img[i], restored[i], target[i]])
        
        # Stack all images
        grid = torch.stack(comparison_images)
        
        # Create grid with 4 columns (input, reference, restored, target)
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
            plt.title(f'{phase.capitalize()} - Step {step}\nColumns: Input | Reference | Restored | Target')
            plt.tight_layout()
            plt.savefig(f"visualizations/{phase}/step_{step}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # Compute losses
        l1_loss = self.l1_loss(restored, clean_patch)
        lpips_loss = self.lpips_loss(restored, clean_patch)
        ssim_loss = self.ssim_loss(restored, clean_patch)
        
        # Take mean of losses to ensure they are scalars
        l1_loss = l1_loss.mean() if l1_loss.dim() > 0 else l1_loss
        lpips_loss = lpips_loss.mean() if lpips_loss.dim() > 0 else lpips_loss
        ssim_loss = ssim_loss.mean() if ssim_loss.dim() > 0 else ssim_loss
        
        total_loss = l1_loss + self.lpips_lambda * lpips_loss + self.ssim_lambda * (1 - ssim_loss)

        # Logging to TensorBoard (if installed) by default
        self.log("l1_loss", l1_loss, sync_dist=True)
        self.log("lpips_loss", lpips_loss, sync_dist=True)
        self.log("ssim_loss", ssim_loss, sync_dist=True)
        self.log("total_loss", total_loss, sync_dist=True)
        
        # Visualize training progress periodically
        if batch_idx % self.train_vis_frequency == 0 and self.global_rank == 0:
            with torch.no_grad():
                grid = self.create_comparison_grid(degrad_patch, restored, clean_patch)
                
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
        ([clean_name, de_id], degrad_patch, target_patch) = batch

        with torch.no_grad():
            restored = self.net(degrad_patch)

        # Use PyTorch-based metrics instead of sklearn-based compute_psnr_ssim
        # Calculate PSNR manually
        mse = torch.mean((restored - target_patch) ** 2)
        psnr_i = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # Use torchmetrics SSIM which handles the tensor format correctly
        ssim_i = self.ssim_loss(restored, target_patch)

        self.log("val_psnr", psnr_i, sync_dist=True)
        self.log("val_ssim", ssim_i, sync_dist=True)
        
        # Visualize validation results periodically
        if batch_idx % self.val_vis_frequency == 0 and self.global_rank == 0:
            grid = self.create_comparison_grid(degrad_patch, restored, target_patch)
            
            # Log to tensorboard/wandb
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_image(
                    'val/comparison', grid, self.global_step
                )
            elif isinstance(self.logger, WandbLogger):
                import wandb
                # Use the same key name as training for consistency
                self.logger.experiment.log({
                    'val/comparison': wandb.Image(grid.cpu().numpy().transpose(1, 2, 0))
                }, step=self.global_step)
            
            # Save locally with epoch info for reference
            self.save_visualization_locally(grid, f"{self.current_epoch:03d}_batch_{batch_idx:03d}", "val")
        
        return {"val_psnr": psnr_i, "val_ssim": ssim_i}
    
    def on_validation_epoch_start(self):
        """Reset validation step counter at the start of each validation epoch"""
        self.val_step_counter = 0
    
    def on_validation_epoch_end(self):
        """Create a summary visualization at the end of each validation epoch"""
        if self.global_rank == 0:
            print(f"Validation epoch {self.current_epoch} completed.")
            print(f"Visualizations saved in: visualizations/val/")
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    print("CUDA devices set to:", os.environ["CUDA_VISIBLE_DEVICES"])

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train-inputref")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # For training
    trainset = ALNDatasetRLSID(
        root_dir=opt.root_dir,
        is_validation=False,
        resize_width_to=512,
        patch_size=256,
        # filter_of_images=[1, 2, 3],  # Optional: only use scenes 001, 002, 003
        max_training_images=30000
    )

    # For validation  
    testset = ALNDatasetRLSID(
        root_dir=opt.root_dir,
        is_validation=True,
        resize_width_to=512,
        patch_size=256,
        max_validation_images=5000
    )

    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    print(f"Training will save visualizations every {model.train_vis_frequency} training steps")
    print(f"Validation will save visualizations every {model.val_vis_frequency} validation steps")
    print("Visualizations will be saved locally in './visualizations/' directory")
    print("Grid format: Input | Reference | Restored | Target")
    
    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, 
                        strategy="ddp_find_unused_parameters_true", logger=logger, 
                        callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader) #, ckpt_path=opt.load_ckpt


if __name__ == '__main__':
    main()