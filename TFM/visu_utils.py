import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import wandb
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_image_grid(input_img, ref_img, relit_img, gt_img):
    """Create a grid with rows for each batch sample and columns for [Input, Reference, Relit, GT]"""

    batch_size = input_img.shape[0]
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    canvas = FigureCanvas(fig)  # Attach canvas to the figure

    if batch_size == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Input', 'Reference', 'Relit', 'Ground Truth']

    for row in range(batch_size):
        images = [input_img[row], ref_img[row], relit_img[row], gt_img[row]]

        for col, (img, title) in enumerate(zip(images, col_titles)):
            img_np = ((img.detach().cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            axes[row, col].imshow(img_np)
            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Render and extract image
    canvas.draw()
    grid_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    grid_array = grid_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    grid_array = grid_array[..., :3]  # Remove alpha channel

    plt.close(fig)
    return grid_array

def create_depth_grid(depth_input, depth_recon):
    """Create a grid with rows for each batch sample and columns for [Input Depth, Relit Depth]"""

    batch_size = depth_input.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(8, 4 * batch_size))
    canvas = FigureCanvas(fig)  # Attach canvas to the figure

    if batch_size == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Input Depth', 'Relit Depth']

    for row in range(batch_size):
        depth_maps = [depth_input[row, 0], depth_recon[row, 0]]

        for col, (depth_map, title) in enumerate(zip(depth_maps, col_titles)):
            depth_np = depth_map.detach().cpu().numpy()

            im = axes[row, col].imshow(depth_np, cmap='viridis')
            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(title, fontsize=12, fontweight='bold')

            if col == 1:
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Render and extract image
    canvas.draw()
    grid_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    grid_array = grid_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    grid_array = grid_array[..., :3]  # Remove alpha channel

    plt.close(fig)
    return grid_array



"""
Visualization utilities for training and validation logging
Clean, modular, and easily configurable visualization functions
"""

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization"""
    if isinstance(tensor, torch.Tensor):
        # Handle different tensor formats
        if tensor.dim() == 4:  # Batch of images
            tensor = tensor.detach().cpu()
            # Clamp values to [0, 1] range
            tensor = torch.clamp(tensor, 0, 1)
            return tensor.numpy()
        elif tensor.dim() == 3:  # Single image
            tensor = tensor.detach().cpu()
            tensor = torch.clamp(tensor, 0, 1)
            return tensor.numpy()
    return tensor


def create_image_grid(images, nrow=4, padding=2, normalize=True):
    """Create a grid of images for visualization"""
    if isinstance(images, torch.Tensor):
        if normalize:
            # Normalize to [0, 1] range
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        grid = make_grid(images, nrow=nrow, padding=padding, normalize=False)
        grid_np = tensor_to_numpy(grid)
        
        # Convert from CHW to HWC for matplotlib
        if grid_np.shape[0] == 3:  # RGB
            grid_np = np.transpose(grid_np, (1, 2, 0))
        elif grid_np.shape[0] == 1:  # Grayscale
            grid_np = grid_np[0]
            
        return grid_np
    return images


def log_training_visualizations(input_img, ref_img, relit_img, step, mode='train', 
                               noisy_input_img=None, noisy_ref_img=None, max_images=4):
    """
    Log training/validation visualizations to wandb
    
    Args:
        input_img: Input image batch
        ref_img: Reference image batch  
        recon_img: Reconstructed image batch
        step: Current step/epoch
        mode: 'train' or 'validation'
        noisy_input_img: Noisy input images (optional)
        noisy_ref_img: Noisy reference images (optional)
        max_images: Maximum number of images to visualize
    """
    
    # Limit number of images to visualize
    batch_size = min(input_img.shape[0], max_images)
    
    # Select subset of images
    input_viz = input_img[:batch_size]
    ref_viz = ref_img[:batch_size]
    recon_viz = relit_img[:batch_size]
    
    # Create image grids
    nrow = min(batch_size, 4)  # Max 4 images per row
    
    input_grid = create_image_grid(input_viz, nrow=nrow)
    ref_grid = create_image_grid(ref_viz, nrow=nrow)
    recon_grid = create_image_grid(recon_viz, nrow=nrow)
    
    # Prepare wandb images
    wandb_images = {
        f'{mode}/input_images': wandb.Image(input_grid, caption=f'{mode.title()} Input Images'),
        f'{mode}/reference_images': wandb.Image(ref_grid, caption=f'{mode.title()} Reference Images'),
        f'{mode}/relit_images': wandb.Image(recon_grid, caption=f'{mode.title()} Relit Images'),
    }
    
    # Add noisy images if provided (training only)
    if mode == 'train' and noisy_input_img is not None and noisy_ref_img is not None:
        noisy_input_viz = noisy_input_img[:batch_size]
        noisy_ref_viz = noisy_ref_img[:batch_size]
        
        noisy_input_grid = create_image_grid(noisy_input_viz, nrow=nrow)
        noisy_ref_grid = create_image_grid(noisy_ref_viz, nrow=nrow)
        
        """wandb_images.update({
            f'{mode}/noisy_input_images': wandb.Image(noisy_input_grid, caption=f'{mode.title()} Noisy Input Images'),
            f'{mode}/noisy_reference_images': wandb.Image(noisy_ref_grid, caption=f'{mode.title()} Noisy Reference Images'),
        })"""
    
    # Create comparison grid (side by side)
    comparison_grid = create_comparison_grid(input_viz, ref_viz, recon_viz, nrow=nrow)
    wandb_images[f'{mode}/comparison'] = wandb.Image(comparison_grid, caption=f'{mode.title()} Comparison: Input | Reference | Relit')
    
    # Log to wandb
    wandb.log(wandb_images, step=step)


def create_comparison_grid(input_img, ref_img, recon_img, nrow=4):
    """Create a comparison grid showing input, reference, and reconstruction side by side"""
    batch_size = input_img.shape[0]
    
    # Interleave images: input, ref, recon, input, ref, recon, ...
    comparison_images = []
    for i in range(batch_size):
        comparison_images.extend([input_img[i], ref_img[i], recon_img[i]])
    
    comparison_tensor = torch.stack(comparison_images)
    
    # Create grid with 3 images per row (input, ref, recon)
    grid = make_grid(comparison_tensor, nrow=3, padding=2, normalize=True)
    grid_np = tensor_to_numpy(grid)
    
    # Convert from CHW to HWC
    if grid_np.shape[0] == 3:
        grid_np = np.transpose(grid_np, (1, 2, 0))
    elif grid_np.shape[0] == 1:
        grid_np = grid_np[0]
    
    return grid_np


def log_relighting_results(input_img, ref_img, relit_img, gt_img, step, mode='train'): #model, 
    """
    Log relighting results for specific target lighting conditions
    
    Args:
        model: The trained model
        input_img: Input image batch
        ref_img: Reference image batch
        target_img: Target lighting reference batch
        step: Current step
        mode: 'train' or 'validation'
    """
    #model.eval()
    
    with torch.no_grad():
        """# Extract features
        intrinsic_input, extrinsic_input = model(input_img, run_encoder=True)
        intrinsic_ref, extrinsic_ref = model(ref_img, run_encoder=True)
        
        # Relight using target lighting
        relit_img = model([intrinsic_input, extrinsic_ref], run_encoder=False).float()
        """
        # Create visualization
        batch_size = min(input_img.shape[0], 4)
        nrow = min(batch_size, 4)
        
        input_viz = input_img[:batch_size]
        ref_viz = ref_img[:batch_size]
        relit_viz = relit_img[:batch_size]
        gt_viz = gt_img[:batch_size]
        
        # Create relighting comparison grid
        relight_comparison = []
        for i in range(batch_size):
            relight_comparison.extend([input_viz[i], ref_viz[i], relit_viz[i], gt_viz[i]])
        
        relight_tensor = torch.stack(relight_comparison)
        relight_grid = make_grid(relight_tensor, nrow=4, padding=2, normalize=True)
        relight_grid_np = tensor_to_numpy(relight_grid)
        
        if relight_grid_np.shape[0] == 3:
            relight_grid_np = np.transpose(relight_grid_np, (1, 2, 0))
        elif relight_grid_np.shape[0] == 1:
            relight_grid_np = relight_grid_np[0]
        
        # Log to wandb
        wandb.log({
            f'{mode}/relighting_results': wandb.Image(
                relight_grid_np, 
                caption=f'{mode.title()} Relighting: Input | Target Light | Relit Result | Ground Truth'
            )
        }, step=step)
    
    #model.train()


def log_feature_visualizations(intrinsic_features, extrinsic_features, step, mode='train'):
    """
    Log feature visualizations (intrinsic and extrinsic components)
    
    Args:
        intrinsic_features: List of intrinsic feature tensors
        extrinsic_features: List of extrinsic feature tensors  
        step: Current step
        mode: 'train' or 'validation'
    """
    wandb_images = {}
    
    # Visualize first few channels of first feature map
    if isinstance(intrinsic_features, list) and len(intrinsic_features) > 0:
        intrinsic_feat = intrinsic_features[0]  # First feature map
        if intrinsic_feat.dim() == 4:  # B, C, H, W
            # Take first 4 channels, first 4 samples
            feat_viz = intrinsic_feat[:4, :4]  # Shape: (4, 4, H, W)
            feat_viz = feat_viz.view(-1, 1, feat_viz.shape[2], feat_viz.shape[3])  # Flatten to grayscale images
            
            intrinsic_grid = create_image_grid(feat_viz, nrow=4, normalize=True)
            wandb_images[f'{mode}/intrinsic_features'] = wandb.Image(
                intrinsic_grid, 
                caption=f'{mode.title()} Intrinsic Features (First 4 channels)'
            )
    
    if isinstance(extrinsic_features, list) and len(extrinsic_features) > 0:
        extrinsic_feat = extrinsic_features[0]  # First feature map
        if extrinsic_feat.dim() == 4:  # B, C, H, W
            # Take first 4 channels, first 4 samples
            feat_viz = extrinsic_feat[:4, :4]  # Shape: (4, 4, H, W)
            feat_viz = feat_viz.view(-1, 1, feat_viz.shape[2], feat_viz.shape[3])  # Flatten to grayscale images
            
            extrinsic_grid = create_image_grid(feat_viz, nrow=4, normalize=True)
            wandb_images[f'{mode}/extrinsic_features'] = wandb.Image(
                extrinsic_grid, 
                caption=f'{mode.title()} Extrinsic Features (First 4 channels)'
            )
    
    if wandb_images:
        wandb.log(wandb_images, step=step)


def save_visualization_locally(images, save_path, title="Visualization"):
    """
    Save visualization locally as well as logging to wandb
    
    Args:
        images: Dict of image grids or single image grid
        save_path: Path to save the visualization
        title: Title for the plot
    """
    if isinstance(images, dict):
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        if n_images == 1:
            axes = [axes]
        
        for idx, (name, img) in enumerate(images.items()):
            axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[idx].set_title(name)
            axes[idx].axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(images, cmap='gray' if len(images.shape) == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Configuration presets for different visualization needs
VISUALIZATION_CONFIGS = {
    'minimal': {
        'viz_log_freq': 100,
        'max_viz_images': 2,
        'max_val_batches': 3,
    },
    'standard': {
        'viz_log_freq': 50,
        'max_viz_images': 4,
        'max_val_batches': 5,
    },
    'detailed': {
        'viz_log_freq': 20,
        'max_viz_images': 8,
        'max_val_batches': 10,
    }
}


def apply_visualization_config(args, config_name='standard'):
    """Apply a visualization configuration preset to args"""
    if config_name in VISUALIZATION_CONFIGS:
        config = VISUALIZATION_CONFIGS[config_name]
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    return args