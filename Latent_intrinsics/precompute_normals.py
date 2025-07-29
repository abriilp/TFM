import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils import DepthNormalConsistencyLoss  # Replace with actual import

# Initialize model
model = DepthNormalConsistencyLoss(enable_normals=True).to('cuda')
model.eval()

# Paths
source_dir = '/data/storage/datasets/RLSID/Image'
target_dir = '/home/apinyol/TFM/Data/RLSID/Image/normals'
avg_target_dir = os.path.join(target_dir, 'averaged')

# Allowed image extensions
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Transform to model input ([-1, 1], CHW)
to_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def save_normal_image(tensor, path):
    """Convert normal map tensor to RGB image and save"""
    normal_rgb = (tensor + 1.0) / 2.0  # [-1, 1] → [0, 1]
    normal_rgb = torch.clamp(normal_rgb, 0, 1)
    normal_img = (normal_rgb * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(normal_img).save(path)

# Accumulate normals per scene
scene_normals = defaultdict(list)

for root, _, files in os.walk(source_dir):
    for fname in tqdm(files, desc=f"Processing {root}"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_extensions:
            continue

        src_path = os.path.join(root, fname)
        rel_path = os.path.relpath(src_path, source_dir)
        dst_path = os.path.join(target_dir, os.path.splitext(rel_path)[0] + '.png')

        # Extract scene name (assumes structure source_dir/scene_name/image.jpg)
        scene_name = os.path.normpath(rel_path).split(os.sep)[0].split('_')[:2]
        scene_name = scene_name[0]+'_'+scene_name[1] # Join first two parts for scene name
        print(f"SCENE NAME: {scene_name}")

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            image = Image.open(src_path).convert('RGB')
            img_tensor = to_tensor(image).unsqueeze(0).to(model.device)

            with torch.no_grad():
                normal = model.get_normal_map(img_tensor)[0]  # shape: (C, H, W)

            # Save individual normal
            save_normal_image(normal, dst_path)

            # Accumulate for averaging
            scene_normals[scene_name].append(normal.cpu())

        except Exception as e:
            print(f"Failed to process {src_path}: {e}")

# Compute and save averaged normals
for scene, normals in scene_normals.items():
    stacked = torch.stack(normals)  # shape: (N, C, H, W)
    avg = stacked.mean(dim=0)
    avg = torch.nn.functional.normalize(avg, dim=0)  # Normalize each pixel vector

    avg_path = os.path.join(avg_target_dir, f"{scene}.png")
    os.makedirs(os.path.dirname(avg_path), exist_ok=True)
    save_normal_image(avg, avg_path)
    print(f"Saved averaged normal for scene '{scene}' to {avg_path}")

