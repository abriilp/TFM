import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import re
from collections import defaultdict

class RSRDataset(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier=10, 
                 total_split=4, split_id=0, light_colors=[1]):
        """
        Dataset for lighting scenes with opposing light position pairs.
        
        Args:
            root: Path to dataset root containing scene folders
            img_transform: [group_transform, single_transform] for data augmentation
            epoch_multiplier: Multiplier for epoch length
            total_split: Total number of splits for distributed training
            split_id: Current split ID for distributed training
        """
        self.epoch_multiplier = epoch_multiplier
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        self.light_colors = light_colors
        
        # Define opposing light position pairs basedf on your 3x3 grid
        # Grid layout:
        # 5 4 3
        # 6 1 2  
        # 7 8 9
        self.opposing_pairs = {
            1: [5, 7, 9, 3, 4, 6, 2, 8],  # center -> corners
            2: [6, 5, 7],     # right -> left and bottom
            3: [7, 6, 8],     # top-right -> bottom corners
            4: [8, 7, 9],        # top -> bottom
            5: [9, 8, 2],     # top-left -> center and bottom-right
            6: [2, 3, 9],     # left -> right positions
            7: [2, 3, 4],     # bottom-left -> center and top-right
            8: [2, 3, 5],     # bottom -> top positions
            9: [4, 5, 6]   # bottom-right -> center and corners
        }
        
        # Load and organize images by scene
        self.scene_images = self._load_scenes(root, total_split, split_id)
        print(f'Initialized with {len(self.scene_images)} scenes')
        
    def _parse_filename(self, filename):
        """
        Parse filename to extract lighting parameters.
        Format: {index}_{group}_{pan}_{tilt}_{R}_{G}_{B}_{scene}_{not_use}_{light_color}_{light_pos}
        """
        basename = os.path.basename(filename).split('.')[0]  # Remove extension
        parts = basename.split('_')
        
        if len(parts) < 11:
            return None
            
        try:
            return {
                'index': int(parts[0]),
                'group': int(parts[1]),
                'pan': float(parts[2]),
                'tilt': float(parts[3]),
                'R': int(parts[4]),
                'G': int(parts[5]),
                'B': int(parts[6]),
                'scene': int(parts[7]),
                'not_use': int(parts[8]),
                'light_color': int(parts[9]),
                'light_pos': int(parts[10])
            }
        except (ValueError, IndexError):
            return None
    
    def _load_scenes(self, root, total_split, split_id):
        """Load and organize images by scene, filtering for light color 1"""
        scene_folders = sorted(glob.glob(os.path.join(root, '*')))
        scene_folders = compute_rank_split(scene_folders, total_split, split_id)
        
        scene_images = {}
        
        for scene_folder in scene_folders:
            if not os.path.isdir(scene_folder):
                continue
                
            scene_name = os.path.basename(scene_folder)
            image_files = glob.glob(os.path.join(scene_folder, '*'))
            
            # Group images by light position for this scene
            light_groups = defaultdict(list)
            
            for img_file in image_files:
                params = self._parse_filename(img_file)
                if params is None:
                    continue
                    
                # Only use light color 1
                if params['light_color'] not in self.light_colors:
                    continue
                    
                light_pos = params['light_pos']
                if light_pos in range(1, 10):  # Valid light positions 1-9
                    light_groups[light_pos].append(img_file)
            
            # Only keep scenes that have images for multiple light positions
            valid_positions = [pos for pos, files in light_groups.items() if len(files) > 0]
            if len(valid_positions) >= 2:
                # Load images for valid positions
                scene_data = {}
                for pos in valid_positions:
                    if light_groups[pos]:  # Take first image for each position
                        img_path = light_groups[pos][0]
                        try:
                            img = Image.open(img_path).convert('RGB')
                            scene_data[pos] = np.array(img)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                            continue
                
                if len(scene_data) >= 2:
                    scene_images[scene_name] = scene_data
                    
        return scene_images
    
    def _get_opposing_position(self, light_pos):
        """Get a random opposing light position"""
        if light_pos in self.opposing_pairs:
            available_opposites = []
            for opp_pos in self.opposing_pairs[light_pos]:
                # Check if we have an image for this opposing position in current scene
                if opp_pos in self.current_scene_data:
                    available_opposites.append(opp_pos)
            
            if available_opposites:
                return np.random.choice(available_opposites)
        
        # Fallback: return any available position that's not the input position
        available_positions = [pos for pos in self.current_scene_data.keys() if pos != light_pos]
        if available_positions:
            return np.random.choice(available_positions)
        
        return None

    def __len__(self):
        return len(self.scene_images) * self.epoch_multiplier * 25

    def __getitem__(self, index):
        index = index % (len(self.scene_images) * 25)
        scene_idx = index // 25
        
        scene_names = list(self.scene_images.keys())
        scene_name = scene_names[scene_idx]
        self.current_scene_data = self.scene_images[scene_name]
        
        # Get available light positions for this scene
        available_positions = list(self.current_scene_data.keys())
        
        if len(available_positions) < 2:
            # Fallback: use the same image twice (shouldn't happen with proper filtering)
            pos1 = available_positions[0]
            pos2 = pos1
        else:
            # Select random input position
            pos1 = np.random.choice(available_positions)
            
            # Get opposing position
            pos2 = self._get_opposing_position(pos1)
            if pos2 is None:
                # Fallback: select random different position
                pos2 = np.random.choice([p for p in available_positions if p != pos1])
        
        # Load images
        img1 = Image.fromarray(self.current_scene_data[pos1])
        img2 = Image.fromarray(self.current_scene_data[pos2])
        
        # Apply transforms
        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)
        
        if self.group_img_transform is not None:
            img1, img2 = self.group_img_transform([img1, img2])
            
        return img1, img2
        
def compute_rank_split(data_list, total_split, split_id):
    """Split data list for distributed training"""
    per_rank = len(data_list) // total_split
    start_idx = split_id * per_rank
    if split_id == total_split - 1:  # Last rank gets remaining data
        end_idx = len(data_list)
    else:
        end_idx = start_idx + per_rank
    return data_list[start_idx:end_idx]