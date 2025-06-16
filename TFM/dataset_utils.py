import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import glob

class RSRDataset(Dataset):
    def __init__(self, root_dir, img_transform=None, scenes=None, is_validation=False, validation_scenes=None):
        """
        RSR Dataset for relighting tasks
        
        Args:
            root_dir: Path to the dataset root directory containing scene folders
            img_transform: List of image transformations [group_transform, single_transform]
            scenes: List of scene names to use. If None, auto-splits all scenes (all-2 for train, 2 for val)
            is_validation: If True, enables validation mode with cross-scene references
            validation_scenes: Specific scenes to use for validation (only used when scenes=None)
        """
        self.root_dir = root_dir
        self.is_validation = is_validation
        
        # Handle transform assignment
        if img_transform is not None and isinstance(img_transform, list):
            self.group_img_transform = img_transform[0]
            self.single_img_transform = img_transform[1]
        else:
            self.group_img_transform = None
            self.single_img_transform = img_transform
        
        # Get all scene directories
        all_scenes = [d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]
        all_scenes.sort()  # Sort for reproducible splits
        
        if scenes is not None:
            # Use explicitly provided scenes
            self.scenes = [s for s in scenes if s in all_scenes]
        else:
            # Auto-split: use all scenes except 2 for training, 2 for validation
            if len(all_scenes) < 2:
                raise ValueError(f"Need at least 2 scenes for auto-split, found {len(all_scenes)}")
            
            # Determine validation scenes
            if validation_scenes is not None:
                val_scenes = [s for s in validation_scenes if s in all_scenes]
                if len(val_scenes) < 2:
                    # Fallback to last 2 scenes if provided validation_scenes are invalid
                    val_scenes = all_scenes[-2:]
                    print(f"Warning: Using last 2 scenes for validation: {val_scenes}")
            else:
                # Use last 2 scenes for validation by default
                val_scenes = all_scenes[-2:]
            
            if self.is_validation:
                self.scenes = val_scenes
                # Store all scenes for cross-scene reference selection
                self.all_available_scenes = all_scenes
            else:
                # Training: use all scenes except validation scenes
                self.scenes = [s for s in all_scenes if s not in val_scenes]
                
        if not self.scenes:
            raise ValueError(f"No valid scenes found in {root_dir}")
        
        print(f"Using scenes: {self.scenes}")
        if self.is_validation and hasattr(self, 'all_available_scenes'):
            print(f"All available scenes for cross-reference: {self.all_available_scenes}")
        
        # Parse all images and organize by scene, camera_pos, light_pos
        self.image_data = {}  # scene -> camera_pos -> light_pos -> image_path
        self.all_images = []  # List of all valid image paths for iteration
        
        # If validation mode with cross-scene references, we need data from all scenes
        scenes_to_parse = self.all_available_scenes if (self.is_validation and hasattr(self, 'all_available_scenes')) else self.scenes
        
        for scene in scenes_to_parse:
            scene_dir = os.path.join(root_dir, scene)
            if not os.path.exists(scene_dir):
                continue
                
            image_files = glob.glob(os.path.join(scene_dir, "*.jpg")) + \
                         glob.glob(os.path.join(scene_dir, "*.png"))
            
            scene_data = {}
            for img_path in image_files:
                try:
                    filename = os.path.basename(img_path)
                    # Parse filename: {index}_{group}_{pan}_{tilt}_{R}_{G}_{B}_{scene}_{camera_pos}_{light_color}_{light_pos}
                    parts = filename.split('_')
                    if len(parts) >= 11:
                        light_color = parts[9]
                        # Only use light color 1
                        if light_color == '1':
                            camera_pos = parts[8]
                            light_pos = parts[10].split('.')[0]  # Remove file extension
                            
                            if camera_pos not in scene_data:
                                scene_data[camera_pos] = {}
                            if light_pos not in scene_data[camera_pos]:
                                scene_data[camera_pos][light_pos] = []
                            
                            scene_data[camera_pos][light_pos].append(img_path)
                            
                            # Only add to all_images if it's from our target scenes
                            if scene in self.scenes:
                                self.all_images.append({
                                    'path': img_path,
                                    'scene': scene,
                                    'camera_pos': camera_pos,
                                    'light_pos': light_pos
                                })
                except (IndexError, ValueError) as e:
                    print(f"Skipping invalid filename: {filename} - {e}")
                    continue
            
            if scene_data:
                self.image_data[scene] = scene_data
        
        if not self.all_images:
            raise ValueError("No valid images found with light_color=1")
        
        print(f"Total valid images: {len(self.all_images)}")
        print(f"Scenes with data: {list(self.image_data.keys())}")
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        input_info = self.all_images[idx]
        input_path = input_info['path']
        input_scene = input_info['scene']
        input_camera_pos = input_info['camera_pos']
        input_light_pos = input_info['light_pos']
        
        # Load input image
        input_img = Image.open(input_path).convert('RGB')
        
        if self.is_validation:
            # VALIDATION MODE: Reference from DIFFERENT scenes for cross-scene evaluation
            # Get all images from scenes different from input scene
            cross_scene_images = []
            for scene_name, scene_data in self.image_data.items():
                if scene_name != input_scene:  # Different scene
                    for cam_pos, cam_data in scene_data.items():
                        for light_pos, img_paths in cam_data.items():
                            for img_path in img_paths:
                                cross_scene_images.append({
                                    'path': img_path,
                                    'scene': scene_name,
                                    'camera_pos': cam_pos,
                                    'light_pos': light_pos
                                })
            
            if cross_scene_images:
                ref_info = random.choice(cross_scene_images)
                ref_path = ref_info['path']
                ref_light_pos = ref_info['light_pos']
            else:
                # Fallback: use any available reference if no cross-scene images found
                all_available_images = []
                for scene_data in self.image_data.values():
                    for cam_data in scene_data.values():
                        for img_paths in cam_data.values():
                            all_available_images.extend([{'path': p, 'light_pos': p.split('_')[10].split('.')[0]} for p in img_paths])
                
                if all_available_images:
                    ref_info = random.choice(all_available_images)
                    ref_path = ref_info['path']
                    ref_light_pos = ref_info['light_pos']
                else:
                    ref_path = input_path
                    ref_light_pos = input_light_pos
                    print("Warning: No reference images found, using input as reference")
            
            # GT: Same scene and camera_pos as input, same light_pos as reference
            gt_candidates = []
            if (input_scene in self.image_data and 
                input_camera_pos in self.image_data[input_scene] and 
                ref_light_pos in self.image_data[input_scene][input_camera_pos]):
                gt_candidates = self.image_data[input_scene][input_camera_pos][ref_light_pos]
            
            if gt_candidates:
                gt_path = random.choice(gt_candidates)
            else:
                # Fallback: use input as GT if no matching GT found
                gt_path = input_path
                print(f"Warning: No GT found for scene={input_scene}, camera_pos={input_camera_pos}, light_pos={ref_light_pos}")
            
            # Load reference and GT images
            ref_img = Image.open(ref_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            
            # Apply transforms
            if self.single_img_transform is not None:
                input_img = self.single_img_transform(input_img)
                ref_img = self.single_img_transform(ref_img)
                gt_img = self.single_img_transform(gt_img)

            if self.group_img_transform is not None:
                input_img, ref_img, gt_img = self.group_img_transform([input_img, ref_img, gt_img])
            
            return input_img, ref_img, gt_img
            
        else:
            # TRAINING MODE: Reference from same scene and camera_pos, different light_pos
            ref_candidates = []
            
            if (input_scene in self.image_data and 
                input_camera_pos in self.image_data[input_scene]):
                
                scene_camera_data = self.image_data[input_scene][input_camera_pos]
                # Get all light positions except the current one
                available_light_pos = [lp for lp in scene_camera_data.keys() if lp != input_light_pos]
                
                if available_light_pos:
                    ref_light_pos = random.choice(available_light_pos)
                    ref_candidates = scene_camera_data[ref_light_pos]
            
            if ref_candidates:
                ref_path = random.choice(ref_candidates)
            else:
                # Fallback: use a random image from the same scene
                scene_images = [img for img in self.all_images if img['scene'] == input_scene]
                if scene_images:
                    ref_info = random.choice(scene_images)
                    ref_path = ref_info['path']
                else:
                    # Last resort: use input image
                    ref_path = input_path
                    print(f"Warning: Using input as reference for training - scene={input_scene}, camera_pos={input_camera_pos}")
            
            # Load reference image
            ref_img = Image.open(ref_path).convert('RGB')
            
            # Apply transforms
            if self.single_img_transform is not None:
                input_img = self.single_img_transform(input_img)
                ref_img = self.single_img_transform(ref_img)

            if self.group_img_transform is not None:
                input_img, ref_img = self.group_img_transform([input_img, ref_img])
            
            return input_img, ref_img