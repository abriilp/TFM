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
        #print(f"Fetching item {idx} from dataset")
        #print(f"Total images: {len(self.all_images)}")
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
                if scene_name != input_scene and scene_name in self.scenes:  # Different scene
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
                print("Warning: No cross-scene reference images found, using any available image")
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
                print(f"Warning: No reference found for scene={input_scene}, camera_pos={input_camera_pos}, using input as reference")
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


class RLSIDDataset(Dataset):
    def __init__(self, root_dir, img_transform=None, mode='train', annotation_file=None, 
                 use_reflectance=False, use_shading=False, cross_scene_validation=False):
        """
        ReLighting Surreal Intrinsic Dataset (RLSID) for relighting tasks
        
        Args:
            root_dir: Path to the dataset root directory containing Image, Reflectance, Shading folders
            img_transform: List of image transformations [group_transform, single_transform] or single transform
            mode: 'train', 'val', 'test_quantitative', 'test_qualitative', or 'custom'
            annotation_file: Path to annotation file (train.txt, val_pairs.txt, etc.)
                           If None, will try to auto-detect based on mode
            use_reflectance: If True, also loads reflectance components
            use_shading: If True, also loads shading components
            cross_scene_validation: If True, ensures reference images come from different scenes (validation mode)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.use_reflectance = use_reflectance
        self.use_shading = use_shading
        self.cross_scene_validation = cross_scene_validation
        
        # Handle transform assignment
        if img_transform is not None and isinstance(img_transform, list):
            self.group_img_transform = img_transform[0]
            self.single_img_transform = img_transform[1]
        else:
            self.group_img_transform = None
            self.single_img_transform = img_transform
        
        # Set up folder paths
        self.image_dir = os.path.join(root_dir, 'Image')
        self.reflectance_dir = os.path.join(root_dir, 'Reflectance') if use_reflectance else None
        self.shading_dir = os.path.join(root_dir, 'Shading') if use_shading else None
        self.annotation_dir = os.path.join(root_dir, 'annotation')
        
        # Verify directories exist
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        if use_reflectance and not os.path.exists(self.reflectance_dir):
            raise ValueError(f"Reflectance directory not found: {self.reflectance_dir}")
        
        if use_shading and not os.path.exists(self.shading_dir):
            raise ValueError(f"Shading directory not found: {self.shading_dir}")
        
        # Load data based on mode
        if mode in ['train', 'val', 'test_quantitative', 'test_qualitative']:
            self.data_pairs = self._load_annotation_file(annotation_file)
        elif mode == 'custom':
            # Custom mode: load all images and create pairs dynamically
            self.data_pairs = self._create_custom_pairs()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if not self.data_pairs:
            raise ValueError(f"No valid data pairs found for mode: {mode}")
        
        print(f"Loaded {len(self.data_pairs)} pairs for mode: {mode}")
        
    def _load_annotation_file(self, annotation_file):
        """Load data pairs from annotation file"""
        if annotation_file is None:
            # Auto-detect annotation file based on mode
            filename_map = {
                'train': 'train.txt',
                'val': 'val_pairs.txt',
                'test_quantitative': 'test_quantitative_pairs.txt',
                'test_qualitative': 'test_qualitative_pairs.txt'
            }
            if self.mode not in filename_map:
                raise ValueError(f"No default annotation file for mode: {self.mode}")
            annotation_file = os.path.join(self.annotation_dir, filename_map[self.mode])
        
        if not os.path.exists(annotation_file):
            raise ValueError(f"Annotation file not found: {annotation_file}")
        
        pairs = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse different annotation formats
                if self.mode == 'train':
                    # Training set might just list image names
                    pairs.append({'input': line.strip()})
                else:
                    # Validation/test sets should have pairs
                    parts = line.split()
                    if len(parts) >= 2:
                        pairs.append({
                            'input': parts[0].strip(),
                            'reference': parts[1].strip()
                        })
                    else:
                        # Fallback: treat as single image
                        pairs.append({'input': parts[0].strip()})
        
        return pairs
    
    def _create_custom_pairs(self):
        """Create custom pairs by scanning all images"""
        image_files = glob.glob(os.path.join(self.image_dir, "*.png"))
        
        # Parse all images and organize by scene
        scene_data = {}  # scene_id -> {background_id -> [image_info]}
        
        for img_path in image_files:
            try:
                filename = os.path.basename(img_path)
                # Parse filename: aaa_bbb_xxx_yyy_zzz.png
                parts = filename.split('_')
                if len(parts) >= 5:
                    scene_id = parts[0]
                    background_id = parts[1]
                    pan = parts[2]
                    tilt = parts[3]
                    color = parts[4].split('.')[0]  # Remove .png extension
                    
                    scene_key = f"{scene_id}_{background_id}"
                    
                    if scene_key not in scene_data:
                        scene_data[scene_key] = []
                    
                    scene_data[scene_key].append({
                        'filename': filename,
                        'path': img_path,
                        'scene_id': scene_id,
                        'background_id': background_id,
                        'pan': pan,
                        'tilt': tilt,
                        'color': color,
                        'scene_key': scene_key
                    })
            except (IndexError, ValueError) as e:
                print(f"Skipping invalid filename: {filename} - {e}")
                continue
        
        # Create pairs
        pairs = []
        for scene_key, images in scene_data.items():
            for img_info in images:
                pairs.append({'input': img_info['filename'], 'scene_data': scene_data})
        
        return pairs
    
    def _parse_filename(self, filename):
        """Parse RLSID filename format: aaa_bbb_xxx_yyy_zzz.png"""
        parts = filename.split('_')
        if len(parts) < 5:
            raise ValueError(f"Invalid filename format: {filename}")
        
        return {
            'scene_id': parts[0],
            'background_id': parts[1],
            'pan': parts[2],
            'tilt': parts[3],
            'color': parts[4].split('.')[0],
            'scene_key': f"{parts[0]}_{parts[1]}"
        }
    
    def _get_image_path(self, filename, image_type='Image'):
        """Get full path for an image file"""
        if image_type == 'Image':
            return os.path.join(self.image_dir, filename)
        elif image_type == 'Reflectance':
            return os.path.join(self.reflectance_dir, filename)
        elif image_type == 'Shading':
            return os.path.join(self.shading_dir, filename)
        else:
            raise ValueError(f"Unknown image type: {image_type}")
    
    def _find_reference_image(self, input_filename, input_info):
        """Find appropriate reference image based on mode and settings"""
        if self.mode == 'train' or (self.mode == 'custom' and not self.cross_scene_validation):
            # Training mode: same scene, different lighting
            if 'scene_data' in self.data_pairs[0]:
                scene_data = self.data_pairs[0]['scene_data']
                scene_key = input_info['scene_key']
                
                if scene_key in scene_data:
                    # Find images with different lighting conditions
                    candidates = [img for img in scene_data[scene_key] 
                                if img['color'] != input_info['color']]
                    
                    if candidates:
                        return random.choice(candidates)['filename']
            
            # Fallback: use input as reference
            return input_filename
        
        elif self.cross_scene_validation:
            # Cross-scene validation: different scene
            if 'scene_data' in self.data_pairs[0]:
                scene_data = self.data_pairs[0]['scene_data']
                input_scene_key = input_info['scene_key']
                
                # Find images from different scenes
                candidates = []
                for scene_key, images in scene_data.items():
                    if scene_key != input_scene_key:
                        candidates.extend([img['filename'] for img in images])
                
                if candidates:
                    return random.choice(candidates)
            
            # Fallback: use input as reference
            return input_filename
        
        else:
            # Default: use input as reference
            return input_filename
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        input_filename = pair['input']
        
        # Parse input filename
        input_info = self._parse_filename(input_filename)
        
        # Get reference filename
        if 'reference' in pair:
            ref_filename = pair['reference']
        else:
            ref_filename = self._find_reference_image(input_filename, input_info)
        
        # Load input image
        input_path = self._get_image_path(input_filename, 'Image')
        input_img = Image.open(input_path).convert('RGB')
        
        # Load reference image
        ref_path = self._get_image_path(ref_filename, 'Image')
        ref_img = Image.open(ref_path).convert('RGB')
        
        # Prepare return data
        return_data = [input_img, ref_img]
        
        # Load reflectance components if requested
        if self.use_reflectance:
            input_refl_path = self._get_image_path(input_filename, 'Reflectance')
            ref_refl_path = self._get_image_path(ref_filename, 'Reflectance')
            
            input_refl = Image.open(input_refl_path).convert('RGB')
            ref_refl = Image.open(ref_refl_path).convert('RGB')
            
            return_data.extend([input_refl, ref_refl])
        
        # Load shading components if requested
        if self.use_shading:
            input_shad_path = self._get_image_path(input_filename, 'Shading')
            ref_shad_path = self._get_image_path(ref_filename, 'Shading')
            
            input_shad = Image.open(input_shad_path).convert('RGB')
            ref_shad = Image.open(ref_shad_path).convert('RGB')
            
            return_data.extend([input_shad, ref_shad])
        
        # Apply single image transforms
        if self.single_img_transform is not None:
            return_data = [self.single_img_transform(img) for img in return_data]
        
        # Apply group transforms
        if self.group_img_transform is not None:
            return_data = self.group_img_transform(return_data)
        
        return tuple(return_data)


# Example usage:
"""
# Basic usage with training data
train_dataset = RLSIDDataset(
    root_dir='./data',
    mode='train',
    img_transform=[group_transform, single_transform]
)

# Validation with cross-scene references
val_dataset = RLSIDDataset(
    root_dir='./data',
    mode='val',
    cross_scene_validation=True,
    img_transform=[group_transform, single_transform]
)

# Custom mode with reflectance and shading
custom_dataset = RLSIDDataset(
    root_dir='./data',
    mode='custom',
    use_reflectance=True,
    use_shading=True,
    img_transform=[group_transform, single_transform]
)

# Test with specific annotation file
test_dataset = RLSIDDataset(
    root_dir='./data',
    mode='test_quantitative',
    annotation_file='./data/annotation/test_quantitative_pairs.txt'
)
"""