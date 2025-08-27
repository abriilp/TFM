import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import glob

class RSRDataset(Dataset):
    def __init__(self, root_dir, img_transform=None, scenes=None, is_validation=False, validation_scenes=None, use_all_scenes=False):
        """
        RSR Dataset for relighting tasks
        
        Args:
            root_dir: Path to the dataset root directory containing scene folders
            img_transform: List of image transformations [group_transform, single_transform]
            scenes: List of scene names to use. If None, auto-splits all scenes (all-2 for train, 2 for val)
            is_validation: If True, enables validation mode with cross-scene references
            validation_scenes: Specific scenes to use for validation (only used when scenes=None)
            use_all_scenes: If True, uses all available scenes regardless of other parameters
        """
        self.root_dir = root_dir
        self.is_validation = is_validation
        self.use_all_scenes = use_all_scenes
        
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
        
        if self.use_all_scenes:
            # Use all available scenes
            self.scenes = all_scenes
            print(f"Using all scenes: {self.scenes}")
        elif scenes is not None:
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
            
        # Store all available scenes for validation mode if not already set
        if self.is_validation and not hasattr(self, 'all_available_scenes'):
            self.all_available_scenes = all_scenes
                
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


import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset

class RLSIDDataset(Dataset):
    def __init__(self, root_dir, img_transform=None, validation_scenes=None, 
                 is_validation=False, image_folder='Image', max_training_images=None,
                 enable_normals=False):
        """
        RLSID Dataset for relighting tasks
        
        Args:
            root_dir: Path to the dataset root directory (should contain Image/, Reflectance/, Shading/ folders)
            img_transform: List of image transformations [group_transform, single_transform]
            validation_scenes: List of scene indices to use for validation (e.g., ['004', '005']). 
                             Training will use all other available scenes.
            is_validation: If True, enables validation mode (returns 3 images: input, ref, gt)
            image_folder: Which folder to use ('Image', 'Reflectance', or 'Shading')
            max_training_images: Maximum number of images to use for training. If None, uses all.
                               Only applies to training mode, validation uses all available images.
            enable_normals: If True, loads and returns normal maps. If False, skips normals entirely.
        """
        self.root_dir = root_dir
        self.is_validation = is_validation
        self.image_folder = image_folder
        self.max_training_images = max_training_images
        self.enable_normals = enable_normals
        
        # Handle transform assignment
        if img_transform is not None and isinstance(img_transform, list):
            self.group_img_transform = img_transform[0]
            self.single_img_transform = img_transform[1]
        else:
            self.group_img_transform = None
            self.single_img_transform = img_transform
        
        # Path to the image directory
        self.image_dir = os.path.join(root_dir, image_folder)
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        # Path to the normals directory (only check if normals are enabled)
        if self.enable_normals:
            self.normals_dir = '/home/apinyol/TFM/Data/RLSID_normals/Image/normals'
            if not os.path.exists(self.normals_dir):
                print(f"Warning: Normals directory not found: {self.normals_dir}. Disabling normals.")
                self.enable_normals = False
        
        # Get all image files
        image_files = glob.glob(os.path.join(self.image_dir, "*.png"))
        
        # Parse all images and organize by scene_background -> lighting_conditions
        self.image_data = {}  # scene_background -> lighting -> [image_paths]
        all_scene_backgrounds = set()
        
        for img_path in image_files:
            try:
                filename = os.path.basename(img_path)
                # Parse filename: aaa_bbb_xxx_yyy_zzz.png
                name_parts = filename.replace('.png', '').split('_')
                if len(name_parts) >= 5:
                    scene_idx = name_parts[0]  # aaa
                    background = name_parts[1]  # bbb
                    pan = name_parts[2]         # xxx
                    tilt = name_parts[3]        # yyy
                    color = name_parts[4]       # zzz
                    
                    scene_background = f"{scene_idx}_{background}"
                    lighting_condition = f"{pan}_{tilt}_{color}"
                    
                    all_scene_backgrounds.add(scene_background)
                    
                    if scene_background not in self.image_data:
                        self.image_data[scene_background] = {}
                    if lighting_condition not in self.image_data[scene_background]:
                        self.image_data[scene_background][lighting_condition] = []
                    
                    self.image_data[scene_background][lighting_condition].append(img_path)
                    
            except (IndexError, ValueError) as e:
                print(f"Skipping invalid filename: {filename} - {e}")
                continue
        
        # Determine which scenes to use
        all_scene_backgrounds = sorted(list(all_scene_backgrounds))
        
        if validation_scenes is None:
            # Auto-split: use 80% for training, 20% for validation
            num_scenes = len(all_scene_backgrounds)
            if num_scenes < 2:
                raise ValueError(f"Need at least 2 scenes for auto-split, found {num_scenes}")
            
            split_idx = max(1, int(num_scenes * 0.8))
            self.train_scenes = all_scene_backgrounds[:split_idx]
            self.validation_scenes = all_scene_backgrounds[split_idx:]
        else:
            # Use provided validation scenes, training uses all others
            self.validation_scenes = [sb for sb in all_scene_backgrounds 
                                    if any(sb.startswith(f"{vs}_") for vs in validation_scenes)]
            self.train_scenes = [sb for sb in all_scene_backgrounds 
                               if sb not in self.validation_scenes]
        
        # Select scenes based on mode
        if self.is_validation:
            self.active_scenes = self.validation_scenes
            # Store all scenes for cross-scene reference in validation
            self.all_scenes = list(self.image_data.keys())
        else:
            self.active_scenes = self.train_scenes
        
        if not self.active_scenes:
            mode = "validation" if self.is_validation else "training"
            raise ValueError(f"No scenes found for {mode} mode")
        
        # Build list of all valid images for the current mode
        self.all_images = []
        for scene_bg in self.active_scenes:
            if scene_bg in self.image_data:
                for lighting, img_paths in self.image_data[scene_bg].items():
                    for img_path in img_paths:
                        filename = os.path.basename(img_path)
                        
                        # Only add normals path if normals are enabled
                        if self.enable_normals:
                            normals_path = os.path.join(self.normals_dir, filename)
                        else:
                            normals_path = None
                            
                        self.all_images.append({
                            'path': img_path,
                            'scene_background': scene_bg,
                            'lighting': lighting,
                            'normals_path': normals_path
                        })
        
        if not self.all_images:
            raise ValueError("No valid images found")
        
        # Apply training image limit BEFORE computing validation pairs
        original_count = len(self.all_images)
        if not self.is_validation and self.max_training_images is not None:
            if self.max_training_images < len(self.all_images):
                # Use a fixed seed for reproducibility across workers
                import random
                random.seed(42)  # Fixed seed for reproducible sampling
                random.shuffle(self.all_images)
                self.all_images = self.all_images[:self.max_training_images]
                print(f"Limited training images from {original_count} to {len(self.all_images)}")
        
        # Pre-compute valid scene pairs for validation AFTER all_images is finalized
        if self.is_validation:
            self._compute_validation_pairs()
        
        # Final check after all modifications
        if not self.all_images:
            raise ValueError("No valid images found after all filtering")
        
        print(f"Mode: {'Validation' if self.is_validation else 'Training'}")
        print(f"Normals enabled: {self.enable_normals}")
        print(f"Active scenes: {self.active_scenes}")
        print(f"Final dataset size: {len(self.all_images)}")
        if not self.is_validation and self.max_training_images is not None:
            print(f"Training image limit: {self.max_training_images} (from {original_count} available)")
        if self.is_validation:
            print(f"All available scenes for cross-reference: {self.all_scenes}")
            print(f"Valid scene pairs found: {len(self.valid_pairs)}")
    
    def _get_deterministic_random(self, idx, salt=0):
        """
        Create a deterministic random generator based on index and salt.
        This ensures consistent behavior across multiple DataLoader workers.
        """
        # Use a combination of idx and salt as seed for reproducible randomness
        seed = hash((idx, salt)) % (2**32)
        rng = random.Random(seed)
        return rng
    
    def _get_normals_path(self, img_path):
        """Get the corresponding normals path for an image path"""
        if not self.enable_normals:
            return None
        filename = os.path.basename(img_path)
        return os.path.join(self.normals_dir, filename)
    
    def _load_normals(self, normals_path):
        """Load normal image from path"""
        if not self.enable_normals or normals_path is None:
            return None
            
        try:
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.PILToTensor()
            ])

            normals_img = Image.open(normals_path).convert('RGB')
            return normals_img
        except Exception as e:
            print(f"Error loading normals from {normals_path}: {e}")
            # Return a black image as fallback
            return Image.new('RGB', (256, 256), (0, 0, 0))
    
    def _compute_validation_pairs(self):
        """
        Pre-compute pairs of validation scenes that share at least one lighting condition.
        This ensures we can always find valid input-reference-GT triplets.
        """
        self.valid_pairs = {}  # input_scene -> {ref_scene: [shared_lightings]}
        self.scene_to_valid_refs = {}  # input_scene -> [list of valid reference scenes]
        
        for input_scene in self.validation_scenes:
            if input_scene not in self.image_data:
                continue
                
            input_lightings = set(self.image_data[input_scene].keys())
            self.valid_pairs[input_scene] = {}
            self.scene_to_valid_refs[input_scene] = []
            
            for ref_scene in self.validation_scenes:
                if ref_scene == input_scene or ref_scene not in self.image_data:
                    continue
                
                ref_lightings = set(self.image_data[ref_scene].keys())
                shared_lightings = input_lightings.intersection(ref_lightings)
                
                if shared_lightings:
                    self.valid_pairs[input_scene][ref_scene] = list(shared_lightings)
                    self.scene_to_valid_refs[input_scene].append(ref_scene)
        
        # Remove scenes that have no valid reference scenes
        scenes_with_refs = [scene for scene in self.scene_to_valid_refs 
                           if len(self.scene_to_valid_refs[scene]) > 0]
        
        # Filter all_images to only include scenes that have valid references
        if self.is_validation:
            original_size = len(self.all_images)
            self.all_images = [img for img in self.all_images 
                             if img['scene_background'] in scenes_with_refs]
            print(f"Filtered validation images from {original_size} to {len(self.all_images)} (keeping only scenes with valid references)")
        
        print(f"Scenes with valid reference pairs: {scenes_with_refs}")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        # Add safety check
        if idx >= len(self.all_images):
            raise IndexError(f"Index {idx} out of range. Dataset size: {len(self.all_images)}")
        
        input_info = self.all_images[idx]
        input_path = input_info['path']
        input_scene_bg = input_info['scene_background']
        input_lighting = input_info['lighting']
        input_normals_path = input_info['normals_path']
        
        # Load input image
        input_img = Image.open(input_path).convert('RGB')
        
        if self.is_validation:
            # VALIDATION MODE: 3 images + optionally 2 normals
            # Use deterministic random generation based on index
            rng = self._get_deterministic_random(idx)
            
            # Get valid reference scenes for this input scene
            valid_ref_scenes = self.scene_to_valid_refs.get(input_scene_bg, [])
            
            if not valid_ref_scenes:
                raise ValueError(f"No valid reference scenes found for {input_scene_bg}")
            
            # Choose a deterministic reference scene based on index
            ref_scene = rng.choice(valid_ref_scenes)
            
            # Get all lightings available in the reference scene that also exist in input scene
            available_ref_lightings = self.valid_pairs[input_scene_bg][ref_scene]
            
            # Choose a deterministic lighting condition
            ref_lighting = rng.choice(available_ref_lightings)
            
            # Get reference image with chosen lighting
            ref_candidates = self.image_data[ref_scene][ref_lighting]
            ref_path = rng.choice(ref_candidates)
            
            # GT: same scene as input, same lighting as reference
            gt_candidates = self.image_data[input_scene_bg][ref_lighting]
            gt_path = rng.choice(gt_candidates)
            
            # Load reference and GT images
            ref_img = Image.open(ref_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            
            # Load normals only if enabled
            if self.enable_normals:
                input_normals = self._load_normals(input_normals_path)
                ref_normals_path = self._get_normals_path(ref_path)
                ref_normals = self._load_normals(ref_normals_path)
            else:
                input_normals = None
                ref_normals = None
            
            # Apply transforms
            if self.single_img_transform is not None:
                input_img = self.single_img_transform(input_img)
                ref_img = self.single_img_transform(ref_img)
                gt_img = self.single_img_transform(gt_img)
                if self.enable_normals and input_normals is not None:
                    input_normals = self.single_img_transform(input_normals)
                if self.enable_normals and ref_normals is not None:
                    ref_normals = self.single_img_transform(ref_normals)
            
            if self.group_img_transform is not None:
                if self.enable_normals and input_normals is not None and ref_normals is not None:
                    input_img, ref_img, gt_img = self.group_img_transform([input_img, ref_img, gt_img])
                    input_normals, ref_normals = self.group_img_transform([input_normals, ref_normals])
                else:
                    input_img, ref_img, gt_img = self.group_img_transform([input_img, ref_img, gt_img])
            
            # Return appropriate tuple based on whether normals are enabled
            if self.enable_normals:
                return input_img, ref_img, gt_img, input_normals, ref_normals
            else:
                return input_img, ref_img, gt_img
        
        else:
            # TRAINING MODE: 2 images + optionally 1 normal from same scene, different lighting
            # Use deterministic random generation based on index
            rng = self._get_deterministic_random(idx)
            
            ref_candidates = []
            
            if input_scene_bg in self.image_data:
                # Get all lighting conditions for this scene except the current one
                available_lightings = [l for l in self.image_data[input_scene_bg].keys() 
                                     if l != input_lighting]
                
                if available_lightings:
                    ref_lighting = rng.choice(available_lightings)
                    ref_candidates = self.image_data[input_scene_bg][ref_lighting]
            
            if ref_candidates:
                ref_path = rng.choice(ref_candidates)
            else:
                print(f"Warning: No different lighting found for scene={input_scene_bg}, using random image from same scene")
                # Fallback: use any image from the same scene
                scene_images = [img for img in self.all_images 
                              if img['scene_background'] == input_scene_bg]
                if len(scene_images) > 1:
                    # Try to avoid using the same image
                    ref_candidates = [img for img in scene_images if img['path'] != input_path]
                    if ref_candidates:
                        ref_info = rng.choice(ref_candidates)
                        ref_path = ref_info['path']
                    else:
                        ref_path = input_path
                else:
                    ref_path = input_path
            
            # Load reference image
            ref_img = Image.open(ref_path).convert('RGB')
            
            # Load input normals only if enabled (only input normals for training mode)
            if self.enable_normals:
                input_normals = self._load_normals(input_normals_path)
            else:
                input_normals = None
            
            # Apply transforms
            if self.single_img_transform is not None:
                input_img = self.single_img_transform(input_img)
                ref_img = self.single_img_transform(ref_img)
                if self.enable_normals and input_normals is not None:
                    input_normals = self.single_img_transform(input_normals)
            
            if self.group_img_transform is not None:
                if self.enable_normals and input_normals is not None:
                    input_img, ref_img, input_normals = self.group_img_transform([input_img, ref_img, input_normals])
                else:
                    input_img, ref_img = self.group_img_transform([input_img, ref_img])
            
            # Return appropriate tuple based on whether normals are enabled
            if self.enable_normals:
                return input_img, ref_img, input_normals
            else:
                return input_img, ref_img

# Example usage:
"""
import torchvision.transforms as transforms

transform_val = [None, 
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])]
# Validation with cross-scene references
val_dataset = RLSIDDataset(
    root_dir='/data/storage/datasets/RLSID',
    validation_scenes=['00012', '00014', '00020', '00022', '00029', '00030', '00036', '00042', '00043', '00049', '00054', '00058', '00060', '00070', '00076', '00081', '00087', '00098', '00103', '00112', '00113', '00115', '00121', '00124', '00131', '00137', '00138', '00145', '00156', '00157', '00164', '00175', '00176', '00177', '00178', '00179', '00181', '00190', '00198', '00200', '00205', '00206', '00209', '00217', '00219', '00230', '00240', '00244', '00246', '00247', '00255', '00260', '00262', '00276', '00281', '00282', '00283', '00286', '00291', '00306', '00312', '00317', '00320', '00321', '00325', '00332', '00334', '00341', '00343', '00344', '00346', '00349', '00352', '00356', '00361', '00364', '00369', '00371', '00372', '00373', '00380', '00384', '00385', '00392', '00400', '00402', '00411', '00413', '00414', '00419', '00422', '00424', '00428', '00431', '00435', '00437', '00442', '00443', '00448', '00457', '00459', '00461', '00466', '00476', '00477', '00491', '00494', '00495', '00498', '00505', '00506', '00515', '00527', '00529', '00533', '00542', '00552', '00556', '00558', '00566', '00569', '00573', '00576', '00577', '00581', '00586', '00587', '00594', '00600', '00603', '00604', '00605', '00608', '00619', '00636', '00648', '00663', '00669', '00673', '00678', '00687', '00697', '00702', '00704', '00707', '00713', '00726', '00727', '00731', '00736', '00743', '00744', '00749', '00760', '00768', '00772', '00776', '00777', '00785', '00788', '00790', '00792', '00794', '00796', '00802', '00806', '00820', '00821', '00824', '00828', '00832', '00847', '00851', '00852', '00855', '00857', '00867', '00871', '00879', '00880', '00883', '00900', '00902', '00904', '00905', '00909', '00921', '00923', '00931', '00935', '00942', '00946', '00961', '00971', '00981', '00986', '00988', '01000', '01006', '01009', '01014', '01018', '01020', '01024', '01031', '01035', '01036', '01038', '01040', '01041', '01044', '01059', '01060', '01064', '01065', '01066', '01067', '01069', '01076', '01085', '01086', '01088', '01090', '01095', '01097', '01113', '01115', '01124', '01139', '01148', '01150', '01152', '01154', '01155', '01158', '01163', '01166', '01168', '01178', '01182', '01191', '01197', '01200', '01201', '01207', '01210', '01211', '01213', '01214', '01218', '01221', '01223', '01229', '01233', '01236', '01244', '01250', '01253', '01258', '01263', '01270', '01278', '01293', '01295', '01301', '01311', '01314', '01323', '01329', '01338', '01339', '01342', '01343', '01348', '01356', '01363', '01367', '01370', '01384', '01393', '01398', '01404', '01405', '01410', '01416', '01419', '01423', '01424', '01426', '01431', '01433', '01438', '01444', '01451', '01460', '01463', '01466', '01468', '01469', '01479', '01481', '01485', '01487', '01492', '01498', '01505', '01519', '01520', '01525', '01526', '01537', '01547', '01561', '01566', '01570', '01577', '01579', '01582', '01583', '01585', '01588', '01589', '01593', '01594', '01599', '01603', '01604', '01605', '01606', '01610', '01611', '01620', '01622', '01627', '01637', '01646', '01649', '01650', '01652', '01657', '01658', '01668', '01670', '01672', '01678', '01680', '01681', '01686', '01694', '01710', '01711', '01716', '01721', '01726', '01732', '01733', '01735', '01737', '01740', '01744', '01748', '01754', '01766', '01767', '01773', '01779', '01781', '01786', '01790', '01793', '01796', '01804', '01806', '01807', '01809', '01811', '01821', '01831', '01836', '01842', '01845', '01851', '01852', '01858', '01865', '01877', '01887', '01895', '01896', '01901', '01903', '01906', '01912', '01922', '01924', '01929', '01931', '01932', '01935', '01940', '01941', '01946', '01964', '01966', '01969', '01976', '01981', '01985', '01987', '01990', '01993', '01995', '02007', '02047', '02051', '02052', '02057', '02062', '02069', '02073', '02076', '02081', '02083', '02100', '02101', '02102', '02110', '02119', '02125', '02129', '02130', '02131', '02135', '02136', '02145', '02150', '02163', '02167', '02189', '02202', '02216', '02219', '02229', '02230', '02233', '02234', '02235', '02236', '02247', '02256', '02258', '02261', '02263', '02265', '02266', '02267', '02273', '02278', '02287', '02288', '02291', '02307', '02311', '02322', '02335', '02343', '02352', '02355', '02356', '02359', '02373', '02374', '02380', '02385', '02387', '02392', '02396', '02400', '02407', '02416', '02417', '02418', '02424', '02425', '02429', '02441', '02452', '02456', '02491', '02501', '02503', '02513', '02515', '02517', '02518', '02520', '02526', '02530', '02537', '02539', '02542', '02543', '02548', '02550', '02561', '02562', '02569', '02580', '02581', '02584', '02591', '02594', '02604', '02605', '02607', '02624', '02631', '02632', '02634', '02642', '02645', '02653', '02660', '02661', '02669', '02670', '02672', '02676', '02678', '02684', '02694', '02696', '02697', '02701', '02709', '02711', '02713', '02734', '02736', '02737', '02743', '02752', '02767', '02770', '02776', '02779', '02783', '02784', '02790', '02791', '02793', '02797', '02808', '02810', '02814', '02820', '02826', '02828', '02829', '02833', '02834', '02839', '02852', '02854', '02856', '02858', '02859', '02860', '02863', '02865', '02870', '02874', '02886', '02888', '02894', '02900', '02905', '02906', '02911', '02913', '02917', '02922', '02928', '02942', '02945', '02946', '02952', '02958', '02973', '02975', '02976', '02977', '02982', '02995', '02996', '02997', '03001', '03006', '03013', '03015', '03018', '03024', '03038', '03048', '03056', '03057', '03061', '03071', '03072', '03078', '03084', '03087', '03099', '03101', '03116', '03118', '03120', '03128', '03133', '03143', '03145', '03149', '03150', '03152', '03154', '03157', '03158', '03161', '03162', '03164', '03169', '03170', '03172', '03184', '03191', '03192', '03207', '03208', '03213', '03219', '03220', '03237', '03241', '03247', '03251', '03256', '03258', '03271', '03273', '03283', '03288', '03290', '03297', '03302', '03307', '03308', '03333', '03340', '03343', '03345', '03352', '03362', '03366', '03368', '03379', '03391', '03395', '03397', '03398', '03416', '03419', '03430', '03431', '03440', '03447', '03450', '03459', '03467', '03469', '03489', '03491', '03492', '03501', '03503', '03507', '03511', '03518', '03519', '03520', '03539', '03547', '03556', '03569', '03571', '03574', '03595', '03604', '03617', '03621', '03631', '03638', '03640', '03647', '03648', '03656', '03662', '03668', '03670', '03672', '03675', '03680', '03682', '03690', '03691', '03693', '03694', '03706', '03713', '03716', '03717', '03724', '03731', '03733', '03734', '03747', '03758', '03781', '03782', '03791', '03794', '03795', '03801', '03802', '03813', '03815', '03817', '03821', '03826', '03830', '03832', '03856', '03861', '03864', '03867', '03873', '03877', '03883', '03889', '03891', '03892', '03895', '03896', '03900', '03901', '03904', '03909', '03912', '03915', '03920', '03925', '03944', '03948', '03951', '03956', '03960', '03967', '03968', '03970', '03981', '03986', '03997', '03998', '04002', '04004', '04016', '04019', '04022', '04023', '04027', '04032', '04041', '04046', '04049', '04050', '04051', '04054', '04055', '04059', '04060', '04064', '04072', '04074', '04075', '04077', '04082', '04087', '04096', '04097', '04104', '04105', '04115', '04116', '04117', '04120', '04122', '04128', '04130', '04140', '04142', '04146', '04149', '04152', '04156', '04165', '04170', '04175', '04180', '04191', '04192', '04202', '04212', '04213', '04222', '04226', '04232', '04235', '04237', '04254', '04258', '04259', '04267', '04279', '04283', '04287', '04289', '04290', '04291', '04295', '04298', '04313', '04317', '04319', '04326', '04328', '04329', '04363', '04364', '04365', '04368', '04372', '04373', '04376', '04386', '04387', '04389', '04390', '04396', '04398', '04402', '04405', '04408', '04417', '04438', '04439', '04446', '04456', '04460', '04464', '04466', '04467', '04474', '04476', '04489', '04490', '04494', '04498', '04507', '04518', '04519', '04520', '04522', '04536', '04546', '04550', '04560', '04581', '04586', '04589', '04597', '04598', '04601', '04604', '04607', '04608', '04616', '04622', '04624', '04626', '04637', '04641', '04644', '04645', '04650', '04656', '04660', '04669', '04672', '04676', '04678', '04681', '04685', '04689', '04690', '04693', '04694', '04698', '04706', '04707', '04715', '04728', '04729', '04742', '04744', '04751', '04752', '04753', '04756', '04758', '04760', '04767', '04794', '04798', '04800', '04801', '04803', '04813', '04815', '04818', '04825', '04827', '04832', '04842', '04861', '04862', '04875', '04881', '04889', '04890', '04895', '04902', '04923', '04934', '04943', '04950', '04955', '04961', '04972', '04977', '04978', '04980', '04987', '04995', '04998', '05005', '05006', '05013', '05016', '05019', '05025', '05027', '05029', '05036', '05037', '05042', '05046', '05056', '05062', '05063', '05066', '05068', '05081', '05089', '05091', '05092', '05093', '05095', '05097', '05099', '05103', '05108', '05111', '05112', '05116', '05117', '05124', '05125', '05127', '05128', '05135', '05136', '05138', '05139', '05144', '05147', '05148', '17219', '17235', '17239', '17245', '17250', '17258', '17263', '17275', '17276', '17278', '17286', '17287', '17291', '17293', '17294', '17297', '17307', '17309', '17313', '17318', '17324', '17329', '17333', '17334', '17341', '17347', '17359', '17362', '17364', '17367', '17371', '17378', '17385', '17386', '17389', '17402', '17417', '17418', '17421', '17425', '17439', '17441', '17446', '17447', '17455', '17457', '17462', '17466', '17475', '17477', '17483', '17485', '17497', '17506', '17514', '17518', '17521', '17527', '17528', '17538', '17539', '17541', '17546', '17547', '17550', '17554', '17562', '17564', '17565', '17566', '17580', '17583', '17597', '17599', '17601', '17602', '17603', '17606', '17611', '17612', '17616', '17621', '17623', '17635', '17637', '17656', '17662', '17665', '17666', '17674', '17681', '17682', '17683', '17690', '17693', '17696', '17704', '17705', '17719', '17722', '17727', '17729', '17730', '17741', '17745', '17746', '17747', '17751', '17753', '17756', '17757', '17761', '17763', '17765', '17768', '17770', '17771', '17784', '17786', '17788', '17800', '17801', '17805', '17806', '17807', '17809', '17815', '17821', '17828', '17829', '17831', '17836', '17837', '17851', '17855', '17863', '17865', '17873', '17876', '17877', '17883', '17884', '17893', '17894', '17897', '17906', '17908', '17911', '17919', '17932', '17933', '17937', '17946', '17947', '17949', '17953', '17957', '17962', '17968', '17974', '17987', '17990', '17994', '17995', '18002', '18007', '18008', '18011', '18018', '18022', '18026', '18028', '18033', '18037', '18041', '18043', '18045', '18048', '18055', '18063', '18075', '18078', '18080', '18082', '18092', '18094', '18097', '18104', '18106', '18107', '18110', '18111', '18112', '18126', '18128', '18152', '18159', '18162', '18171', '18172', '18178', '18179', '18180', '18181', '18195', '18197', '18206', '18208', '18209', '18211', '18221', '18228', '18232', '18237', '18238', '18239', '18241', '18253', '18270', '18280', '18288', '18292', '18294', '18299', '18301', '18302', '18304', '18312', '18319', '18320', '18323', '18327', '18335', '18337', '18342', '18353', '18354', '18357', '18358', '18360', '18364', '18365', '18375', '18377', '18381', '18382', '18383', '18393', '18394', '18401', '18405', '18414', '18417', '18419', '18431', '18434', '18436', '18441', '18446', '18447', '18451', '18453', '18459', '18460', '18462', '18466', '18468', '18477', '18482', '18490', '18492', '18493', '18496', '18499', '18501', '18511', '18518', '18532', '18542', '18543', '18546', '18551', '18556', '18561', '18566', '18570', '18573', '18582', '18587', '18596', '18601', '18602', '18616', '18617', '18620', '18632', '18648', '18653', '18655', '18660', '18661', '18665', '18676', '18680', '18681', '18683', '18684', '18692', '18694', '18698', '18702', '18710', '18712', '18716', '18718', '18725', '18734', '18739', '18741', '18743', '18754', '18756', '18759', '18761', '18771', '18777', '18778', '18780', '18783', '18786', '18796', '18797', '18802', '18813', '18814', '18815', '18831', '18835', '18836', '18841', '18849', '18850', '18853', '18857', '18859', '18865', '18872', '18878', '18881', '18887', '18892', '18900', '18911', '18912', '18916', '18922', '18928', '18940', '18949', '18959', '18960', '18962', '18981', '18986', '18988', '18989', '18994', '19001', '19008', '19017', '19026', '19034', '19038', '19040', '19045', '19046', '19047', '19050', '19052', '19054', '19058', '19059', '19060', '19066', '19071', '19074', '19077', '19079', '19088', '19096', '19098', '19101', '19102', '19108', '19111', '19114', '19120', '19121', '19125', '19128', '19130', '19136', '19147', '19151', '19154', '19155', '19165', '19168', '19170', '19181', '19189', '19194', '19196', '19202', '19204', '19207', '19211', '19214', '19220', '19225', '19229', '19235', '19237', '19240', '19251', '19257', '19259', '19266', '19267', '19268', '19274', '19277', '19285', '19297', '19298', '19304', '19309', '19313', '19320', '19324', '19326', '19327', '19331', '19335', '19345', '19347', '19348', '19357', '19359', '19364', '19370', '19373', '19374', '19375', '19377', '19383', '19384', '19386', '19388', '19398', '19400', '19405', '19408', '19409', '19419', '19421', '19423', '19426', '19428', '19431', '19432', '19437', '19440', '19441', '19443', '19448', '19451', '19452', '19455', '19459', '19460', '19464', '19467', '19475', '19480', '19483', '19487', '19496', '19498', '19512', '19520', '19528', '19532', '19534', '19535', '19542', '19550', '19555', '19561', '19572', '19573', '19579', '19581', '19586', '19587', '19591', '19604', '19606', '19636', '19650', '19652', '19654', '19655', '19658', '19660', '19665', '19673', '19675', '19690', '19708', '19711', '19718', '19729', '19733', '19745', '19749', '19755', '19756', '19757', '19764', '19772', '19776', '19778', '19786', '19787', '19789', '19790', '19792', '19793', '19796', '19800', '19801', '19805', '19820', '19823', '19829', '19840', '19845', '19857', '19866', '19871', '19879', '19881', '19890', '19904', '19913', '19918', '19923', '19931', '19932', '19936', '19937', '19941', '19945', '19956', '19962', '19965', '19966', '19976', '19978', '19979', '19984', '19989', '19993', '19994', '19998', '20002', '20003', '20007', '20011', '20012', '20014', '20018', '20019', '20020', '20032', '20034', '20037', '20039', '20040', '20041', '20052', '20065', '20074', '20084', '20085', '20086', '20087', '20088', '20098', '20100', '20117', '20118', '20129', '20130', '20144', '20161', '20166', '20170', '20179', '20183', '20186', '20188', '20189', '20215', '20237', '20251', '20253', '20255', '20257', '20265', '20266', '20269', '20274', '20275', '20276', '20280', '20283', '20284', '20285', '20286', '20289', '20290', '20303', '20307', '20313', '20329', '20339', '20342', '20348', '20352', '20366', '20367', '20374', '20375', '20376', '20382', '20389', '20391', '20393', '20396', '20399', '20402', '20403', '20405', '20414', '20416', '20424', '20435', '20438', '20441', '20449', '20452', '20461', '20463', '20465', '20468', '20472', '20477', '20483', '20490', '20494', '20495', '20504', '20511', '20515', '20516', '20518', '20520', '20523', '20526', '20529', '20530', '20535', '20541', '20561', '20566', '20569', '20573', '20579', '20595', '20599', '20605', '20606', '20608', '20614', '20618', '20621', '20629', '20631', '20643', '20644', '20662', '20669', '20677', '20679', '20682', '20688', '20691', '20692', '20701', '20709', '20711', '20713', '20721', '20724', '20729', '20737', '20739', '20740', '20741', '20743', '20755', '20768', '20776', '20782', '20787', '20793', '20800', '20803', '20804', '20813', '20816', '20817', '20820', '20822', '20829', '20835', '20837', '20838', '20846', '20849', '20858', '20866', '20874', '20888', '20889', '20892', '20898', '20919', '20921', '20923', '20929', '20933', '20936', '20943', '20944', '20946', '20951', '20958', '20962', '20964', '20966', '20976', '20977', '20988', '20996', '21010', '21011', '21012', '21026', '21036', '21042', '21051', '21053', '21054', '21060', '21065', '21068', '21078', '21080', '21082', '21083', '21096', '21102', '21106', '21112', '21115', '21116', '21131', '21133', '21138', '21149', '21155', '21158', '21162', '21164', '21172', '21175', '21193', '21202', '21203', '21213', '21214', '21218', '21219', '21223', '21224', '21233', '21235', '21239', '21244', '21247', '21251', '21253', '21256', '21260', '21265', '21270', '21273', '21276', '21277', '21284', '21286', '21289', '21293', '21294', '21304', '21310', '21313', '21314', '21318', '21333', '21340', '21348', '21353', '21354', '21363', '21369', '21373', '21380', '21381', '21383', '21387', '21392', '21394', '21403', '21406', '21407', '21411', '21412', '21425', '21426', '21428', '21435', '21439', '21442', '21444', '21447', '21449', '21455', '21457', '21468', '21471', '21483', '21485', '21488', '21489', '21495', '21496', '21497', '21504', '21505', '21506', '21507', '21510', '21512', '21516', '21518', '21519', '21521', '21530', '21537', '21540', '21542', '21544', '21550', '21561', '21565', '21570', '21572', '21577', '21579', '21590', '21594', '21595', '21597', '21603', '21605', '21607', '21613', '21614', '21623', '21626', '21632', '21639', '21640', '21644', '21645', '21648', '21649', '21652', '21655', '21659', '21663', '21664', '21666', '21673', '21674', '21683', '21686', '21690', '21695', '21702', '21703', '21705', '21709', '21713', '21717', '21724', '21728', '21742', '21743', '21752', '21758', '21770', '21772', '21783', '21784', '21793', '21805', '21809', '21821', '21827', '21838', '21843', '21844', '21846', '21851', '21872', '21875', '21877', '21878', '21887', '21891', '21896', '21903', '21907', '21915', '21916', '21919', '21920', '21922', '21929', '21936', '21937', '21946', '21947', '21949', '21951', '21954', '21955', '21957', '21958', '21960', '21964', '21968', '21971', '21973', '21979', '21981', '21986', '21988', '21989', '21993', '21998', '22004', '22005', '22006', '22010', '22045', '22048', '22056', '22062', '22063', '22069', '22074', '22079', '22095', '22097', '22100', '22105', '22106', '22111', '22122', '22123', '22124', '22127', '22128', '22129'],
    img_transform=transform_val,
    is_validation=True,
    )
    # Example usage of the dataset
for i in range(len(val_dataset)):
    input_img, ref_img, gt, normals_in, normals_ref = val_dataset[i]
    print(f"Input image shape: {input_img.shape}, Reference image shape: {ref_img.shape}")
    #tensor to PIL for saving
    input_img = transforms.ToPILImage()(input_img)
    ref_img = transforms.ToPILImage()(ref_img)
    gt_img = transforms.ToPILImage()(gt)
    normals_in = transforms.ToPILImage()(normals_in)
    normals_ref = transforms.ToPILImage()(normals_ref)
    input_img.save('input_image.png')
    ref_img.save('ref_image.png')
    gt_img.save('gt_image.png')
    normals_in.save('normals_in.png')
    normals_ref.save('normals_ref.png')"""