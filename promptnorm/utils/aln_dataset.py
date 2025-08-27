from PIL import Image
import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import torch

class ALNDatasetGeom(Dataset):
    def __init__(self, input_folder, target_folder, geom_folder, reference_folder, resize_width_to=None, patch_size=None, filter_of_images=None):
        super(ALNDatasetGeom, self).__init__()
        self.input_folder = input_folder
        self.geom_folder = geom_folder
        self.target_folder = target_folder
        self.reference_folder = reference_folder
        self.resize_width_to = resize_width_to
        self.patch_size = patch_size
        self.filter_of_images = filter_of_images

        self._init_paths()
    
        self.toTensor = ToTensor()

    def _init_paths(self):
        self.image_paths = []
        for input_path in glob(self.input_folder + '/*'):
            if self.filter_of_images is not None:
                if not any(["/"+str(filt)+"_in" in input_path for filt in self.filter_of_images]):
                    continue
            target_path = self.target_folder + '/' + input_path.split('/')[-1].replace('_in.png', '_gt.png')
            geom_path = self.geom_folder + '/' + input_path.split('/')[-1].replace('_in.png', '_normal.png')
            reference_path = self.reference_folder + '/' + input_path.split('/')[-1].replace('_in.png', '_ref.png')

            # Ensure target and reference files exist
            if os.path.exists(target_path) and os.path.exists(reference_path):
                self.image_paths.append({
                    'input': input_path, 
                    'target': target_path, 
                    'geom': geom_path,
                    'reference': reference_path
                })
            else:
                missing_files = []
                if not os.path.exists(target_path):
                    missing_files.append(f"Target: {target_path}")
                if not os.path.exists(reference_path):
                    missing_files.append(f"Reference: {reference_path}")
                raise FileNotFoundError(f"Files not found: {', '.join(missing_files)}")

        print(f"Found {len(self.image_paths)} image pairs")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        input_path = self.image_paths[idx]['input']
        geom_path = self.image_paths[idx]['geom']
        target_path = self.image_paths[idx]['target']
        reference_path = self.image_paths[idx]['reference']

        input_img = self.toTensor(Image.open(input_path).convert('RGB'))
        normal_img = self.toTensor(Image.open(geom_path).convert('RGB'))
        target_img = self.toTensor(Image.open(target_path).convert('RGB'))
        reference_img = self.toTensor(Image.open(reference_path).convert('RGB'))

        if self.resize_width_to is not None:
            input_img = TF.resize(input_img, (int((input_img.shape[1]*self.resize_width_to)/input_img.shape[2]), self.resize_width_to))
            normal_img = TF.resize(normal_img, (int((normal_img.shape[1]*self.resize_width_to)/normal_img.shape[2]), self.resize_width_to))
            target_img = TF.resize(target_img, (int((target_img.shape[1]*self.resize_width_to)/target_img.shape[2]), self.resize_width_to))
            reference_img = TF.resize(reference_img, (int((reference_img.shape[1]*self.resize_width_to)/reference_img.shape[2]), self.resize_width_to))

            # invert the image horizontally with a probability of 0.5
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                normal_img = TF.hflip(normal_img)
                target_img = TF.hflip(target_img)
                reference_img = TF.hflip(reference_img)

        if self.patch_size is not None:
            # crop input and target images with the same random crop
            i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.patch_size, self.patch_size))
            input_img = TF.crop(input_img, i, j, h, w)
            normal_img = TF.crop(normal_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)
            reference_img = TF.crop(reference_img, i, j, h, w)

        # Concatenate input image with reference image along the channel dimension
        concatenated_input = torch.cat([input_img, reference_img], dim=0)

        return [input_path.split('/')[-1].split('_')[0], 0], concatenated_input, normal_img, target_img
    

#provar abril
#provar abril
import os
import glob
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ALNDatasetGeomRLSID(Dataset):
    def __init__(self, root_dir, is_validation=False, resize_width_to=None, 
                 patch_size=None, filter_of_images=None, image_folder='Image', 
                 normals_folder=None, max_training_images=None, max_validation_images=None, split_seed=42):
        """
        Modified RLSID Dataset to replace ALNDatasetGeom
        
        Args:
            root_dir: Path to the dataset root directory
            is_validation: If True, uses validation scenes (20%), else training scenes (80%)
            resize_width_to: Width to resize images to (maintaining aspect ratio)
            patch_size: Size for random cropping
            filter_of_images: List of image filters (scene indices to include)
            image_folder: Which folder to use ('Image', 'Reflectance', or 'Shading')
            normals_folder: Path to the normals directory (can be absolute or relative to root_dir)
            max_training_images: Maximum number of images for training per epoch (None for all)
            max_validation_images: Maximum number of images for validation per epoch (None for all)
            split_seed: Seed for reproducible train/validation split
        """
        super(ALNDatasetGeomRLSID, self).__init__()
        
        self.root_dir = root_dir
        self.is_validation = is_validation
        self.resize_width_to = resize_width_to
        self.patch_size = patch_size
        self.filter_of_images = filter_of_images
        self.normals_folder = normals_folder
        self.max_training_images = max_training_images
        self.max_validation_images = max_validation_images
        self.split_seed = split_seed
        self.image_folder = image_folder
        
        # Current epoch number (for reproducible per-epoch sampling)
        self.current_epoch = 0
        
        # Initialize paths and data structures
        self._init_paths()
        
        # Transform to tensor (same as ALNDataset)
        self.toTensor = ToTensor()
        
        # Initialize first epoch
        self.set_epoch(0)
        
    def _init_paths(self):
        """Initialize image paths and organize by scenes"""
        # Path to the image directory
        self.image_dir = os.path.join(self.root_dir, self.image_folder)
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        # Path to the normals directory
        if self.normals_folder is not None:
            if os.path.isabs(self.normals_folder):
                # Absolute path
                self.normals_dir = self.normals_folder
            else:
                # Relative to root_dir
                self.normals_dir = os.path.join(self.root_dir, self.normals_folder)
        else:
            # Default: look for normals in image_folder/normals
            self.normals_dir = os.path.join(self.image_dir, 'normals')
        
        if not os.path.exists(self.normals_dir):
            print(f"Warning: Normals directory not found: {self.normals_dir}")
            self.normals_dir = None
        
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
                    
                    # Apply filter if specified
                    if self.filter_of_images is not None:
                        if scene_idx not in [str(f) for f in self.filter_of_images]:
                            continue
                    
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
        
        # Deterministic 80/20 split using fixed seed
        all_scene_backgrounds = sorted(list(all_scene_backgrounds))
        random.seed(self.split_seed)
        random.shuffle(all_scene_backgrounds)
        
        num_scenes = len(all_scene_backgrounds)
        if num_scenes < 2:
            raise ValueError(f"Need at least 2 scenes for split, found {num_scenes}")
        
        split_idx = max(1, int(num_scenes * 0.8))
        self.train_scenes = all_scene_backgrounds[:split_idx]
        self.validation_scenes = all_scene_backgrounds[split_idx:]
        
        # Select scenes based on mode
        if self.is_validation:
            self.active_scenes = self.validation_scenes
        else:
            self.active_scenes = self.train_scenes
        
        if not self.active_scenes:
            mode = "validation" if self.is_validation else "training"
            raise ValueError(f"No scenes found for {mode} mode")
        
        # Pre-compute valid cross-scene pairs
        self._compute_cross_scene_pairs()
        
        # Build ALL possible image pairs list (don't limit here - do it per epoch)
        self.all_image_paths = []
        
        # For each scene in active scenes, create pairs with other scenes
        for input_scene in self.active_scenes:
            if input_scene not in self.valid_pairs:
                continue
                
            for input_lighting, input_imgs in self.image_data[input_scene].items():
                for input_img in input_imgs:
                    # Find valid reference scenes and lightings
                    valid_refs = self.valid_pairs[input_scene]
                    if valid_refs:
                        for ref_scene, shared_lightings in valid_refs.items():
                            for ref_lighting in shared_lightings:
                                ref_imgs = self.image_data[ref_scene][ref_lighting]
                                for ref_img in ref_imgs:
                                    # GT: same scene as input, same lighting as reference
                                    if ref_lighting in self.image_data[input_scene]:
                                        gt_imgs = self.image_data[input_scene][ref_lighting]
                                        for gt_img in gt_imgs:
                                            self.all_image_paths.append({
                                                'input': input_img,
                                                'reference': ref_img,
                                                'target': gt_img,
                                                'geom': self._get_normals_path(input_img) if self.normals_dir else None
                                            })
        
        # Current epoch image paths (will be set by set_epoch)
        self.current_image_paths = []
        
        print(f"Mode: {'Validation' if self.is_validation else 'Training'}")
        print(f"Active scenes: {len(self.active_scenes)}")
        print(f"Total possible image pairs: {len(self.all_image_paths)}")
        
        # Determine max images for this mode
        self.max_images_per_epoch = self.max_validation_images if self.is_validation else self.max_training_images
        if self.max_images_per_epoch is not None:
            print(f"Max images per epoch: {self.max_images_per_epoch}")
        else:
            print("Using all images every epoch")
    
    def set_epoch(self, epoch):
        """
        Set the current epoch and sample a random subset of images for this epoch.
        This should be called at the beginning of each epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        
        if self.max_images_per_epoch is None or self.max_images_per_epoch >= len(self.all_image_paths):
            # Use all images
            self.current_image_paths = self.all_image_paths.copy()
        else:
            # Sample a random subset for this epoch
            # Use epoch-specific seed for reproducible but different sampling each epoch
            epoch_seed = self.split_seed + epoch * 1000
            random.seed(epoch_seed)
            self.current_image_paths = random.sample(self.all_image_paths, self.max_images_per_epoch)
        
        # Shuffle the current epoch's images for good measure
        random.shuffle(self.current_image_paths)
        
        if epoch == 0 or epoch % 10 == 0:  # Print every 10 epochs to avoid spam
            print(f"Epoch {epoch}: Using {len(self.current_image_paths)} images")
    
    def _compute_cross_scene_pairs(self):
        """
        Pre-compute pairs of scenes that share at least one lighting condition.
        """
        self.valid_pairs = {}  # input_scene -> {ref_scene: [shared_lightings]}
        
        for input_scene in self.active_scenes:
            if input_scene not in self.image_data:
                continue
                
            input_lightings = set(self.image_data[input_scene].keys())
            self.valid_pairs[input_scene] = {}
            
            # Look for reference scenes in ALL scenes (not just active ones for cross-scene)
            all_scenes = list(self.image_data.keys())
            for ref_scene in all_scenes:
                if ref_scene == input_scene or ref_scene not in self.image_data:
                    continue
                
                ref_lightings = set(self.image_data[ref_scene].keys())
                shared_lightings = input_lightings.intersection(ref_lightings)
                
                if shared_lightings:
                    self.valid_pairs[input_scene][ref_scene] = list(shared_lightings)
    
    def _get_normals_path(self, img_path):
        """Get the corresponding normals path for an image path"""
        if self.normals_dir is None:
            return None
        filename = os.path.basename(img_path)
        return os.path.join(self.normals_dir, filename)
    
    def _load_normals(self, normals_path):
        """Load normal image from path"""
        if normals_path is None or not os.path.exists(normals_path):
            # Return a default normal map (pointing forward)
            return Image.new('RGB', (256, 256), (128, 128, 255))
        
        try:
            return Image.open(normals_path).convert('RGB')
        except Exception as e:
            print(f"Error loading normals from {normals_path}: {e}")
            return Image.new('RGB', (256, 256), (128, 128, 255))
    
    def __len__(self):
        """Return the number of images in the current epoch"""
        return len(self.current_image_paths)
    
    def __getitem__(self, idx):
        """
        Returns same format as ALNDatasetGeom:
        [filename_prefix, 0], concatenated_input, normal_img, target_img
        """
        paths = self.current_image_paths[idx]
        input_path = paths['input']
        reference_path = paths['reference']
        target_path = paths['target']
        geom_path = paths['geom']
        
        # Load images
        input_img = self.toTensor(Image.open(input_path).convert('RGB'))
        reference_img = self.toTensor(Image.open(reference_path).convert('RGB'))
        target_img = self.toTensor(Image.open(target_path).convert('RGB'))
        normal_img = self.toTensor(self._load_normals(geom_path))
        
        # Apply same transformations as ALNDataset
        if self.resize_width_to is not None:
            # Resize maintaining aspect ratio
            input_img = TF.resize(input_img, (int((input_img.shape[1]*self.resize_width_to)/input_img.shape[2]), self.resize_width_to))
            reference_img = TF.resize(reference_img, (int((reference_img.shape[1]*self.resize_width_to)/reference_img.shape[2]), self.resize_width_to))
            target_img = TF.resize(target_img, (int((target_img.shape[1]*self.resize_width_to)/target_img.shape[2]), self.resize_width_to))
            normal_img = TF.resize(normal_img, (int((normal_img.shape[1]*self.resize_width_to)/normal_img.shape[2]), self.resize_width_to))
            
            # Random horizontal flip with probability 0.5
            if not self.is_validation and random.random() > 0.5:
                input_img = TF.hflip(input_img)
                reference_img = TF.hflip(reference_img)
                target_img = TF.hflip(target_img)
                normal_img = TF.hflip(normal_img)
        
        if not self.is_validation and self.patch_size is not None:
            # Random crop with same parameters for all images
            i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.patch_size, self.patch_size))
            input_img = TF.crop(input_img, i, j, h, w)
            reference_img = TF.crop(reference_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)
            normal_img = TF.crop(normal_img, i, j, h, w)
        
        # Concatenate input image with reference image along the channel dimension (same as ALNDataset)
        concatenated_input = torch.cat([input_img, normal_img], dim=0)
        
        # Return same format as ALNDatasetGeom: [filename_prefix, 0], concatenated_input, normal_img, target_img
        filename_prefix = os.path.basename(input_path).split('_')[0]
        
        return [filename_prefix, 0], concatenated_input, reference_img, target_img


class ALNDatasetGeomRSR(Dataset):
    def __init__(self, root_dir, is_validation=False, resize_width_to=None, 
                 patch_size=None, filter_of_images=None, 
                 max_training_images=None, max_validation_images=None, 
                 split_seed=42, use_all_scenes=False, validation_scenes=None,
                 normals_folder=None):
        """
        Modified RSR Dataset to replace ALNDatasetGeom
        
        Args:
            root_dir: Path to the dataset root directory containing scene folders
            is_validation: If True, uses validation scenes, else training scenes
            resize_width_to: Width to resize images to (maintaining aspect ratio)
            patch_size: Size for random cropping
            filter_of_images: List of scene names to include (if None, uses auto-split)
            max_training_images: Maximum number of images for training per epoch (None for all)
            max_validation_images: Maximum number of images for validation per epoch (None for all)
            split_seed: Seed for reproducible train/validation split
            use_all_scenes: If True, uses all available scenes regardless of other parameters
            validation_scenes: Specific scenes to use for validation (when filter_of_images=None)
            normals_folder: Path to normals directory (can be None for no normals support)
        """
        super(ALNDatasetGeomRSR, self).__init__()
        
        self.root_dir = root_dir
        self.is_validation = is_validation
        self.resize_width_to = resize_width_to
        self.patch_size = patch_size
        self.filter_of_images = filter_of_images
        self.max_training_images = max_training_images
        self.max_validation_images = max_validation_images
        self.split_seed = split_seed
        self.use_all_scenes = use_all_scenes
        self.validation_scenes = validation_scenes
        self.normals_folder = normals_folder
        
        # Current epoch number (for reproducible per-epoch sampling)
        self.current_epoch = 0
        
        # Initialize paths and data structures
        self._init_paths()
        
        # Transform to tensor (same as ALNDataset)
        self.toTensor = ToTensor()
        
        # Initialize first epoch
        self.set_epoch(0)
        
    def _init_paths(self):
        """Initialize image paths and organize by scenes"""
        
        # Setup normals directory if provided
        if self.normals_folder is not None:
            if os.path.isabs(self.normals_folder):
                self.normals_dir = self.normals_folder
            else:
                self.normals_dir = os.path.join(self.root_dir, self.normals_folder)
            
            if not os.path.exists(self.normals_dir):
                print(f"Warning: Normals directory not found: {self.normals_dir}")
                self.normals_dir = None
        else:
            self.normals_dir = None
        
        # Get all scene directories
        all_scenes = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        all_scenes.sort()  # Sort for reproducible splits
        
        if not all_scenes:
            raise ValueError(f"No scene directories found in {self.root_dir}")
        
        # Determine which scenes to use
        if self.use_all_scenes:
            # Use all available scenes
            self.scenes = all_scenes
            self.all_available_scenes = all_scenes
            print(f"Using all scenes: {self.scenes}")
        elif self.filter_of_images is not None:
            # Use explicitly provided scenes
            self.scenes = [s for s in self.filter_of_images if s in all_scenes]
            self.all_available_scenes = all_scenes  # Keep all for cross-scene references
        else:
            # Auto-split: use all scenes except last 2 for training, last 2 for validation
            if len(all_scenes) < 2:
                raise ValueError(f"Need at least 2 scenes for auto-split, found {len(all_scenes)}")
            
            # Determine validation scenes
            if self.validation_scenes is not None:
                val_scenes = [s for s in self.validation_scenes if s in all_scenes]
                if len(val_scenes) < 1:
                    # Fallback to last 2 scenes if provided validation_scenes are invalid
                    val_scenes = all_scenes[-2:]
                    print(f"Warning: Using last 2 scenes for validation: {val_scenes}")
            else:
                # Use last 2 scenes for validation by default
                val_scenes = all_scenes[-2:]
            
            if self.is_validation:
                self.scenes = val_scenes
            else:
                # Training: use all scenes except validation scenes
                self.scenes = [s for s in all_scenes if s not in val_scenes]
            
            # Store all available scenes for cross-scene reference selection
            self.all_available_scenes = all_scenes
        
        if not self.scenes:
            raise ValueError(f"No valid scenes found")
        
        print(f"Mode: {'Validation' if self.is_validation else 'Training'}")
        print(f"Using scenes: {self.scenes}")
        if hasattr(self, 'all_available_scenes'):
            print(f"All available scenes for cross-reference: {self.all_available_scenes}")
        
        # Parse all images and organize by scene, camera_pos, light_pos
        self.image_data = {}  # scene -> camera_pos -> light_pos -> [image_paths]
        
        # If validation mode with cross-scene references, we need data from all scenes
        scenes_to_parse = self.all_available_scenes if hasattr(self, 'all_available_scenes') else self.scenes
        
        for scene in scenes_to_parse:
            scene_dir = os.path.join(self.root_dir, scene)
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
                        # Only use light color 1 (same as original RSR dataset)
                        if light_color == '1':
                            camera_pos = parts[8]
                            light_pos = parts[10].split('.')[0]  # Remove file extension
                            
                            if camera_pos not in scene_data:
                                scene_data[camera_pos] = {}
                            if light_pos not in scene_data[camera_pos]:
                                scene_data[camera_pos][light_pos] = []
                            
                            scene_data[camera_pos][light_pos].append(img_path)
                            
                except (IndexError, ValueError) as e:
                    print(f"Skipping invalid filename: {filename} - {e}")
                    continue
            
            if scene_data:
                self.image_data[scene] = scene_data
        
        # Build ALL possible image pairs list (don't limit here - do it per epoch)
        self.all_image_paths = []
        
        # Create image pairs based on mode
        if self.is_validation:
            # VALIDATION MODE: Cross-scene references
            self._create_validation_pairs()
        else:
            # TRAINING MODE: Same-scene references with different lighting
            self._create_training_pairs()
        
        print(f"Total possible image pairs: {len(self.all_image_paths)}")
        
        # Determine max images for this mode
        self.max_images_per_epoch = self.max_validation_images if self.is_validation else self.max_training_images
        if self.max_images_per_epoch is not None:
            print(f"Max images per epoch: {self.max_images_per_epoch}")
        else:
            print("Using all images every epoch")
    
    def _create_validation_pairs(self):
        """Create cross-scene image pairs for validation"""
        for input_scene in self.scenes:
            if input_scene not in self.image_data:
                continue
                
            for input_camera_pos, input_cam_data in self.image_data[input_scene].items():
                for input_light_pos, input_imgs in input_cam_data.items():
                    for input_img in input_imgs:
                        # Find cross-scene references (from different scenes)
                        for ref_scene in self.all_available_scenes:
                            if ref_scene == input_scene or ref_scene not in self.image_data:
                                continue
                            
                            # Get all possible references from this different scene
                            for ref_camera_pos, ref_cam_data in self.image_data[ref_scene].items():
                                for ref_light_pos, ref_imgs in ref_cam_data.items():
                                    for ref_img in ref_imgs:
                                        # GT: Same scene and camera_pos as input, same light_pos as reference
                                        if (ref_light_pos in self.image_data[input_scene][input_camera_pos]):
                                            gt_candidates = self.image_data[input_scene][input_camera_pos][ref_light_pos]
                                            for gt_img in gt_candidates:
                                                self.all_image_paths.append({
                                                    'input': input_img,
                                                    'reference': ref_img,
                                                    'target': gt_img,
                                                    'geom': self._get_normals_path(input_img) if self.normals_dir else None
                                                })
    
    def _create_training_pairs(self):
        """Create same-scene image pairs for training"""
        for input_scene in self.scenes:
            if input_scene not in self.image_data:
                continue
                
            for input_camera_pos, input_cam_data in self.image_data[input_scene].items():
                for input_light_pos, input_imgs in input_cam_data.items():
                    for input_img in input_imgs:
                        # Find references from same scene and camera_pos but different light_pos
                        available_light_pos = [lp for lp in input_cam_data.keys() if lp != input_light_pos]
                        
                        for ref_light_pos in available_light_pos:
                            ref_imgs = input_cam_data[ref_light_pos]
                            for ref_img in ref_imgs:
                                # For training, target is the same as reference
                                # (input gets relit to look like reference)
                                self.all_image_paths.append({
                                    'input': input_img,
                                    'reference': ref_img,
                                    'target': ref_img,  # Target is same as reference in training
                                    'geom': self._get_normals_path(input_img) if self.normals_dir else None
                                })
    
    def set_epoch(self, epoch):
        """
        Set the current epoch and sample a random subset of images for this epoch.
        This should be called at the beginning of each epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        
        if self.max_images_per_epoch is None or self.max_images_per_epoch >= len(self.all_image_paths):
            # Use all images
            self.current_image_paths = self.all_image_paths.copy()
        else:
            # Sample a random subset for this epoch
            # Use epoch-specific seed for reproducible but different sampling each epoch
            epoch_seed = self.split_seed + epoch * 1000
            random.seed(epoch_seed)
            self.current_image_paths = random.sample(self.all_image_paths, self.max_images_per_epoch)
        
        # Shuffle the current epoch's images for good measure
        random.shuffle(self.current_image_paths)
        
        if epoch == 0 or epoch % 10 == 0:  # Print every 10 epochs to avoid spam
            print(f"Epoch {epoch}: Using {len(self.current_image_paths)} images")
    
    def _get_normals_path(self, img_path):
        """Get the corresponding normals path for an image path"""
        if self.normals_dir is None:
            return None
        
        # Get the relative path from the root_dir to maintain scene structure
        rel_path = os.path.relpath(img_path, self.root_dir)
        normals_path = os.path.join(self.normals_dir, rel_path).split('.')[0] + '.png'

        return normals_path
    
    def _load_normals(self, normals_path):
        """Load normal image from path"""
        if normals_path is None or not os.path.exists(normals_path):
            # Return a default normal map (pointing forward)
            return Image.new('RGB', (256, 256), (128, 128, 255))
        
        try:
            return Image.open(normals_path).convert('RGB')
        except Exception as e:
            print(f"Error loading normals from {normals_path}: {e}")
            return Image.new('RGB', (256, 256), (128, 128, 255))
    
    def __len__(self):
        """Return the number of images in the current epoch"""
        return len(self.current_image_paths)
    
    def __getitem__(self, idx):
        """
        Returns same format as ALNDatasetGeom:
        [filename_prefix, 0], concatenated_input, normal_img, target_img
        """
        paths = self.current_image_paths[idx]
        input_path = paths['input']
        reference_path = paths['reference']
        target_path = paths['target']
        geom_path = paths['geom']
        
        # Load images
        input_img = self.toTensor(Image.open(input_path).convert('RGB'))
        reference_img = self.toTensor(Image.open(reference_path).convert('RGB'))
        target_img = self.toTensor(Image.open(target_path).convert('RGB'))
        normal_img = self.toTensor(self._load_normals(geom_path))
        
        # Apply same transformations as ALNDataset
        if self.resize_width_to is not None:
            # Resize maintaining aspect ratio
            input_img = TF.resize(input_img, (int((input_img.shape[1]*self.resize_width_to)/input_img.shape[2]), self.resize_width_to))
            reference_img = TF.resize(reference_img, (int((reference_img.shape[1]*self.resize_width_to)/reference_img.shape[2]), self.resize_width_to))
            target_img = TF.resize(target_img, (int((target_img.shape[1]*self.resize_width_to)/target_img.shape[2]), self.resize_width_to))
            normal_img = TF.resize(normal_img, (int((normal_img.shape[1]*self.resize_width_to)/normal_img.shape[2]), self.resize_width_to))
            
            # Random horizontal flip with probability 0.5 (only for training)
            if not self.is_validation and random.random() > 0.5:
                input_img = TF.hflip(input_img)
                reference_img = TF.hflip(reference_img)
                target_img = TF.hflip(target_img)
                normal_img = TF.hflip(normal_img)
        
        if not self.is_validation and self.patch_size is not None:
            # Random crop with same parameters for all images (only for training)
            i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.patch_size, self.patch_size))
            input_img = TF.crop(input_img, i, j, h, w)
            reference_img = TF.crop(reference_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)
            normal_img = TF.crop(normal_img, i, j, h, w)
        
        # Concatenate input image with reference image along the channel dimension (same as ALNDataset)
        concatenated_input = torch.cat([input_img, reference_img], dim=0)
        
        # Return same format as ALNDatasetGeom: [filename_prefix, 0], concatenated_input, normal_img, target_img
        filename_prefix = os.path.basename(input_path).split('_')[0]
        
        return [filename_prefix, 0], concatenated_input, normal_img, target_img