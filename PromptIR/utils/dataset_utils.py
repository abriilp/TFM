import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = "/home/apinyol/TFM/TFM/PromptIR/data_dir/rainy/rainTrain.txt" #self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids+= ["" + id_.strip() for id_ in open(rs)] #self.args.derain_dir
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
    

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    
import os
import glob
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ALNDatasetRLSID(Dataset):
    def __init__(self, root_dir, is_validation=False, resize_width_to=None, 
                 patch_size=None, filter_of_images=None, image_folder='Image', max_training_images=None, max_validation_images=None, split_seed=42):
        """
        Modified RLSID Dataset to replace ALNDatasetGeom
        
        Args:
            root_dir: Path to the dataset root directory
            is_validation: If True, uses validation scenes (20%), else training scenes (80%)
            resize_width_to: Width to resize images to (maintaining aspect ratio)
            patch_size: Size for random cropping
            filter_of_images: List of image filters (scene indices to include)
            image_folder: Which folder to use ('Image', 'Reflectance', or 'Shading')
            max_training_images: Maximum number of images for training per epoch (None for all)
            max_validation_images: Maximum number of images for validation per epoch (None for all)
            split_seed: Seed for reproducible train/validation split
        """
        super(ALNDatasetRLSID, self).__init__()
        
        self.root_dir = root_dir
        self.is_validation = is_validation
        self.resize_width_to = resize_width_to
        self.patch_size = patch_size
        self.filter_of_images = filter_of_images
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
                                                'target': gt_img
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
    
    def __len__(self):
        """Return the number of images in the current epoch"""
        return len(self.current_image_paths)
    
    def __getitem__(self, idx):
        """
        Returns same format as ALNDatasetGeom:
        [filename_prefix, 0], concatenated_input, target_img
        """
        paths = self.current_image_paths[idx]
        input_path = paths['input']
        reference_path = paths['reference']
        target_path = paths['target']
        
        # Load images
        input_img = self.toTensor(Image.open(input_path).convert('RGB'))
        reference_img = self.toTensor(Image.open(reference_path).convert('RGB'))
        target_img = self.toTensor(Image.open(target_path).convert('RGB'))
        
        # Apply same transformations as ALNDataset
        if self.resize_width_to is not None:
            # Resize maintaining aspect ratio
            input_img = TF.resize(input_img, (int((input_img.shape[1]*self.resize_width_to)/input_img.shape[2]), self.resize_width_to))
            reference_img = TF.resize(reference_img, (int((reference_img.shape[1]*self.resize_width_to)/reference_img.shape[2]), self.resize_width_to))
            target_img = TF.resize(target_img, (int((target_img.shape[1]*self.resize_width_to)/target_img.shape[2]), self.resize_width_to))
            
            # Random horizontal flip with probability 0.5
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                reference_img = TF.hflip(reference_img)
                target_img = TF.hflip(target_img)
        
        if self.patch_size is not None:
            # Random crop with same parameters for all images
            i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.patch_size, self.patch_size))
            input_img = TF.crop(input_img, i, j, h, w)
            reference_img = TF.crop(reference_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)
        
        # Concatenate input image with reference image along the channel dimension (same as ALNDataset)
        concatenated_input = torch.cat([input_img, reference_img], dim=0)
        
        # Return same format as ALNDatasetGeom: [filename_prefix, 0], concatenated_input, target_img
        filename_prefix = os.path.basename(input_path).split('_')[0]
        
        return [filename_prefix, 0], concatenated_input, target_img
