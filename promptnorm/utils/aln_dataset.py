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