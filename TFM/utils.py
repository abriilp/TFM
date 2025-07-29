import builtins, pdb
import datetime
import os,glob
import time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import cv2
import json
F = torch.nn.functional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Standard library imports
import argparse
import builtins
import copy
import glob
import json
import math
import os
import pdb
import random
import shutil
import time
from typing import Union, List, Optional, Callable

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from PIL import Image
from torch import autograd
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Third-party loss/metrics
from pytorch_losses import gradient_loss
from pytorch_ssim import SSIM as compute_SSIM_loss

from dataset_utils import *

# wandb import
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from diffusers import DiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

class affine_crop_resize(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, flip = True, **kargs):
        self.size = size
        super(affine_crop_resize, self).__init__(size, **kargs)
        self.flip = flip
    def __call__(self, img_list):
        img = img_list[0]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img_h, img_w = img.shape[1:]
        affine = random_crop_resize_affine(j,j+w,i,i+h, img_w, img_h)
        if self.flip and np.random.rand() >= 0.5:
            affine = affine @ flip_affine()
        affine = torch.from_numpy(affine)[:2][None,...]
        out_img = []
        for img in img_list:
            out_img.append(apply_affine(img[None,...], affine, out_size = self.size)[0])
        return out_img

def apply_affine(img, affine, out_size = None, mode = 'bilinear', align_corners = False):
    if out_size is None:
        out_size = img.shape
    elif isinstance(out_size, int):
        out_size = torch.Size([img.shape[0], img.shape[1], out_size, out_size])
    elif isinstance(out_size, tuple):
        out_size = torch.Size([img.shape[0], img.shape[1], out_size[0], out_size[1]])
    grid = F.affine_grid(affine.float(), out_size, align_corners = align_corners)
    out = F.grid_sample(img, grid, mode, align_corners = align_corners)
    return out

def random_crop_resize_affine(x1,x2,y1,y2,width,height):
    affine = np.eye(3)
    affine[0,0] = (x2 - x1) / (width - 1)
    affine[1,1] = (y2 - y1) / (height - 1)
    affine[0,2] = (x1 + x2 - width + 1) / (width - 1)
    affine[1,2] = (y1 + y2 - height + 1) / (height - 1)
    return affine

def flip_affine(hori = True):
    affine = np.eye(3)
    if hori:
        affine[0,0] = -1
        affine[0,2] = 0
    else:
        affine[1,1] = -1
        affine[1,2] = 0
    return affine

class multi_affine_crop_resize(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, flip = True, **kargs):
        self.size = size
        super(multi_affine_crop_resize, self).__init__(size, **kargs)
        self.flip = flip

    def get_params(self, height, width, scale, ratio, num_of_sample):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        target_area = area * torch.empty(num_of_sample).uniform_(scale[0], scale[1])
        aspect_ratio = torch.exp(torch.empty(num_of_sample).uniform_(log_ratio[0], log_ratio[1]))

        w = torch.round(torch.sqrt(target_area * aspect_ratio)).long()
        h = torch.round(torch.sqrt(target_area / aspect_ratio)).long()
        mask1 = torch.logical_and(w > 0, w <= width)
        mask2 = torch.logical_and(h > 0, h <= height)
        mask = torch.logical_and(mask1, mask2)
        h = h[mask]
        w = w[mask]
        i = (torch.rand(size = (h.shape[0],)) * (height - h + 1)).long()
        j = (torch.rand(size = (h.shape[0],)) * (width- w + 1)).long()
        x1 = j
        x2 = j + w
        y1 = i
        y2 = i + h
        affine = torch.eye(3)[None,...].expand(i.shape[0],-1,-1).clone()
        affine[:,0,0] = (x2 - x1) / (width - 1)
        affine[:,1,1] = (y2 - y1) / (height - 1)
        affine[:,0,2] = (x1 + x2 - width + 1) / (width - 1)
        affine[:,1,2] = (y1 + y2 - height + 1) / (height - 1)
        return affine

    def get_patch_affine(self, img, patch_size, stride):
        img_h, img_w = img.shape[2:]
        patch_size_h = min(patch_size, img_h)
        patch_size_w = min(patch_size, img_w)
        img = torch.stack(torch.meshgrid(torch.arange(img_h), torch.arange(img_w)))[None,...]
        patch_num_h = (img_h - (patch_size_h - 1) - 1 + stride - 1) // stride + 1
        patch_num_w = (img_w - (patch_size_w - 1) - 1 + stride - 1) // stride + 1
        i = torch.arange(patch_num_h) * stride
        j = torch.arange(patch_num_w) * stride
        #print(patch_num_h, patch_num_w, img.shape)
        if (i[-1] + patch_size_h) > img_h:
            i[-1] = img_h - patch_size_h
        if (j[-1] + patch_size_w) > img_w:
            j[-1] = img_w - patch_size_w
        coor = torch.stack(torch.meshgrid(i, j)).reshape(2,-1)
        i = coor[0]
        j = coor[1]
        x1 = j
        x2 = j + patch_size_w
        y1 = i
        y2 = i + patch_size_h
        affine = torch.eye(3)[None,...].expand(i.shape[0],-1,-1).clone()
        affine[:,0,0] = (x2 - x1) / img_w
        affine[:,1,1] = (y2 - y1) / img_h
        affine[:,0,2] = (x1 + x2 - img_w) / img_w
        affine[:,1,2] = (y1 + y2 - img_h) / img_h
        return affine
        #img_patch = apply_affine(img.expand(affine.shape[0],-1,-1,-1).float(), affine = affine[:,:2],out_size = (patch_size, patch_size))
        #img_patch.reshape(patch_num_h, patch_num_w, 2, patch_size, patch_size)
        #print(img_patch.reshape(patch_num_h, patch_num_w, 2, patch_size, patch_size)[-1,0,0])
        #pdb.set_trace()

    def __call__(self, img, num_of_sample = 100):
        assert img.shape[0] == 1
        img_h, img_w = img.shape[2:]
        affine_list = []
        counter = 0
        while True:
            affine = self.get_params(img_h, img_w, self.scale, self.ratio, num_of_sample)
            affine_list.append(affine)
            if counter + affine.shape[0] >= num_of_sample:
                break
            counter += affine.shape[0]
        affine = torch.cat(affine_list)[:num_of_sample]
        flip_affine = torch.eye(3)[None,...].expand(affine.shape[0],-1,-1).clone()
        flip_affine[:,0,0] = (torch.rand(flip_affine.shape[0]) - 0.5).sign()
        affine = affine @ flip_affine
        return affine
        #affine = torch.from_numpy(affine)[:2][None,...]
        #out_img = []
        #for img in img_list:
        #    out_img.append(apply_affine(img[None,...], affine, out_size = self.size)[0])
        #return out_img


class IIW(data.Dataset):
    def __init__(self, root, img_transform = None, split = 'train'):
        #if split == 'train':
        #    img_index = np.load('iiw_train_ids.npy')
        #else:
        #    img_index = np.load('iiw_test_ids.npy')
        with open('val_list.txt') as f:
            val_img_list = [file_name.replace('.png\n','') for file_name in f.readlines()]
        img_index = [file_name.split('/')[-1].replace('.png', '') for file_name in glob.glob(root + '/*.png')]
        train_img_list = list(set(img_index) - set(val_img_list))
        if split == 'train':
            self.img_index = np.array(train_img_list)
        else:
            self.img_index = np.array(val_img_list)
        self.img_index.sort()
        self.root = root
        self.img_transform = img_transform
        self.split = split

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, index):
        img = Image.open(self.root + '/' + self.img_index[index] + '.png')
        img_h, img_w = np.array(img).shape[:2]
        img = self.img_transform(img)
        return img, np.array([img_h, img_w]), self.img_index[index]
    

class IIW2(data.Dataset):
    def __init__(self, root, img_transform = None, split = 'train', return_two_images = False):
        #if split == 'train':
        #    img_index = np.load('iiw_train_ids.npy')
        #else:
        #    img_index = np.load('iiw_test_ids.npy')
        with open('val_list.txt') as f:
            val_img_list = [file_name.replace('.png\n','') for file_name in f.readlines()]
        img_index = [file_name.split('/')[-1].replace('.png', '') for file_name in glob.glob(root + '/*.png')]
        train_img_list = list(set(img_index) - set(val_img_list))
        if split == 'train':
            self.img_index = np.array(train_img_list)
        else:
            self.img_index = np.array(val_img_list)
        self.img_index.sort()
        self.root = root
        self.img_transform = img_transform
        self.split = split
        self.return_two_images = return_two_images

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, index):
        if self.return_two_images:
            # Load first image
            img1 = Image.open(self.root + '/' + self.img_index[index] + '.png')
            
            # Select a different random image for the second one
            available_indices = list(range(len(self.img_index)))
            available_indices.remove(index)  # Remove current index to ensure different images
            second_index = np.random.choice(available_indices)
            img2 = Image.open(self.root + '/' + self.img_index[second_index] + '.png')
            
            # Apply transforms if provided
            if self.img_transform is not None:
                # Check if img_transform is a tuple/list (like in MIT_Dataset_PreLoad)
                if isinstance(self.img_transform, (tuple, list)) and len(self.img_transform) == 2:
                    single_img_transform = self.img_transform[1]
                    group_img_transform = self.img_transform[0]
                    
                    img1 = single_img_transform(img1)
                    img2 = single_img_transform(img2)
                    
                    if group_img_transform is not None:
                        img1, img2 = group_img_transform([img1, img2])
                else:
                    # Single transform for both images
                    img1 = self.img_transform(img1)
                    img2 = self.img_transform(img2)
            
            return img1, img2
        else:
            # Original behavior - return single image with metadata
            img = Image.open(self.root + '/' + self.img_index[index] + '.png')
            img_h, img_w = np.array(img).shape[:2]
            img = self.img_transform(img)
            return img, np.array([img_h, img_w]), self.img_index[index]

def compute_rank_split(data_list, total_split, split_id):
    num_of_data = len(data_list)
    num_per_split = (num_of_data + (total_split - 1)) // total_split
    split_data_list = data_list[num_per_split * split_id: num_per_split * (split_id + 1)]
    extra_list_num = int(np.ceil(num_per_split)) - len(split_data_list)
    if extra_list_num > 0:
        split_data_list = split_data_list + split_data_list[:extra_list_num]
    return split_data_list


def parallel_load_image(img_path_list):
    class loader(torch.utils.data.Dataset):
        def __init__(self, img_path_list):
            self.img_path_list = img_path_list

        def __len__(self):
            return len(self.img_path_list)

        def __getitem__(self, index):
            img_list = sorted(glob.glob(self.img_path_list[index] + '/*.jpg'))
            images = []
            for img_path in img_list:
                img = Image.open(img_path).convert('RGB') # Resize to 256x256 ABRIL
                img_resize = img.resize((128, 128)) # Resize to 256x256 ABRIL
                images.append(np.array(img_resize))# Resize to 256x256 ABRIL
                #images.append(np.array(Image.open(img_path).convert('RGB')))
            return np.stack(images)
        
    dataset = loader(img_path_list)
    def collate_fn(batch):
        return batch
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=None,
        batch_size=10,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    images_list = []
    for images in tqdm(data_loader):
        images_list += images
    return images_list

class MIT_Dataset_PreLoad(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier = 10,
                    total_split = 4, split_id = 0):
        img_list = compute_rank_split(sorted(glob.glob(root + '/*')), total_split, split_id)
        self.images = parallel_load_image(img_list)
        self.epoch_multiplier = epoch_multiplier
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        print('init', len(self.images), img_list[0])

    def __len__(self):
        return len(self.images) * self.epoch_multiplier * 25

    def __getitem__(self, index):
        index = index % (len(self.images) * 25)
        folder_idx = index // 25
        images = self.images[folder_idx]

        light_index = np.random.randint(25)
        img1 = Image.fromarray(images[light_index])
        img2 = Image.fromarray(images[(light_index + 1 + np.random.randint(24)) % 25])

        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)

        if self.group_img_transform is not None:
            img1, img2 = self.group_img_transform([img1, img2])
        return img1, img2

class MIT_Dataset(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier = 10,
                    eval_mode = False):
        self.epoch_multiplier = epoch_multiplier
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_folder_list = img_folder_list
        self.img_idx = np.arange(25).tolist()
        self.eval_mode = eval_mode
        self.eval_pair_light_shift = np.random.randint(1, 25, (len(img_folder_list) * 25))
        self.eval_pair_folder_shift = np.random.randint(1, len(img_folder_list), (len(img_folder_list) * 25))
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        print('init')

    def __len__(self):
        return len(self.img_folder_list) * self.epoch_multiplier * len(self.img_idx)

    def __getitem__(self, index):
        index = index % (len(self.img_folder_list) * len(self.img_idx))
        folder_idx = index // len(self.img_idx)
        folder_path = self.img_folder_list[folder_idx]
        if self.eval_mode:
            ref_folder_path = self.img_folder_list[(folder_idx + self.eval_pair_folder_shift[index]) % len(self.img_folder_list)]
        else:
            ref_folder_path = self.img_folder_list[np.random.randint(len(self.img_folder_list))]

        folder_offset = index % len(self.img_idx)
        if self.eval_mode:
            pair_img_folder_offset = (folder_offset + self.eval_pair_light_shift[index]) % len(self.img_idx)
        else:
            pair_img_folder_offset = np.random.choice(np.where(np.arange(len(self.img_idx)) != folder_offset)[0])
        folder_offset = self.img_idx[folder_offset]
        pair_img_folder_offset = self.img_idx[pair_img_folder_offset]

        img1 = Image.open(f'{folder_path}/dir_{folder_offset}_mip2.jpg')
        img2 = Image.open(f'{folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')
        img3 = Image.open(f'{ref_folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')

        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)
        img3 = self.single_img_transform(img3)

        if self.group_img_transform is not None:
            img1, img2, img3 = self.group_img_transform([img1, img2, img3])
        return img1, img3, img2

class MIT_Dataset_show(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier = 10,
                    eval_mode = False, return_seg_map = False, seg_transform = None,
                    eval_pair_folder_shift = 5,
                    eval_pair_light_shift = 5,
                    ):
        self.return_seg_map = return_seg_map
        self.seg_transform = seg_transform
        self.epoch_multiplier = epoch_multiplier
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_folder_list = img_folder_list
        #with open('train.txt' if train else 'val.txt', 'w') as f:
        #    for name in sub_img_folder_list:
        #        f.write(name.split('/')[-1] + '\n')
        #pdb.set_trace()
        train_idx = [idx for idx in np.arange(25).tolist()]
        #self.img_idx = train_idx
        #self.img_idx = train_idx if train else test_idx
        self.img_idx = train_idx
        self.eval_mode = eval_mode
        self.eval_pair_light_shift = np.random.randint(1, 25, (len(img_folder_list) * 25))
        self.eval_pair_folder_shift = np.random.randint(1, len(img_folder_list), (len(img_folder_list) * 25))
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        print('init')

    def __len__(self):
        return len(self.img_folder_list) * self.epoch_multiplier * len(self.img_idx)

    def get_img(self, img_folder_idx, img_idx):
        folder_path = self.img_folder_list[img_folder_idx]
        img = Image.open(f'{folder_path}/dir_{img_idx}_mip2.jpg')
        img = self.single_img_transform(self.group_img_transform(img))

        json_data = json.load(open(f'{folder_path}/meta.json'))
        mask = np.zeros((1000, 1500))
        bbox = json_data['chrome']['bounding_box']
        bbox = np.array([bbox['x']/ 4, bbox['y'] / 4, bbox['w'] / 4, bbox['h'] / 4]).astype(np.int32)
        mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 1
        coor_y, coor_x= np.where(np.array(self.group_img_transform(Image.fromarray(mask))) > 0)
        box_y_min = coor_y.min()
        box_y_max = coor_y.max()
        box_x_min = coor_x.min()
        box_x_max = coor_x.max()
        box_w = box_x_max - box_x_min + 1
        box_h = box_y_max - box_y_min + 1
        box_size = max(box_h, box_w)
        def compute_new_coor(box_size, img_size, coor_min, coor_max):
            aug_size = box_size - (coor_max - coor_min + 1)
            aug_edge = np.ceil(aug_size * 0.5)
            new_coor_min = coor_min - aug_edge
            new_coor_max = coor_max + aug_edge
            if new_coor_min < 0:
                new_coor_max += (-1 * new_coor_min)
                new_coor_min = 0
            if new_coor_max >= img_size:
                new_coor_min -= (new_coor_max - img_size + 1)
                new_coor_max = img_size - 1
            return int(new_coor_min), int(new_coor_max)
        box_x_min, box_x_max = compute_new_coor(box_size, 256, box_x_min, box_x_max)
        box_y_min, box_y_max = compute_new_coor(box_size, 256, box_y_min, box_y_max)
        img_box = img[:,box_y_min:box_y_max + 1, box_x_min:box_x_max+1]
        return img, F.interpolate(img_box[None,:], size = (256, 256), mode = 'bilinear')[0], [box_y_min, box_y_max, box_x_min, box_x_max]

    def __getitem__(self, index):
        index = index % (len(self.img_folder_list) * len(self.img_idx))
        folder_idx = index // len(self.img_idx)
        folder_path = self.img_folder_list[folder_idx]
        #if self.eval_mode:
        #    ref_folder_path = self.img_folder_list[(folder_idx + self.eval_pair_folder_shift[index]) % len(self.img_folder_list)]
        #else:
        ref_folder_idx = (folder_idx + np.random.randint(len(self.img_folder_list) - 1) + 1) % len(self.img_folder_list)
        ref_folder_path = self.img_folder_list[ref_folder_idx]

        folder_offset = index % len(self.img_idx)
        #if self.eval_mode:
        #    pair_img_folder_offset = (folder_offset + self.eval_pair_light_shift[index]) % len(self.img_idx)
        #else:
        pair_img_folder_offset = np.random.choice(np.where(np.arange(len(self.img_idx)) != folder_offset)[0])
        folder_offset = self.img_idx[folder_offset]
        pair_img_folder_offset = self.img_idx[pair_img_folder_offset]

        folder_name = folder_path.split('/')[-1]

        img1 = Image.open(f'{folder_path}/dir_{folder_offset}_mip2.jpg')
        img2 = Image.open(f'{folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')
        img3 = Image.open(f'{ref_folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')

        img1 = self.single_img_transform(self.group_img_transform(img1))
        img2 = self.single_img_transform(self.group_img_transform(img2))
        img3 = self.single_img_transform(self.group_img_transform(img3))

        return img1, img2, img3, folder_offset, pair_img_folder_offset, folder_idx, ref_folder_idx

class MIT_Dataset_Normal(data.Dataset):
    def __init__(self, root,  group_img_transform = None, epoch_multiplier = 1):
        self.group_img_transform = group_img_transform
        self.surface_normal = '/net/projects/willettlab/roxie62/dataset/mit_multiview_normal_omi'
        self.epoch_multiplier = epoch_multiplier
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_folder_list = img_folder_list
        #with open('train.txt' if train else 'val.txt', 'w') as f:
        #    for name in sub_img_folder_list:
        #        f.write(name.split('/')[-1] + '\n')
        #pdb.set_trace()
        train_idx = [idx for idx in np.arange(25).tolist()]
        self.img_idx = train_idx
        #self.img_idx = train_idx if train else test_idx
        print('init')

    def __len__(self):
        return len(self.img_folder_list) * self.epoch_multiplier * len(self.img_idx)

    def __getitem__(self, index):
        index = index % (len(self.img_folder_list) * len(self.img_idx))
        folder_idx = index // len(self.img_idx)
        folder_path = self.img_folder_list[folder_idx]
        folder_offset = index % len(self.img_idx)

        folder_name = folder_path.split('/')[-1]

        img = Image.open(f'{folder_path}/dir_{folder_offset}_mip2.jpg')
        #img = surface_normal_img_transform(img)
        normal = torch.from_numpy(np.array(Image.open(self.surface_normal + f'/{folder_name}.png'))).permute(2,0,1)[None,...].float()
        normal = normal / 255 * 2 - 1
        #normal = np.load(self.surface_normal + f'/{folder_name}.npy')
        #img_pil = img.resize((normal.shape[1], normal.shape[1]))
#
        #fig, ax = plt.subplots(2)
        #ax[0].imshow(img_pil)
        #ax[1].imshow(((normal * 0.5 + 0.5) * 255).astype(np.uint8))
        #plt.savefig('img.png')
        #pdb.set_trace()

        img = np.array(img) * 1.0 / 255 * 2 - 1
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)[None,...].float()
        #normal = torch.from_numpy(normal).permute(2,0,1)[None,...]
        normal = F.normalize(torchvision.transforms.Resize(256)(normal), dim = 1)[0]
        img = torchvision.transforms.CenterCrop(256)(torchvision.transforms.Resize(256)(img))[0]
        if self.group_img_transform is not None:
            img, normal = self.group_img_transform([img, normal])
        #transforms.ToPILImage()(normal * 0.5 + 0.5).save('img1.png')
        #transforms.ToPILImage()(img * 0.5 + 0.5).save('img2.png')
        return img, normal, folder_idx, folder_offset

#train_dataset = MIT_Dataset_Normal('/net/projects/willettlab/roxie62/dataset/mit_multiview_resize', 5, [0,1,2,3], train = True)
#train_dataset[0]
def init_ema_model(model, ema_model):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()
    ):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    for param_q, param_k in zip(
        model.buffers(), ema_model.buffers()
    ):
        param_k.data.copy_(param_q.data)  # initialize

def update_ema_model(model, ema_model, m):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.raw_val = []

    def reset(self):
        self.raw_val = []

    def update(self, val):
        self.val = val
        self.raw_val.append(val)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        data = np.array(self.raw_val)
        return fmtstr.format(name = self.name, val = self.val, avg = data.mean())


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'





# Validation utils abril
def create_validation_datasets(args, transform_val):
    """Create validation datasets for multiple datasets"""
    validation_datasets = {}
    
    # Get validation datasets to create (default to MIT and RSR_256)
    val_datasets = getattr(args, 'validation_datasets', ['mit', 'rsr_256', 'rlsid']) #, 
    
    for dataset_name in val_datasets:
        print(f"Creating validation dataset for: {dataset_name}")
        
        if dataset_name == 'mit':
            val_dataset = MIT_Dataset('/home/apinyol/TFM/Data/multi_illumination_test_mip2_jpg', 
                                     transform_val, eval_mode=True)
            validation_datasets['MIT'] = val_dataset
            
        elif dataset_name == 'rsr_256':
            # Use RSR validation path - you might need to adjust this path
            rsr_val_path = getattr(args, 'rsr_val_path', '/home/apinyol/TFM/Data/RSR_256')
            val_dataset = RSRDataset(root_dir=rsr_val_path, 
                                   img_transform=transform_val, 
                                   is_validation=True, validation_scenes=["scene_01", "scene_03", "scene_05", "scene_06", "scene_07", "scene_08", "scene_10", "scene_21", "scene_22", "scene_23", "scene_24", "scene_26", "scene_27", "scene_29", "scene_30"])
            validation_datasets['RSR_256'] = val_dataset
            
        elif dataset_name == 'iiw':
            # Add IIW validation if needed
            iiw_val_path = getattr(args, 'iiw_val_path', '/home/apinyol/TFM/Data/iiw-dataset/data')
            val_dataset = IIW2(root=iiw_val_path,
                              img_transform=transform_val,
                              split='val',
                              return_two_images=True)
            validation_datasets['IIW'] = val_dataset

        elif dataset_name == 'rlsid':
            # Add RLSID validation if needed
            rlsid_val_path = getattr(args, 'rlsid_val_path', '/data/storage/datasets/RLSID')
            """val_dataset = RLSIDDataset(root_dir=args.data_path,
                    validation_scenes=['00012', '00014', '00020', '00022', '00029', '00030', '00036', '00042', '00043', '00049', '00054', '00058', '00060', '00070', '00076', '00081', '00087', '00098', '00103', '00112', '00113', '00115', '00121', '00124', '00131', '00137', '00138', '00145', '00156', '00157', '00164', '00175', '00176', '00177', '00178', '00179', '00181', '00190', '00198', '00200', '00205', '00206', '00209', '00217', '00219', '00230', '00240', '00244', '00246', '00247', '00255', '00260', '00262', '00276', '00281', '00282', '00283', '00286', '00291', '00306', '00312', '00317', '00320', '00321', '00325', '00332', '00334', '00341', '00343', '00344', '00346', '00349', '00352', '00356', '00361', '00364', '00369', '00371', '00372', '00373', '00380', '00384', '00385', '00392', '00400', '00402', '00411', '00413', '00414', '00419', '00422', '00424', '00428', '00431', '00435', '00437', '00442', '00443', '00448', '00457', '00459', '00461', '00466', '00476', '00477', '00491', '00494', '00495', '00498', '00505', '00506', '00515', '00527', '00529', '00533', '00542', '00552', '00556', '00558', '00566', '00569', '00573', '00576', '00577', '00581', '00586', '00587', '00594', '00600', '00603', '00604', '00605', '00608', '00619', '00636', '00648', '00663', '00669', '00673', '00678', '00687', '00697', '00702', '00704', '00707', '00713', '00726', '00727', '00731', '00736', '00743', '00744', '00749', '00760', '00768', '00772', '00776', '00777', '00785', '00788', '00790', '00792', '00794', '00796', '00802', '00806', '00820', '00821', '00824', '00828', '00832', '00847', '00851', '00852', '00855', '00857', '00867', '00871', '00879', '00880', '00883', '00900', '00902', '00904', '00905', '00909', '00921', '00923', '00931', '00935', '00942', '00946', '00961', '00971', '00981', '00986', '00988', '01000', '01006', '01009', '01014', '01018', '01020', '01024', '01031', '01035', '01036', '01038', '01040', '01041', '01044', '01059', '01060', '01064', '01065', '01066', '01067', '01069', '01076', '01085', '01086', '01088', '01090', '01095', '01097', '01113', '01115', '01124', '01139', '01148', '01150', '01152', '01154', '01155', '01158', '01163', '01166', '01168', '01178', '01182', '01191', '01197', '01200', '01201', '01207', '01210', '01211', '01213', '01214', '01218', '01221', '01223', '01229', '01233', '01236', '01244', '01250', '01253', '01258', '01263', '01270', '01278', '01293', '01295', '01301', '01311', '01314', '01323', '01329', '01338', '01339', '01342', '01343', '01348', '01356', '01363', '01367', '01370', '01384', '01393', '01398', '01404', '01405', '01410', '01416', '01419', '01423', '01424', '01426', '01431', '01433', '01438', '01444', '01451', '01460', '01463', '01466', '01468', '01469', '01479', '01481', '01485', '01487', '01492', '01498', '01505', '01519', '01520', '01525', '01526', '01537', '01547', '01561', '01566', '01570', '01577', '01579', '01582', '01583', '01585', '01588', '01589', '01593', '01594', '01599', '01603', '01604', '01605', '01606', '01610', '01611', '01620', '01622', '01627', '01637', '01646', '01649', '01650', '01652', '01657', '01658', '01668', '01670', '01672', '01678', '01680', '01681', '01686', '01694', '01710', '01711', '01716', '01721', '01726', '01732', '01733', '01735', '01737', '01740', '01744', '01748', '01754', '01766', '01767', '01773', '01779', '01781', '01786', '01790', '01793', '01796', '01804', '01806', '01807', '01809', '01811', '01821', '01831', '01836', '01842', '01845', '01851', '01852', '01858', '01865', '01877', '01887', '01895', '01896', '01901', '01903', '01906', '01912', '01922', '01924', '01929', '01931', '01932', '01935', '01940', '01941', '01946', '01964', '01966', '01969', '01976', '01981', '01985', '01987', '01990', '01993', '01995', '02007', '02047', '02051', '02052', '02057', '02062', '02069', '02073', '02076', '02081', '02083', '02100', '02101', '02102', '02110', '02119', '02125', '02129', '02130', '02131', '02135', '02136', '02145', '02150', '02163', '02167', '02189', '02202', '02216', '02219', '02229', '02230', '02233', '02234', '02235', '02236', '02247', '02256', '02258', '02261', '02263', '02265', '02266', '02267', '02273', '02278', '02287', '02288', '02291', '02307', '02311', '02322', '02335', '02343', '02352', '02355', '02356', '02359', '02373', '02374', '02380', '02385', '02387', '02392', '02396', '02400', '02407', '02416', '02417', '02418', '02424', '02425', '02429', '02441', '02452', '02456', '02491', '02501', '02503', '02513', '02515', '02517', '02518', '02520', '02526', '02530', '02537', '02539', '02542', '02543', '02548', '02550', '02561', '02562', '02569', '02580', '02581', '02584', '02591', '02594', '02604', '02605', '02607', '02624', '02631', '02632', '02634', '02642', '02645', '02653', '02660', '02661', '02669', '02670', '02672', '02676', '02678', '02684', '02694', '02696', '02697', '02701', '02709', '02711', '02713', '02734', '02736', '02737', '02743', '02752', '02767', '02770', '02776', '02779', '02783', '02784', '02790', '02791', '02793', '02797', '02808', '02810', '02814', '02820', '02826', '02828', '02829', '02833', '02834', '02839', '02852', '02854', '02856', '02858', '02859', '02860', '02863', '02865', '02870', '02874', '02886', '02888', '02894', '02900', '02905', '02906', '02911', '02913', '02917', '02922', '02928', '02942', '02945', '02946', '02952', '02958', '02973', '02975', '02976', '02977', '02982', '02995', '02996', '02997', '03001', '03006', '03013', '03015', '03018', '03024', '03038', '03048', '03056', '03057', '03061', '03071', '03072', '03078', '03084', '03087', '03099', '03101', '03116', '03118', '03120', '03128', '03133', '03143', '03145', '03149', '03150', '03152', '03154', '03157', '03158', '03161', '03162', '03164', '03169', '03170', '03172', '03184', '03191', '03192', '03207', '03208', '03213', '03219', '03220', '03237', '03241', '03247', '03251', '03256', '03258', '03271', '03273', '03283', '03288', '03290', '03297', '03302', '03307', '03308', '03333', '03340', '03343', '03345', '03352', '03362', '03366', '03368', '03379', '03391', '03395', '03397', '03398', '03416', '03419', '03430', '03431', '03440', '03447', '03450', '03459', '03467', '03469', '03489', '03491', '03492', '03501', '03503', '03507', '03511', '03518', '03519', '03520', '03539', '03547', '03556', '03569', '03571', '03574', '03595', '03604', '03617', '03621', '03631', '03638', '03640', '03647', '03648', '03656', '03662', '03668', '03670', '03672', '03675', '03680', '03682', '03690', '03691', '03693', '03694', '03706', '03713', '03716', '03717', '03724', '03731', '03733', '03734', '03747', '03758', '03781', '03782', '03791', '03794', '03795', '03801', '03802', '03813', '03815', '03817', '03821', '03826', '03830', '03832', '03856', '03861', '03864', '03867', '03873', '03877', '03883', '03889', '03891', '03892', '03895', '03896', '03900', '03901', '03904', '03909', '03912', '03915', '03920', '03925', '03944', '03948', '03951', '03956', '03960', '03967', '03968', '03970', '03981', '03986', '03997', '03998', '04002', '04004', '04016', '04019', '04022', '04023', '04027', '04032', '04041', '04046', '04049', '04050', '04051', '04054', '04055', '04059', '04060', '04064', '04072', '04074', '04075', '04077', '04082', '04087', '04096', '04097', '04104', '04105', '04115', '04116', '04117', '04120', '04122', '04128', '04130', '04140', '04142', '04146', '04149', '04152', '04156', '04165', '04170', '04175', '04180', '04191', '04192', '04202', '04212', '04213', '04222', '04226', '04232', '04235', '04237', '04254', '04258', '04259', '04267', '04279', '04283', '04287', '04289', '04290', '04291', '04295', '04298', '04313', '04317', '04319', '04326', '04328', '04329', '04363', '04364', '04365', '04368', '04372', '04373', '04376', '04386', '04387', '04389', '04390', '04396', '04398', '04402', '04405', '04408', '04417', '04438', '04439', '04446', '04456', '04460', '04464', '04466', '04467', '04474', '04476', '04489', '04490', '04494', '04498', '04507', '04518', '04519', '04520', '04522', '04536', '04546', '04550', '04560', '04581', '04586', '04589', '04597', '04598', '04601', '04604', '04607', '04608', '04616', '04622', '04624', '04626', '04637', '04641', '04644', '04645', '04650', '04656', '04660', '04669', '04672', '04676', '04678', '04681', '04685', '04689', '04690', '04693', '04694', '04698', '04706', '04707', '04715', '04728', '04729', '04742', '04744', '04751', '04752', '04753', '04756', '04758', '04760', '04767', '04794', '04798', '04800', '04801', '04803', '04813', '04815', '04818', '04825', '04827', '04832', '04842', '04861', '04862', '04875', '04881', '04889', '04890', '04895', '04902', '04923', '04934', '04943', '04950', '04955', '04961', '04972', '04977', '04978', '04980', '04987', '04995', '04998', '05005', '05006', '05013', '05016', '05019', '05025', '05027', '05029', '05036', '05037', '05042', '05046', '05056', '05062', '05063', '05066', '05068', '05081', '05089', '05091', '05092', '05093', '05095', '05097', '05099', '05103', '05108', '05111', '05112', '05116', '05117', '05124', '05125', '05127', '05128', '05135', '05136', '05138', '05139', '05144', '05147', '05148', '17219', '17235', '17239', '17245', '17250', '17258', '17263', '17275', '17276', '17278', '17286', '17287', '17291', '17293', '17294', '17297', '17307', '17309', '17313', '17318', '17324', '17329', '17333', '17334', '17341', '17347', '17359', '17362', '17364', '17367', '17371', '17378', '17385', '17386', '17389', '17402', '17417', '17418', '17421', '17425', '17439', '17441', '17446', '17447', '17455', '17457', '17462', '17466', '17475', '17477', '17483', '17485', '17497', '17506', '17514', '17518', '17521', '17527', '17528', '17538', '17539', '17541', '17546', '17547', '17550', '17554', '17562', '17564', '17565', '17566', '17580', '17583', '17597', '17599', '17601', '17602', '17603', '17606', '17611', '17612', '17616', '17621', '17623', '17635', '17637', '17656', '17662', '17665', '17666', '17674', '17681', '17682', '17683', '17690', '17693', '17696', '17704', '17705', '17719', '17722', '17727', '17729', '17730', '17741', '17745', '17746', '17747', '17751', '17753', '17756', '17757', '17761', '17763', '17765', '17768', '17770', '17771', '17784', '17786', '17788', '17800', '17801', '17805', '17806', '17807', '17809', '17815', '17821', '17828', '17829', '17831', '17836', '17837', '17851', '17855', '17863', '17865', '17873', '17876', '17877', '17883', '17884', '17893', '17894', '17897', '17906', '17908', '17911', '17919', '17932', '17933', '17937', '17946', '17947', '17949', '17953', '17957', '17962', '17968', '17974', '17987', '17990', '17994', '17995', '18002', '18007', '18008', '18011', '18018', '18022', '18026', '18028', '18033', '18037', '18041', '18043', '18045', '18048', '18055', '18063', '18075', '18078', '18080', '18082', '18092', '18094', '18097', '18104', '18106', '18107', '18110', '18111', '18112', '18126', '18128', '18152', '18159', '18162', '18171', '18172', '18178', '18179', '18180', '18181', '18195', '18197', '18206', '18208', '18209', '18211', '18221', '18228', '18232', '18237', '18238', '18239', '18241', '18253', '18270', '18280', '18288', '18292', '18294', '18299', '18301', '18302', '18304', '18312', '18319', '18320', '18323', '18327', '18335', '18337', '18342', '18353', '18354', '18357', '18358', '18360', '18364', '18365', '18375', '18377', '18381', '18382', '18383', '18393', '18394', '18401', '18405', '18414', '18417', '18419', '18431', '18434', '18436', '18441', '18446', '18447', '18451', '18453', '18459', '18460', '18462', '18466', '18468', '18477', '18482', '18490', '18492', '18493', '18496', '18499', '18501', '18511', '18518', '18532', '18542', '18543', '18546', '18551', '18556', '18561', '18566', '18570', '18573', '18582', '18587', '18596', '18601', '18602', '18616', '18617', '18620', '18632', '18648', '18653', '18655', '18660', '18661', '18665', '18676', '18680', '18681', '18683', '18684', '18692', '18694', '18698', '18702', '18710', '18712', '18716', '18718', '18725', '18734', '18739', '18741', '18743', '18754', '18756', '18759', '18761', '18771', '18777', '18778', '18780', '18783', '18786', '18796', '18797', '18802', '18813', '18814', '18815', '18831', '18835', '18836', '18841', '18849', '18850', '18853', '18857', '18859', '18865', '18872', '18878', '18881', '18887', '18892', '18900', '18911', '18912', '18916', '18922', '18928', '18940', '18949', '18959', '18960', '18962', '18981', '18986', '18988', '18989', '18994', '19001', '19008', '19017', '19026', '19034', '19038', '19040', '19045', '19046', '19047', '19050', '19052', '19054', '19058', '19059', '19060', '19066', '19071', '19074', '19077', '19079', '19088', '19096', '19098', '19101', '19102', '19108', '19111', '19114', '19120', '19121', '19125', '19128', '19130', '19136', '19147', '19151', '19154', '19155', '19165', '19168', '19170', '19181', '19189', '19194', '19196', '19202', '19204', '19207', '19211', '19214', '19220', '19225', '19229', '19235', '19237', '19240', '19251', '19257', '19259', '19266', '19267', '19268', '19274', '19277', '19285', '19297', '19298', '19304', '19309', '19313', '19320', '19324', '19326', '19327', '19331', '19335', '19345', '19347', '19348', '19357', '19359', '19364', '19370', '19373', '19374', '19375', '19377', '19383', '19384', '19386', '19388', '19398', '19400', '19405', '19408', '19409', '19419', '19421', '19423', '19426', '19428', '19431', '19432', '19437', '19440', '19441', '19443', '19448', '19451', '19452', '19455', '19459', '19460', '19464', '19467', '19475', '19480', '19483', '19487', '19496', '19498', '19512', '19520', '19528', '19532', '19534', '19535', '19542', '19550', '19555', '19561', '19572', '19573', '19579', '19581', '19586', '19587', '19591', '19604', '19606', '19636', '19650', '19652', '19654', '19655', '19658', '19660', '19665', '19673', '19675', '19690', '19708', '19711', '19718', '19729', '19733', '19745', '19749', '19755', '19756', '19757', '19764', '19772', '19776', '19778', '19786', '19787', '19789', '19790', '19792', '19793', '19796', '19800', '19801', '19805', '19820', '19823', '19829', '19840', '19845', '19857', '19866', '19871', '19879', '19881', '19890', '19904', '19913', '19918', '19923', '19931', '19932', '19936', '19937', '19941', '19945', '19956', '19962', '19965', '19966', '19976', '19978', '19979', '19984', '19989', '19993', '19994', '19998', '20002', '20003', '20007', '20011', '20012', '20014', '20018', '20019', '20020', '20032', '20034', '20037', '20039', '20040', '20041', '20052', '20065', '20074', '20084', '20085', '20086', '20087', '20088', '20098', '20100', '20117', '20118', '20129', '20130', '20144', '20161', '20166', '20170', '20179', '20183', '20186', '20188', '20189', '20215', '20237', '20251', '20253', '20255', '20257', '20265', '20266', '20269', '20274', '20275', '20276', '20280', '20283', '20284', '20285', '20286', '20289', '20290', '20303', '20307', '20313', '20329', '20339', '20342', '20348', '20352', '20366', '20367', '20374', '20375', '20376', '20382', '20389', '20391', '20393', '20396', '20399', '20402', '20403', '20405', '20414', '20416', '20424', '20435', '20438', '20441', '20449', '20452', '20461', '20463', '20465', '20468', '20472', '20477', '20483', '20490', '20494', '20495', '20504', '20511', '20515', '20516', '20518', '20520', '20523', '20526', '20529', '20530', '20535', '20541', '20561', '20566', '20569', '20573', '20579', '20595', '20599', '20605', '20606', '20608', '20614', '20618', '20621', '20629', '20631', '20643', '20644', '20662', '20669', '20677', '20679', '20682', '20688', '20691', '20692', '20701', '20709', '20711', '20713', '20721', '20724', '20729', '20737', '20739', '20740', '20741', '20743', '20755', '20768', '20776', '20782', '20787', '20793', '20800', '20803', '20804', '20813', '20816', '20817', '20820', '20822', '20829', '20835', '20837', '20838', '20846', '20849', '20858', '20866', '20874', '20888', '20889', '20892', '20898', '20919', '20921', '20923', '20929', '20933', '20936', '20943', '20944', '20946', '20951', '20958', '20962', '20964', '20966', '20976', '20977', '20988', '20996', '21010', '21011', '21012', '21026', '21036', '21042', '21051', '21053', '21054', '21060', '21065', '21068', '21078', '21080', '21082', '21083', '21096', '21102', '21106', '21112', '21115', '21116', '21131', '21133', '21138', '21149', '21155', '21158', '21162', '21164', '21172', '21175', '21193', '21202', '21203', '21213', '21214', '21218', '21219', '21223', '21224', '21233', '21235', '21239', '21244', '21247', '21251', '21253', '21256', '21260', '21265', '21270', '21273', '21276', '21277', '21284', '21286', '21289', '21293', '21294', '21304', '21310', '21313', '21314', '21318', '21333', '21340', '21348', '21353', '21354', '21363', '21369', '21373', '21380', '21381', '21383', '21387', '21392', '21394', '21403', '21406', '21407', '21411', '21412', '21425', '21426', '21428', '21435', '21439', '21442', '21444', '21447', '21449', '21455', '21457', '21468', '21471', '21483', '21485', '21488', '21489', '21495', '21496', '21497', '21504', '21505', '21506', '21507', '21510', '21512', '21516', '21518', '21519', '21521', '21530', '21537', '21540', '21542', '21544', '21550', '21561', '21565', '21570', '21572', '21577', '21579', '21590', '21594', '21595', '21597', '21603', '21605', '21607', '21613', '21614', '21623', '21626', '21632', '21639', '21640', '21644', '21645', '21648', '21649', '21652', '21655', '21659', '21663', '21664', '21666', '21673', '21674', '21683', '21686', '21690', '21695', '21702', '21703', '21705', '21709', '21713', '21717', '21724', '21728', '21742', '21743', '21752', '21758', '21770', '21772', '21783', '21784', '21793', '21805', '21809', '21821', '21827', '21838', '21843', '21844', '21846', '21851', '21872', '21875', '21877', '21878', '21887', '21891', '21896', '21903', '21907', '21915', '21916', '21919', '21920', '21922', '21929', '21936', '21937', '21946', '21947', '21949', '21951', '21954', '21955', '21957', '21958', '21960', '21964', '21968', '21971', '21973', '21979', '21981', '21986', '21988', '21989', '21993', '21998', '22004', '22005', '22006', '22010', '22045', '22048', '22056', '22062', '22063', '22069', '22074', '22079', '22095', '22097', '22100', '22105', '22106', '22111', '22122', '22123', '22124', '22127', '22128', '22129'],
                    img_transform=transform_val,
                    is_validation=True,
                    )"""
            
            val_dataset = RLSIDDataset(
                    root_dir='/data/storage/datasets/RLSID',
                    validation_scenes=['00012', '00014', '00020', '00022', '00029', '00030', '00036', '00042', '00043', '00049', '00054', '00058', '00060', '00070', '00076', '00081', '00087', '00098', '00103', '00112', '00113', '00115', '00121', '00124', '00131', '00137', '00138', '00145', '00156', '00157', '00164', '00175', '00176', '00177', '00178', '00179', '00181', '00190', '00198', '00200', '00205', '00206', '00209', '00217', '00219', '00230', '00240', '00244', '00246', '00247', '00255', '00260', '00262', '00276', '00281', '00282', '00283', '00286', '00291', '00306', '00312', '00317', '00320', '00321', '00325', '00332', '00334', '00341', '00343', '00344', '00346', '00349', '00352', '00356', '00361', '00364', '00369', '00371', '00372', '00373', '00380', '00384', '00385', '00392', '00400', '00402', '00411', '00413', '00414', '00419', '00422', '00424', '00428', '00431', '00435', '00437', '00442', '00443', '00448', '00457', '00459', '00461', '00466', '00476', '00477', '00491', '00494', '00495', '00498', '00505', '00506', '00515', '00527', '00529', '00533', '00542', '00552', '00556', '00558', '00566', '00569', '00573', '00576', '00577', '00581', '00586', '00587', '00594', '00600', '00603', '00604', '00605', '00608', '00619', '00636', '00648', '00663', '00669', '00673', '00678', '00687', '00697', '00702', '00704', '00707', '00713', '00726', '00727', '00731', '00736', '00743', '00744', '00749', '00760', '00768', '00772', '00776', '00777', '00785', '00788', '00790', '00792', '00794', '00796', '00802', '00806', '00820', '00821', '00824', '00828', '00832', '00847', '00851', '00852', '00855', '00857', '00867', '00871', '00879', '00880', '00883', '00900', '00902', '00904', '00905', '00909', '00921', '00923', '00931', '00935', '00942', '00946', '00961', '00971', '00981', '00986', '00988', '01000', '01006', '01009', '01014', '01018', '01020', '01024', '01031', '01035', '01036', '01038', '01040', '01041', '01044', '01059', '01060', '01064', '01065', '01066', '01067', '01069', '01076', '01085', '01086', '01088', '01090', '01095', '01097', '01113', '01115', '01124', '01139', '01148', '01150', '01152', '01154', '01155', '01158', '01163', '01166', '01168', '01178', '01182', '01191', '01197', '01200', '01201', '01207', '01210', '01211', '01213', '01214', '01218', '01221', '01223', '01229', '01233', '01236', '01244', '01250', '01253', '01258', '01263', '01270', '01278', '01293', '01295', '01301', '01311', '01314', '01323', '01329', '01338', '01339', '01342', '01343', '01348', '01356', '01363', '01367', '01370', '01384', '01393', '01398', '01404', '01405', '01410', '01416', '01419', '01423', '01424', '01426', '01431', '01433', '01438', '01444', '01451', '01460', '01463', '01466', '01468', '01469', '01479', '01481', '01485', '01487', '01492', '01498', '01505', '01519', '01520', '01525', '01526', '01537', '01547', '01561', '01566', '01570', '01577', '01579', '01582', '01583', '01585', '01588', '01589', '01593', '01594', '01599', '01603', '01604', '01605', '01606', '01610', '01611', '01620', '01622', '01627', '01637', '01646', '01649', '01650', '01652', '01657', '01658', '01668', '01670', '01672', '01678', '01680', '01681', '01686', '01694', '01710', '01711', '01716', '01721', '01726', '01732', '01733', '01735', '01737', '01740', '01744', '01748', '01754', '01766', '01767', '01773', '01779', '01781', '01786', '01790', '01793', '01796', '01804', '01806', '01807', '01809', '01811', '01821', '01831', '01836', '01842', '01845', '01851', '01852', '01858', '01865', '01877', '01887', '01895', '01896', '01901', '01903', '01906', '01912', '01922', '01924', '01929', '01931', '01932', '01935', '01940', '01941', '01946', '01964', '01966', '01969', '01976', '01981', '01985', '01987', '01990', '01993', '01995', '02007', '02047', '02051', '02052', '02057', '02062', '02069', '02073', '02076', '02081', '02083', '02100', '02101', '02102', '02110', '02119', '02125', '02129', '02130', '02131', '02135', '02136', '02145', '02150', '02163', '02167', '02189', '02202', '02216', '02219', '02229', '02230', '02233', '02234', '02235', '02236', '02247', '02256', '02258', '02261', '02263', '02265', '02266', '02267', '02273', '02278', '02287', '02288', '02291', '02307', '02311', '02322', '02335', '02343', '02352', '02355', '02356', '02359', '02373', '02374', '02380', '02385', '02387', '02392', '02396', '02400', '02407', '02416', '02417', '02418', '02424', '02425', '02429', '02441', '02452', '02456', '02491', '02501', '02503', '02513', '02515', '02517', '02518', '02520', '02526', '02530', '02537', '02539', '02542', '02543', '02548', '02550', '02561', '02562', '02569', '02580', '02581', '02584', '02591', '02594', '02604', '02605', '02607', '02624', '02631', '02632', '02634', '02642', '02645', '02653', '02660', '02661', '02669', '02670', '02672', '02676', '02678', '02684', '02694', '02696', '02697', '02701', '02709', '02711', '02713', '02734', '02736', '02737', '02743', '02752', '02767', '02770', '02776', '02779', '02783', '02784', '02790', '02791', '02793', '02797', '02808', '02810', '02814', '02820', '02826', '02828', '02829', '02833', '02834', '02839', '02852', '02854', '02856', '02858', '02859', '02860', '02863', '02865', '02870', '02874', '02886', '02888', '02894', '02900', '02905', '02906', '02911', '02913', '02917', '02922', '02928', '02942', '02945', '02946', '02952', '02958', '02973', '02975', '02976', '02977', '02982', '02995', '02996', '02997', '03001', '03006', '03013', '03015', '03018', '03024', '03038', '03048', '03056', '03057', '03061', '03071', '03072', '03078', '03084', '03087', '03099', '03101', '03116', '03118', '03120', '03128', '03133', '03143', '03145', '03149', '03150', '03152', '03154', '03157', '03158', '03161', '03162', '03164', '03169', '03170', '03172', '03184', '03191', '03192', '03207', '03208', '03213', '03219', '03220', '03237', '03241', '03247', '03251', '03256', '03258', '03271', '03273', '03283', '03288', '03290', '03297', '03302', '03307', '03308', '03333', '03340', '03343', '03345', '03352', '03362', '03366', '03368', '03379', '03391', '03395', '03397', '03398', '03416', '03419', '03430', '03431', '03440', '03447', '03450', '03459', '03467', '03469', '03489', '03491', '03492', '03501', '03503', '03507', '03511', '03518', '03519', '03520', '03539', '03547', '03556', '03569', '03571', '03574', '03595', '03604', '03617', '03621', '03631', '03638', '03640', '03647', '03648', '03656', '03662', '03668', '03670', '03672', '03675', '03680', '03682', '03690', '03691', '03693', '03694', '03706', '03713', '03716', '03717', '03724', '03731', '03733', '03734', '03747', '03758', '03781', '03782', '03791', '03794', '03795', '03801', '03802', '03813', '03815', '03817', '03821', '03826', '03830', '03832', '03856', '03861', '03864', '03867', '03873', '03877', '03883', '03889', '03891', '03892', '03895', '03896', '03900', '03901', '03904', '03909', '03912', '03915', '03920', '03925', '03944', '03948', '03951', '03956', '03960', '03967', '03968', '03970', '03981', '03986', '03997', '03998', '04002', '04004', '04016', '04019', '04022', '04023', '04027', '04032', '04041', '04046', '04049', '04050', '04051', '04054', '04055', '04059', '04060', '04064', '04072', '04074', '04075', '04077', '04082', '04087', '04096', '04097', '04104', '04105', '04115', '04116', '04117', '04120', '04122', '04128', '04130', '04140', '04142', '04146', '04149', '04152', '04156', '04165', '04170', '04175', '04180', '04191', '04192', '04202', '04212', '04213', '04222', '04226', '04232', '04235', '04237', '04254', '04258', '04259', '04267', '04279', '04283', '04287', '04289', '04290', '04291', '04295', '04298', '04313', '04317', '04319', '04326', '04328', '04329', '04363', '04364', '04365', '04368', '04372', '04373', '04376', '04386', '04387', '04389', '04390', '04396', '04398', '04402', '04405', '04408', '04417', '04438', '04439', '04446', '04456', '04460', '04464', '04466', '04467', '04474', '04476', '04489', '04490', '04494', '04498', '04507', '04518', '04519', '04520', '04522', '04536', '04546', '04550', '04560', '04581', '04586', '04589', '04597', '04598', '04601', '04604', '04607', '04608', '04616', '04622', '04624', '04626', '04637', '04641', '04644', '04645', '04650', '04656', '04660', '04669', '04672', '04676', '04678', '04681', '04685', '04689', '04690', '04693', '04694', '04698', '04706', '04707', '04715', '04728', '04729', '04742', '04744', '04751', '04752', '04753', '04756', '04758', '04760', '04767', '04794', '04798', '04800', '04801', '04803', '04813', '04815', '04818', '04825', '04827', '04832', '04842', '04861', '04862', '04875', '04881', '04889', '04890', '04895', '04902', '04923', '04934', '04943', '04950', '04955', '04961', '04972', '04977', '04978', '04980', '04987', '04995', '04998', '05005', '05006', '05013', '05016', '05019', '05025', '05027', '05029', '05036', '05037', '05042', '05046', '05056', '05062', '05063', '05066', '05068', '05081', '05089', '05091', '05092', '05093', '05095', '05097', '05099', '05103', '05108', '05111', '05112', '05116', '05117', '05124', '05125', '05127', '05128', '05135', '05136', '05138', '05139', '05144', '05147', '05148', '17219', '17235', '17239', '17245', '17250', '17258', '17263', '17275', '17276', '17278', '17286', '17287', '17291', '17293', '17294', '17297', '17307', '17309', '17313', '17318', '17324', '17329', '17333', '17334', '17341', '17347', '17359', '17362', '17364', '17367', '17371', '17378', '17385', '17386', '17389', '17402', '17417', '17418', '17421', '17425', '17439', '17441', '17446', '17447', '17455', '17457', '17462', '17466', '17475', '17477', '17483', '17485', '17497', '17506', '17514', '17518', '17521', '17527', '17528', '17538', '17539', '17541', '17546', '17547', '17550', '17554', '17562', '17564', '17565', '17566', '17580', '17583', '17597', '17599', '17601', '17602', '17603', '17606', '17611', '17612', '17616', '17621', '17623', '17635', '17637', '17656', '17662', '17665', '17666', '17674', '17681', '17682', '17683', '17690', '17693', '17696', '17704', '17705', '17719', '17722', '17727', '17729', '17730', '17741', '17745', '17746', '17747', '17751', '17753', '17756', '17757', '17761', '17763', '17765', '17768', '17770', '17771', '17784', '17786', '17788', '17800', '17801', '17805', '17806', '17807', '17809', '17815', '17821', '17828', '17829', '17831', '17836', '17837', '17851', '17855', '17863', '17865', '17873', '17876', '17877', '17883', '17884', '17893', '17894', '17897', '17906', '17908', '17911', '17919', '17932', '17933', '17937', '17946', '17947', '17949', '17953', '17957', '17962', '17968', '17974', '17987', '17990', '17994', '17995', '18002', '18007', '18008', '18011', '18018', '18022', '18026', '18028', '18033', '18037', '18041', '18043', '18045', '18048', '18055', '18063', '18075', '18078', '18080', '18082', '18092', '18094', '18097', '18104', '18106', '18107', '18110', '18111', '18112', '18126', '18128', '18152', '18159', '18162', '18171', '18172', '18178', '18179', '18180', '18181', '18195', '18197', '18206', '18208', '18209', '18211', '18221', '18228', '18232', '18237', '18238', '18239', '18241', '18253', '18270', '18280', '18288', '18292', '18294', '18299', '18301', '18302', '18304', '18312', '18319', '18320', '18323', '18327', '18335', '18337', '18342', '18353', '18354', '18357', '18358', '18360', '18364', '18365', '18375', '18377', '18381', '18382', '18383', '18393', '18394', '18401', '18405', '18414', '18417', '18419', '18431', '18434', '18436', '18441', '18446', '18447', '18451', '18453', '18459', '18460', '18462', '18466', '18468', '18477', '18482', '18490', '18492', '18493', '18496', '18499', '18501', '18511', '18518', '18532', '18542', '18543', '18546', '18551', '18556', '18561', '18566', '18570', '18573', '18582', '18587', '18596', '18601', '18602', '18616', '18617', '18620', '18632', '18648', '18653', '18655', '18660', '18661', '18665', '18676', '18680', '18681', '18683', '18684', '18692', '18694', '18698', '18702', '18710', '18712', '18716', '18718', '18725', '18734', '18739', '18741', '18743', '18754', '18756', '18759', '18761', '18771', '18777', '18778', '18780', '18783', '18786', '18796', '18797', '18802', '18813', '18814', '18815', '18831', '18835', '18836', '18841', '18849', '18850', '18853', '18857', '18859', '18865', '18872', '18878', '18881', '18887', '18892', '18900', '18911', '18912', '18916', '18922', '18928', '18940', '18949', '18959', '18960', '18962', '18981', '18986', '18988', '18989', '18994', '19001', '19008', '19017', '19026', '19034', '19038', '19040', '19045', '19046', '19047', '19050', '19052', '19054', '19058', '19059', '19060', '19066', '19071', '19074', '19077', '19079', '19088', '19096', '19098', '19101', '19102', '19108', '19111', '19114', '19120', '19121', '19125', '19128', '19130', '19136', '19147', '19151', '19154', '19155', '19165', '19168', '19170', '19181', '19189', '19194', '19196', '19202', '19204', '19207', '19211', '19214', '19220', '19225', '19229', '19235', '19237', '19240', '19251', '19257', '19259', '19266', '19267', '19268', '19274', '19277', '19285', '19297', '19298', '19304', '19309', '19313', '19320', '19324', '19326', '19327', '19331', '19335', '19345', '19347', '19348', '19357', '19359', '19364', '19370', '19373', '19374', '19375', '19377', '19383', '19384', '19386', '19388', '19398', '19400', '19405', '19408', '19409', '19419', '19421', '19423', '19426', '19428', '19431', '19432', '19437', '19440', '19441', '19443', '19448', '19451', '19452', '19455', '19459', '19460', '19464', '19467', '19475', '19480', '19483', '19487', '19496', '19498', '19512', '19520', '19528', '19532', '19534', '19535', '19542', '19550', '19555', '19561', '19572', '19573', '19579', '19581', '19586', '19587', '19591', '19604', '19606', '19636', '19650', '19652', '19654', '19655', '19658', '19660', '19665', '19673', '19675', '19690', '19708', '19711', '19718', '19729', '19733', '19745', '19749', '19755', '19756', '19757', '19764', '19772', '19776', '19778', '19786', '19787', '19789', '19790', '19792', '19793', '19796', '19800', '19801', '19805', '19820', '19823', '19829', '19840', '19845', '19857', '19866', '19871', '19879', '19881', '19890', '19904', '19913', '19918', '19923', '19931', '19932', '19936', '19937', '19941', '19945', '19956', '19962', '19965', '19966', '19976', '19978', '19979', '19984', '19989', '19993', '19994', '19998', '20002', '20003', '20007', '20011', '20012', '20014', '20018', '20019', '20020', '20032', '20034', '20037', '20039', '20040', '20041', '20052', '20065', '20074', '20084', '20085', '20086', '20087', '20088', '20098', '20100', '20117', '20118', '20129', '20130', '20144', '20161', '20166', '20170', '20179', '20183', '20186', '20188', '20189', '20215', '20237', '20251', '20253', '20255', '20257', '20265', '20266', '20269', '20274', '20275', '20276', '20280', '20283', '20284', '20285', '20286', '20289', '20290', '20303', '20307', '20313', '20329', '20339', '20342', '20348', '20352', '20366', '20367', '20374', '20375', '20376', '20382', '20389', '20391', '20393', '20396', '20399', '20402', '20403', '20405', '20414', '20416', '20424', '20435', '20438', '20441', '20449', '20452', '20461', '20463', '20465', '20468', '20472', '20477', '20483', '20490', '20494', '20495', '20504', '20511', '20515', '20516', '20518', '20520', '20523', '20526', '20529', '20530', '20535', '20541', '20561', '20566', '20569', '20573', '20579', '20595', '20599', '20605', '20606', '20608', '20614', '20618', '20621', '20629', '20631', '20643', '20644', '20662', '20669', '20677', '20679', '20682', '20688', '20691', '20692', '20701', '20709', '20711', '20713', '20721', '20724', '20729', '20737', '20739', '20740', '20741', '20743', '20755', '20768', '20776', '20782', '20787', '20793', '20800', '20803', '20804', '20813', '20816', '20817', '20820', '20822', '20829', '20835', '20837', '20838', '20846', '20849', '20858', '20866', '20874', '20888', '20889', '20892', '20898', '20919', '20921', '20923', '20929', '20933', '20936', '20943', '20944', '20946', '20951', '20958', '20962', '20964', '20966', '20976', '20977', '20988', '20996', '21010', '21011', '21012', '21026', '21036', '21042', '21051', '21053', '21054', '21060', '21065', '21068', '21078', '21080', '21082', '21083', '21096', '21102', '21106', '21112', '21115', '21116', '21131', '21133', '21138', '21149', '21155', '21158', '21162', '21164', '21172', '21175', '21193', '21202', '21203', '21213', '21214', '21218', '21219', '21223', '21224', '21233', '21235', '21239', '21244', '21247', '21251', '21253', '21256', '21260', '21265', '21270', '21273', '21276', '21277', '21284', '21286', '21289', '21293', '21294', '21304', '21310', '21313', '21314', '21318', '21333', '21340', '21348', '21353', '21354', '21363', '21369', '21373', '21380', '21381', '21383', '21387', '21392', '21394', '21403', '21406', '21407', '21411', '21412', '21425', '21426', '21428', '21435', '21439', '21442', '21444', '21447', '21449', '21455', '21457', '21468', '21471', '21483', '21485', '21488', '21489', '21495', '21496', '21497', '21504', '21505', '21506', '21507', '21510', '21512', '21516', '21518', '21519', '21521', '21530', '21537', '21540', '21542', '21544', '21550', '21561', '21565', '21570', '21572', '21577', '21579', '21590', '21594', '21595', '21597', '21603', '21605', '21607', '21613', '21614', '21623', '21626', '21632', '21639', '21640', '21644', '21645', '21648', '21649', '21652', '21655', '21659', '21663', '21664', '21666', '21673', '21674', '21683', '21686', '21690', '21695', '21702', '21703', '21705', '21709', '21713', '21717', '21724', '21728', '21742', '21743', '21752', '21758', '21770', '21772', '21783', '21784', '21793', '21805', '21809', '21821', '21827', '21838', '21843', '21844', '21846', '21851', '21872', '21875', '21877', '21878', '21887', '21891', '21896', '21903', '21907', '21915', '21916', '21919', '21920', '21922', '21929', '21936', '21937', '21946', '21947', '21949', '21951', '21954', '21955', '21957', '21958', '21960', '21964', '21968', '21971', '21973', '21979', '21981', '21986', '21988', '21989', '21993', '21998', '22004', '22005', '22006', '22010', '22045', '22048', '22056', '22062', '22063', '22069', '22074', '22079', '22095', '22097', '22100', '22105', '22106', '22111', '22122', '22123', '22124', '22127', '22128', '22129'],
                    img_transform=transform_val,
                    is_validation=True,
                    )
            validation_datasets['RLSID'] = val_dataset
    
    # Print dataset sizes
    for name, dataset in validation_datasets.items():
        print(f'NUM of {name} validation images: {len(dataset)}')
    
    return validation_datasets


# Validation utils abril
def create_datasets(args, transforms):
    """Create validation datasets for multiple datasets"""
    validation_datasets = {}
    
    # Get validation datasets to create (default to MIT and RSR_256)
    val_datasets = getattr(args, 'validation_datasets', ['mit', 'rsr_256'])

    # Dataset initialization
    if args.dataset == 'mit':
        train_dataset = MIT_Dataset_PreLoad(args.data_path, transforms[0], 
                                           total_split=args.world_size, split_id=args.rank)
        if 'mit' in val_datasets:
            validation_datasets['MIT'] = MIT_Dataset('/home/apinyol/TFM/Data/multi_illumination_test_mip2_jpg', 
                                 transforms[1], eval_mode=True)
    elif args.dataset == 'iiw':
        train_dataset = IIW2(root=args.data_path, img_transform=transforms[0], 
                           split='train', return_two_images=True)
        if 'iiw' in val_datasets:
            validation_datasets['IIW'] = IIW2(root=args.data_path, img_transform=transforms[1], 
                          split='val', return_two_images=True)
    elif args.dataset == 'rsr_256':
        train_dataset = RSRDataset(root_dir=args.data_path, img_transform=transforms[0], 
                                 is_validation=False, validation_scenes=["scene_01", "scene_03", "scene_05", "scene_06", "scene_07", "scene_08", "scene_10", "scene_21", "scene_22", "scene_23", "scene_24", "scene_26", "scene_27", "scene_29", "scene_30"])
        if 'rsr_256' in val_datasets:
            validation_datasets['RSR_256'] = RSRDataset(root_dir=args.data_path, img_transform=transforms[1], 
                                is_validation=True, validation_scenes=["scene_01", "scene_03", "scene_05", "scene_06", "scene_07", "scene_08", "scene_10", "scene_21", "scene_22", "scene_23", "scene_24", "scene_26", "scene_27", "scene_29", "scene_30"]) 
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Print dataset sizes
    for name, dataset in validation_datasets.items():
        print(f'NUM of {name} validation images: {len(dataset)}')
    
    return train_dataset, validation_datasets


def create_validation_loaders(validation_datasets, args, val_sampler):
    """Create validation data loaders for all datasets"""
    validation_loaders = {}
    val_batch_size = getattr(args, 'val_batch_size', args.batch_size)
    
    for dataset_name, dataset in validation_datasets.items():
        """if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False, drop_last=False)
        else:
            val_sampler = None"""
        val_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=val_batch_size, 
            shuffle=False,
            num_workers=args.workers, 
            pin_memory=False, 
            sampler=val_sampler, 
            drop_last=False, 
            persistent_workers=True
        )
        validation_loaders[dataset_name] = val_loader
    
    return validation_loaders


# checkpoint management abril

def create_experiment_folder(args):
    """Create a unique experiment folder with timestamp and run ID"""
    import datetime
    import uuid
    
    # Create base checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Generate timestamp and short run ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    
    # Create experiment folder name
    experiment_folder = f"{args.experiment_name}_{timestamp}_{run_id}"
    
    # Create full path
    experiment_path = os.path.join(args.checkpoint_dir, experiment_folder)
    os.makedirs(experiment_path, exist_ok=True)
    
    # Save experiment config
    config_path = os.path.join(experiment_path, "config.json")
    config = {
        "experiment_name": args.experiment_name,
        "timestamp": timestamp,
        "run_id": run_id,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "intrinsics_loss_weight": args.intrinsics_loss_weight,
            "reg_weight": args.reg_weight,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "affine_scale": args.affine_scale,
            "img_size": args.img_size,
            "epochs": args.epochs,
            "enable_depth_loss": args.enable_depth_loss,
            "dataset": args.dataset,
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created experiment folder: {experiment_path}")
    return experiment_path

def find_latest_checkpoint(experiment_path):
    """Find the latest checkpoint in the experiment folder"""
    checkpoint_files = glob.glob(os.path.join(experiment_path, "epoch_*.pth"))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('epoch_')[1].split('.pth')[0]))
    return checkpoint_files[-1]

def find_experiment_folder(args):
    """Find experiment folder for resuming"""
    if args.resume_from and args.is_master:
        if os.path.isfile(args.resume_from):
            return create_experiment_folder(args), args.resume_from
        else:
            print(f"Resume path not found: {args.resume_from}")
            return None, None
    
    if args.auto_resume:
        # Find latest experiment folder with same experiment name
        experiment_folders = glob.glob(os.path.join(args.checkpoint_dir, f"{args.experiment_name}_*"))
        if experiment_folders:
            latest_folder = max(experiment_folders, key=os.path.getctime)
            latest_checkpoint = find_latest_checkpoint(latest_folder)
            if latest_checkpoint:
                return latest_folder, latest_checkpoint
    
    return None, None

def cleanup_old_checkpoints(experiment_path, keep_last=3):
    """Remove old checkpoints, keeping only the last N"""
    if keep_last <= 0:
        return
    
    checkpoint_files = glob.glob(os.path.join(experiment_path, "epoch_*.pth"))
    if len(checkpoint_files) <= keep_last:
        return
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('epoch_')[1].split('.pth')[0]))
    
    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[:-keep_last]:
        try:
            os.remove(old_checkpoint)
            print(f"Removed old checkpoint: {os.path.basename(old_checkpoint)}")
        except OSError:
            pass

def save_checkpoint_improved(state, experiment_path, epoch, is_best=False, keep_last=3):
    """Save checkpoint with better naming and cleanup"""
    # Save epoch checkpoint
    epoch_filename = os.path.join(experiment_path, f"epoch_{epoch:04d}.pth.tar")
    torch.save(state, epoch_filename)
    
    # Save as latest
    latest_filename = os.path.join(experiment_path, "latest.pth.tar")
    torch.save(state, latest_filename)
    
    # Save as best if specified
    if is_best:
        best_filename = os.path.join(experiment_path, "best.pth.tar")
        torch.save(state, best_filename)
    
    # Cleanup old checkpoints
    cleanup_old_checkpoints(experiment_path, keep_last)
    
    print(f"Saved checkpoint: epoch_{epoch:04d}.pth.tar")

def load_checkpoint_improved(checkpoint_path, model, ema_model, optimizer, scaler, args):#canviar device per args, i args.gpu al codi
    """Load checkpoint with error handling"""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}")
        return 0

    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        if args.gpu is None:
            checkpoint = torch.load(checkpoint_path)
        else:
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(checkpoint_path, map_location=loc)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])

        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0
    
def load_checkpoint_improved2(checkpoint_path, model, ema_model, optimizer, scaler, args):
    """Load checkpoint with error handling and partial loading support"""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}")
        return 0
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        if args.gpu is None:
            checkpoint = torch.load(checkpoint_path)
        else:
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(checkpoint_path, map_location=loc)
        
        start_epoch = checkpoint['epoch']
        
        # Load model weights with partial loading support
        _load_model_weights_partial(model, checkpoint['state_dict'])
        
        # Load EMA model weights with partial loading support
        _load_model_weights_partial(ema_model, checkpoint['ema_state_dict'])
        
        # Load optimizer and scaler (these should be compatible)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0


def _load_model_weights_partial(model, state_dict):
    """
    Load weights into model, handling missing keys by initializing them properly.
    This allows loading checkpoints from models with different architectures.
    """
    import torch.nn as nn
    
    # Get current model's state dict
    model_state_dict = model.state_dict()
    
    # Track what gets loaded and what gets initialized
    loaded_keys = []
    missing_keys = []
    unexpected_keys = []
    
    # Load existing weights
    for key in model_state_dict.keys():
        if key in state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == state_dict[key].shape:
                model_state_dict[key] = state_dict[key]
                loaded_keys.append(key)
            else:
                print(f"Shape mismatch for {key}: model={model_state_dict[key].shape}, checkpoint={state_dict[key].shape}")
                missing_keys.append(key)
        else:
            missing_keys.append(key)
    
    # Check for unexpected keys in checkpoint
    for key in state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    
    # Initialize missing weights
    for key in missing_keys:
        param = model_state_dict[key]
        _initialize_parameter(param, key)
        print(f"Initialized missing parameter: {key} with shape {param.shape}")
    
    # Load the updated state dict
    model.load_state_dict(model_state_dict)
    
    # Print summary
    print(f"Loaded {len(loaded_keys)} parameters from checkpoint")
    if missing_keys:
        print(f"Initialized {len(missing_keys)} missing parameters")
    if unexpected_keys:
        print(f"Ignored {len(unexpected_keys)} unexpected keys from checkpoint")


def _initialize_parameter(param, param_name):
    """
    Initialize a parameter based on its name and shape.
    Uses appropriate initialization schemes for different layer types.
    """
    import torch.nn as nn
    
    with torch.no_grad():
        # Determine initialization based on parameter name and shape
        if 'weight' in param_name.lower():
            if len(param.shape) >= 2:  # Linear/Conv layers
                if 'norm' in param_name.lower() or 'bn' in param_name.lower():
                    # Batch norm or layer norm weights
                    nn.init.ones_(param)
                elif 'embedding' in param_name.lower():
                    # Embedding layers
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    # General linear/conv layers - use Xavier/Glorot uniform
                    nn.init.xavier_uniform_(param)
            else:
                # 1D weight parameters (like in some attention mechanisms)
                nn.init.normal_(param, mean=0.0, std=0.02)
                
        elif 'bias' in param_name.lower():
            # Bias parameters
            nn.init.zeros_(param)
            
        elif 'norm' in param_name.lower():
            # Normalization layer parameters
            if len(param.shape) == 1:
                if 'weight' in param_name.lower():
                    nn.init.ones_(param)
                else:
                    nn.init.zeros_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)
                
        else:
            # Default initialization for other parameters
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)
    
    print(f"  -> Used {'Xavier uniform' if len(param.shape) >= 2 and 'weight' in param_name.lower() and 'norm' not in param_name.lower() else 'Normal(0, 0.02)' if 'embedding' in param_name.lower() or len(param.shape) == 1 else 'Zeros' if 'bias' in param_name.lower() else 'Ones' if 'norm' in param_name.lower() and 'weight' in param_name.lower() else 'Default'} initialization")
    


    #depth

class DepthNormalConsistencyLoss(torch.nn.Module):
    """Enhanced loss with both depth and normal consistency"""
    
    def __init__(self, 
                 depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 normals_model_name="GonzaloMG/marigold-e2e-ft-normals",
                 device='cuda',
                 enable_depth=True,
                 enable_normals=True):
        super().__init__()
        self.device = device
        self.enable_depth = enable_depth
        self.enable_normals = enable_normals
        
        # Load Depth model if enabled
        if self.enable_depth:
            self.processor = AutoImageProcessor.from_pretrained(depth_model_name)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)
            self.depth_model.to(device)
            self.depth_model.eval()
            
            # Freeze depth model parameters
            for param in self.depth_model.parameters():
                param.requires_grad = False
        
        # Load Normal model if enabled
        if self.enable_normals:
            self.normal_pipe = DiffusionPipeline.from_pretrained(
                normals_model_name,
                custom_pipeline="GonzaloMG/marigold-e2e-ft-normals",
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            ).to(device)
            self.normal_pipe.set_progress_bar_config(disable=True)
            
            # Freeze normal model parameters
            for param in self.normal_pipe.unet.parameters():
                param.requires_grad = False
            for param in self.normal_pipe.vae.parameters():
                param.requires_grad = False

    def forward(self, img1, img2):
        """
        Compute depth and/or normal consistency loss between two images
        img1, img2: (B, C, H, W) in range [-1, 1]
        """
        total_loss = torch.tensor(0.0, device=img1.device)
        depth1, depth2 = None, None
        normals1, normals2 = None, None
        depth_loss, normal_loss = None, None
        
        # Depth consistency loss
        if self.enable_depth:
            depth1 = self.get_depth_map(img1.detach())
            depth2 = self.get_depth_map(img2)
            
            # Normalize depth maps to [0, 1] for better comparison
            depth1_norm = self.normalize_depth_maps(depth1)
            depth2_norm = self.normalize_depth_maps(depth2)
            
            # Compute L1 loss between depth maps
            depth_loss = F.l1_loss(depth1_norm, depth2_norm)
            total_loss += depth_loss
        
        # Normal consistency loss
        if self.enable_normals:
            normals1 = self.get_normal_map(img1.detach())
            normals2 = self.get_normal_map(img2)
            
            # Normalize normal maps if needed
            normals1_norm = self.normalize_normal_maps(normals1)
            normals2_norm = self.normalize_normal_maps(normals2)
            
            # Compute angular loss between normal maps (cosine similarity)
            normal_loss = self.compute_normal_loss(normals1_norm, normals2_norm)
            total_loss += normal_loss

       #print(f"SHAPES: Normals input --> {normals1_norm.shape}, Normals relight --> {normals2_norm.shape}, Depth input --> {depth1_norm.shape}, Depth relight --> {depth2_norm.shape}")
        
        return total_loss, depth_loss, normal_loss, depth1, depth2, normals1, normals2
    
    def preprocess_image(self, img_tensor):
        """Convert tensor from [-1, 1] to PIL Image format"""
        # Convert from [-1, 1] to [0, 1]
        img_normalized = (img_tensor + 1.0) / 2.0
        # Convert to [0, 255]
        img_255 = (img_normalized * 255.0).clamp(0, 255)
        
        # Convert to PIL format
        img_pil_format = img_255.permute(0, 2, 3, 1).detach().cpu().numpy().astype('uint8')
        
        return img_pil_format
    
    def get_depth_map(self, img_tensor):
        """Get depth map from image tensor"""
        if not self.enable_depth:
            return None
            
        batch_size = img_tensor.shape[0]
        original_device = img_tensor.device
        
        # Detach input for depth estimation
        with torch.no_grad():
            img_array = self.preprocess_image(img_tensor)
        
        depth_maps = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Process single image
                inputs = self.processor(images=img_array[i], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get depth prediction
                outputs = self.depth_model(**inputs)
                depth = outputs.predicted_depth
                
                # Resize depth to match input image size
                depth_resized = F.interpolate(
                    depth.unsqueeze(1), 
                    size=img_tensor.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                depth_maps.append(depth_resized)
        
        # Concatenate and ensure it's on the original device
        depth_result = torch.cat(depth_maps, dim=0)
        return depth_result.to(original_device)
    
    def get_normal_map(self, img_tensor):
        """Get normal map from image tensor using the marigold pipeline"""
        if not self.enable_normals:
            return None
            
        batch_size = img_tensor.shape[0]
        original_device = img_tensor.device
        
        # Convert tensor to PIL images
        with torch.no_grad():
            img_array = self.preprocess_image(img_tensor)
        
        normal_maps = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Convert to PIL Image
                pil_img = Image.fromarray(img_array[i])
                
                # Get normal prediction
                result = self.normal_pipe(pil_img)
                normal_prediction = result.prediction

                # Convert to tensor format (B, C, H, W) - normals are in [-1, 1] range
                if isinstance(normal_prediction, torch.Tensor):
                    normal_tensor = normal_prediction
                else:
                    # Convert PIL to tensor if needed
                    #normal_array = np.array(normal_prediction)
                    normal_tensor = torch.from_numpy(normal_prediction).float()
                    # Normalize from [0, 255] to [-1, 1] if needed
                    if normal_tensor.max() > 1.0:
                        normal_tensor = (normal_tensor / 255.0) * 2.0 - 1.0
                
                # Ensure tensor is on correct device before interpolation
                normal_tensor = normal_tensor.to(self.device).permute(0, 3, 1, 2)
                
                normal_maps.append(normal_tensor)
        
        # Concatenate and ensure it's on the original device
        normal_result = torch.cat(normal_maps, dim=0)
        return normal_result.to(original_device)
    
    def normalize_depth_maps(self, depth_maps):
        """Normalize depth maps to [0, 1] per image"""
        if depth_maps is None:
            return None
            
        normalized = torch.zeros_like(depth_maps)
        for i in range(depth_maps.shape[0]):
            d = depth_maps[i]
            d_min, d_max = d.min(), d.max()
            if d_max > d_min:
                normalized[i] = (d - d_min) / (d_max - d_min)
            else:
                normalized[i] = d
        return normalized
    
    def normalize_normal_maps(self, normal_maps):
        """Normalize normal maps to unit length"""
        if normal_maps is None:
            return None
            
        # Ensure normals are unit vectors
        norm = torch.norm(normal_maps, dim=1, keepdim=True)
        normalized = normal_maps / (norm + 1e-8)
        return normalized
    
    def compute_normal_loss(self, normals1, normals2):
        """Compute angular loss between normal maps using cosine similarity"""
        # Compute cosine similarity (dot product for unit vectors)
        cos_sim = torch.sum(normals1 * normals2, dim=1, keepdim=True)
        
        # Clamp to avoid numerical issues
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # Angular loss: 1 - cos_sim (0 for identical, 2 for opposite)
        angular_loss = 1.0 - cos_sim
        
        return angular_loss.mean()
    
    def create_comprehensive_visualization(self, input_img, relit_img, gt_img, input_depth=None, relit_depth=None, gt_depth=None,
                                        input_normals=None, relit_normals=None, gt_normals=None, epoch=0, dataset_name='validation'):
        """Create comprehensive visualization with 3 columns: Input, Relit, GT"""
        try:
            # Take first image from batch for visualization
            input_vis = self.tensor_to_numpy(input_img[0])
            relit_vis = self.tensor_to_numpy(relit_img[0])
            gt_vis = self.tensor_to_numpy(gt_img[0])
            
            # Determine number of rows based on available data
            n_rows = 1  # Always have image row
            has_depth = input_depth is not None and relit_depth is not None and gt_depth is not None
            has_normals = input_normals is not None and relit_normals is not None and gt_normals is not None
            
            if has_depth:
                n_rows += 1
            if has_normals:
                n_rows += 1
            
            n_cols = 3  # Always 3 columns: Input, Relit, GT
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            fig.suptitle(f'{dataset_name} - Input vs Relit vs GT Comparison (Epoch {epoch})', fontsize=16, fontweight='bold')
            
            # Ensure axes is 2D array
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            current_row = 0
            
            # Row 1: Images
            axes[current_row, 0].imshow(input_vis)
            axes[current_row, 0].set_title('Input Image')
            axes[current_row, 0].axis('off')
            
            axes[current_row, 1].imshow(relit_vis)
            axes[current_row, 1].set_title('Relit Image')
            axes[current_row, 1].axis('off')
            
            axes[current_row, 2].imshow(gt_vis)
            axes[current_row, 2].set_title('Ground Truth')
            axes[current_row, 2].axis('off')
            
            current_row += 1
            
            # Row 2: Depth maps (if available)
            if has_depth:
                input_depth_vis = input_depth[0, 0].cpu().numpy()
                relit_depth_vis = relit_depth[0, 0].cpu().numpy()
                gt_depth_vis = gt_depth[0, 0].cpu().numpy()
                
                # Find common depth range for consistent visualization
                depth_min = min(input_depth_vis.min(), relit_depth_vis.min(), gt_depth_vis.min())
                depth_max = max(input_depth_vis.max(), relit_depth_vis.max(), gt_depth_vis.max())
                
                im1 = axes[current_row, 0].imshow(input_depth_vis, cmap='plasma', vmin=depth_min, vmax=depth_max)
                axes[current_row, 0].set_title('Input Depth')
                axes[current_row, 0].axis('off')
                plt.colorbar(im1, ax=axes[current_row, 0], fraction=0.046, pad=0.04)
                
                im2 = axes[current_row, 1].imshow(relit_depth_vis, cmap='plasma', vmin=depth_min, vmax=depth_max)
                axes[current_row, 1].set_title('Relit Depth')
                axes[current_row, 1].axis('off')
                plt.colorbar(im2, ax=axes[current_row, 1], fraction=0.046, pad=0.04)
                
                im3 = axes[current_row, 2].imshow(gt_depth_vis, cmap='plasma', vmin=depth_min, vmax=depth_max)
                axes[current_row, 2].set_title('GT Depth')
                axes[current_row, 2].axis('off')
                plt.colorbar(im3, ax=axes[current_row, 2], fraction=0.046, pad=0.04)
                
                current_row += 1
            
            # Row 3: Normal maps (if available)
            if has_normals:
                input_normals_vis = self.normal_to_rgb(input_normals[0])
                relit_normals_vis = self.normal_to_rgb(relit_normals[0])
                gt_normals_vis = self.normal_to_rgb(gt_normals[0])
                
                axes[current_row, 0].imshow(input_normals_vis)
                axes[current_row, 0].set_title('Input Normals')
                axes[current_row, 0].axis('off')
                
                axes[current_row, 1].imshow(relit_normals_vis)
                axes[current_row, 1].set_title('Relit Normals')
                axes[current_row, 1].axis('off')
                
                axes[current_row, 2].imshow(gt_normals_vis)
                axes[current_row, 2].set_title('GT Normals')
                axes[current_row, 2].axis('off')
            
            plt.tight_layout()
            
            # Convert to image for wandb
            fig.canvas.draw()
            try:
                buf = fig.canvas.buffer_rgba()
                buf = np.asarray(buf)
                buf = buf[:, :, :3]  # Remove alpha channel
            except AttributeError:
                try:
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                except AttributeError:
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    from PIL import Image
                    img = Image.open(buf)
                    buf = np.array(img)
                    if buf.shape[-1] == 4:
                        buf = buf[:, :, :3]
            
            plt.close()
            return buf
            
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def normal_to_rgb(self, normal_tensor):
        """Convert normal map tensor to RGB for visualization"""
        # Normalize from [-1, 1] to [0, 1]
        normal_rgb = (normal_tensor + 1.0) / 2.0
        normal_rgb = torch.clamp(normal_rgb, 0, 1)
        return normal_rgb.permute(1, 2, 0).cpu().numpy()
    
    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy for visualization"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor.permute(1, 2, 0).cpu().numpy()