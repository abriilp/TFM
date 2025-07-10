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
    val_datasets = getattr(args, 'validation_datasets', ['mit', 'rsr_256'])
    
    for dataset_name in val_datasets:
        print(f"Creating validation dataset for: {dataset_name}")
        
        if dataset_name == 'mit':
            val_dataset = MIT_Dataset('/home/apinyol/TFM/Data/multi_illumination_test_mip2_jpg', 
                                     transform_val, eval_mode=True)
            validation_datasets['MIT'] = val_dataset
            
        elif dataset_name == 'rsr_256':
            # Use RSR validation path - you might need to adjust this path
            rsr_val_path = getattr(args, 'rsr_val_path', args.data_path if hasattr(args, 'data_path') else '/home/apinyol/TFM/Data/RSR_256')
            val_dataset = RSRDataset(root_dir=rsr_val_path, 
                                   img_transform=transform_val, 
                                   is_validation=True)
            validation_datasets['RSR_256'] = val_dataset
            
        elif dataset_name == 'iiw':
            # Add IIW validation if needed
            iiw_val_path = getattr(args, 'iiw_val_path', '/path/to/iiw/data')
            val_dataset = IIW2(root=iiw_val_path,
                              img_transform=transform_val,
                              split='val',
                              return_two_images=True)
            validation_datasets['IIW'] = val_dataset
    
    # Print dataset sizes
    for name, dataset in validation_datasets.items():
        print(f'NUM of {name} validation images: {len(dataset)}')
    
    return validation_datasets


def create_validation_loaders(validation_datasets, args, val_sampler):
    """Create validation data loaders for all datasets"""
    validation_loaders = {}
    val_batch_size = getattr(args, 'val_batch_size', args.batch_size)
    
    for dataset_name, dataset in validation_datasets.items():
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


def validate_multi_datasets(validation_loaders, model, epoch, args, current_step=None):
    """Validate on multiple datasets and log results separately"""
    from visu_utils import log_relighting_results
    
    model.eval()
    all_val_metrics = {}
    
    # Global step counter for wandb logging
    if current_step is None:
        current_step = epoch * sum(len(loader) for loader in validation_loaders.values())

    with torch.no_grad():
        step_offset = 0
        
        for dataset_name, val_loader in validation_loaders.items():
            print(f"Validating on {dataset_name} dataset...")
            dataset_metrics = {}
            
            # Track metrics across batches
            reconstruction_losses = []
            input_reconstruction_losses = []
            
            for i, batch in enumerate(val_loader):
                val_step = current_step + step_offset + i
                if i >= getattr(args, 'max_val_batches', 3):  # Limit validation batches for speed
                    break
                
                # Handle different batch formats
                if len(batch) == 3:
                    input_img, ref_img, gt_img = batch
                else:
                    # Fallback for 2-image case
                    input_img, ref_img = batch
                    gt_img = ref_img  # Use ref as gt fallback
                    
                input_img = input_img.to(args.gpu)
                ref_img = ref_img.to(args.gpu)
                gt_img = gt_img.to(args.gpu)
                
                # Forward pass without noise for validation
                intrinsic_input, extrinsic_input = model(input_img, run_encoder=True)
                intrinsic_ref, extrinsic_ref = model(ref_img, run_encoder=True)
                
                # Proper relighting for validation - use input intrinsics with ref extrinsics
                relit_img = model([intrinsic_input, extrinsic_ref], run_encoder=False).float()
                
                # Compute validation metrics against ground truth
                val_reconstruction_loss = nn.MSELoss()(relit_img, gt_img)
                val_input_reconstruction = nn.MSELoss()(relit_img, input_img)  # For comparison
                
                reconstruction_losses.append(val_reconstruction_loss.item())
                input_reconstruction_losses.append(val_input_reconstruction.item())
                
                # Log validation visualizations (only for first batch of each dataset)
                if args.is_master and not args.disable_wandb and i == 0:
                    log_relighting_results(
                        input_img=input_img,
                        ref_img=ref_img,
                        relit_img=relit_img,
                        gt_img=gt_img,
                        step=val_step,
                        mode=f'validation_{dataset_name.lower()}'
                    )
            
            # Aggregate metrics for this dataset
            if reconstruction_losses:
                dataset_metrics = {
                    f'val_{dataset_name.lower()}_reconstruction_loss_vs_gt': sum(reconstruction_losses) / len(reconstruction_losses),
                    f'val_{dataset_name.lower()}_reconstruction_loss_vs_input': sum(input_reconstruction_losses) / len(input_reconstruction_losses),
                    f'val_{dataset_name.lower()}_epoch': epoch,
                    f'val_{dataset_name.lower()}_num_batches': len(reconstruction_losses),
                }
                
                all_val_metrics[dataset_name] = dataset_metrics
                
                # Log to wandb with dataset-specific prefix
                if args.is_master and not args.disable_wandb:
                    wandb.log(dataset_metrics, step=current_step + step_offset)
            
            step_offset += len(val_loader)
    
    model.train()
    return all_val_metrics





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
    epoch_filename = os.path.join(experiment_path, f"epoch_{epoch:04d}.pth")
    torch.save(state, epoch_filename)
    
    # Save as latest
    latest_filename = os.path.join(experiment_path, "latest.pth")
    torch.save(state, latest_filename)
    
    # Save as best if specified
    if is_best:
        best_filename = os.path.join(experiment_path, "best.pth")
        torch.save(state, best_filename)
    
    # Cleanup old checkpoints
    cleanup_old_checkpoints(experiment_path, keep_last)
    
    print(f"Saved checkpoint: epoch_{epoch:04d}.pth")

def load_checkpoint_improved(checkpoint_path, model, ema_model, optimizer, scaler, args):
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