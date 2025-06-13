import torchvision, glob
import torch.utils.data as data
from PIL import ImageFilter, ImageOps
import numpy as np
import random, pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class MIT_Dataset(data.Dataset):
    def __init__(self, root, img_transform):
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_meta = {'img_root':root, 'img_folder_list':[], 'img_folder_img_num':[], 'img_num_cumsum':[]}
        for img_folder_path in img_folder_list:
            img_list = glob.glob(img_folder_path + '/dir*.jpg')
            self.img_meta['img_folder_list'].append(img_folder_path.split('/')[-1])
            self.img_meta['img_folder_img_num'].append(len(img_list))
        self.img_meta['img_num_cumsum'] = np.cumsum(np.array(self.img_meta['img_folder_img_num']))
        self.img_transform = img_transform

    def __len__(self):
        return self.img_meta['img_num_cumsum'][-1]

    def __getitem__(self, index):
        folder_index = np.searchsorted(self.img_meta['img_num_cumsum'], index, side = 'right')
        if folder_index == 0:
            folder_offset = index
        else:
            folder_offset = index - self.img_meta['img_num_cumsum'][folder_index - 1]
        img_list = glob.glob(self.img_meta['img_root'] + '/' + self.img_meta['img_folder_list'][folder_index] + '/dir*.jpg')
        img_list.sort()
        pair_img_folder_offset = np.random.choice(np.where(np.arange(len(img_list)) != folder_offset)[0])
        img1 = Image.open(img_list[folder_index])
        img2 = Image.open(img_list[pair_img_folder_offset])
        return self.img_transform(img1), self.img_transform(img2)

    def get_img_folder_list(self, folder_index):
        #folder_index = np.searchsorted(self.img_meta['img_num_cumsum'], index, side = 'right')
        img_list = glob.glob(self.img_meta['img_root'] + '/' + self.img_meta['img_folder_list'][folder_index] + '/dir*.jpg')
        img_list.sort()
        img_tensor_list = []
        for img_path in img_list:
            img = Image.open(img_path)
            img_tensor_list.append(self.img_transform(img))
        return img_tensor_list
