from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
import os
import pickle

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)




class IRSDataset(data.Dataset):
    def __init__(self, opt, seed=0):
        super(IRSDataset, self).__init__()
        self.opt = opt
        self.image_root_path = "/data/124-1/datasets/ISR/"
        self.scenes = sorted(os.listdir(self.image_root_path))

        self.name = opt['name']
        if self.name == "ValSet":
            self.samples = pickle.load(open("/data/124-1/datasets/ISR/val_samples.pkl", "rb"))
            self.imgsXscene = pickle.load(open("/data/124-1/datasets/ISR/val_dict.pkl", "rb"))

        elif self.name == "TrainSet":
            self.samples = pickle.load(open("/data/124-1/datasets/ISR/train_samples.pkl", "rb"))
            self.imgsXscene = pickle.load(open("/data/124-1/datasets/ISR/train_dict.pkl", "rb"))

        elif self.name == "TrainMiniSet":
            self.samples = pickle.load(open("/data/124-1/datasets/ISR/mini_train_samples.pkl", "rb"))
            self.imgsXscene = pickle.load(open("/data/124-1/datasets/ISR/mini_train_dict.pkl", "rb"))
            
        elif self.name == "TestSet":
            self.samples = pickle.load(open("/data/124-1/datasets/ISR/test_samples.pkl", "rb"))
            self.imgsXscene = pickle.load(open("/data/124-1/datasets/ISR/test_dict.pkl", "rb"))
            
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, _, i1, i2 = self.samples[idx]
        lq_obj = self.imgsXscene[scene][i1]
        gt_obj = self.imgsXscene[scene][i2]
        
        des_light = torch.Tensor([gt_obj['pan'], gt_obj['tilt']])
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = gt_obj['img_path']
        lq_path = lq_obj['img_path']

        # image range: [0, 1], float32., H W 3
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        # if self.opt['phase'] == 'train':
            # gt_size = self.opt['gt_size']
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # print(img_lq.shape,img_gt.shape,img_lq.min(),img_gt.min(),img_lq.max(),img_gt.max(),lq_path,gt_path)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'des_light': des_light}


class CustomDataset(data.Dataset):
    def __init__(self, opt, seed=0):
        super(CustomDataset, self).__init__()
        self.opt = opt
        self.image_root_path = "/data/124-1/datasets/RSR_256/"
        self.scenes = sorted(os.listdir(self.image_root_path))

        self.name = opt['name']
        if self.name == "ValSet":
            self.scenes = [f'scene_{n}' for n in range(61, 71)]
        elif self.name == "TrainSet":
            self.scenes = [scene for scene in self.scenes if not (60 < int(scene.split('_')[-1]) < 72)]
        elif self.name == "TrainMiniSet":
            self.scenes = [f'scene_{n}' for n in ['01','03']]
        
        self.imgsXscene = {scene: self.read_imgs_per_scene(scene) for scene in self.scenes}
        self.samples = self.generate_all_samples(seed)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

    def read_imgs_per_scene(self, scene_name):
        scene_dir = os.path.join(self.image_root_path, scene_name)
        scene = {}
        # transform = T.ConvertImageDtype(torch.float32)
        for img in os.listdir(scene_dir):
            image = {}
            image['pan'], image['tilt'], rgb, index_view = self.get_pan_and_tilt_from_filename(img)
            
            if rgb != [255, 255, 255]:
                continue
            
            image['img_path'] = os.path.join(self.image_root_path, scene_name, img)

            if index_view not in scene:
                scene[index_view] = []
            scene[index_view].append(image)
        return scene

    def get_pan_and_tilt_from_filename(self, filename):
        filename_parts = filename.split('_')
        pan = int(filename_parts[2]) / 360
        tilt = int(filename_parts[3]) / 360
        rgb = [0, 0, 0]  # Placeholder for RGB values
        rgb[0] = int(filename_parts[4])
        rgb[1] = int(filename_parts[5])
        rgb[2] = int(filename_parts[6])
        index_scene = int(filename_parts[1])
        return pan, tilt, rgb, index_scene

    def generate_all_samples(self, seed):
        all_samples = []
        rng = random.Random(seed)
        for scene in self.scenes:
            for index_view, imgs in self.imgsXscene[scene].items():
                if len(imgs) < 2:
                    continue
                for i in range(len(imgs)):
                    for j in range(len(imgs)):
                        if i != j:
                            all_samples.append((scene, index_view, i, j))
        rng.shuffle(all_samples)
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, index_view, i1, i2 = self.samples[idx]
        lq_obj = self.imgsXscene[scene][index_view][i1]
        gt_obj = self.imgsXscene[scene][index_view][i2]
        
        des_light = torch.Tensor([gt_obj['pan'], gt_obj['tilt']])
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = gt_obj['img_path']
        lq_path = lq_obj['img_path']

        # image range: [0, 1], float32., H W 3
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        # if self.opt['phase'] == 'train':
            # gt_size = self.opt['gt_size']
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # print(img_lq.shape,img_gt.shape,img_lq.min(),img_gt.min(),img_lq.max(),img_gt.max(),lq_path,gt_path)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'des_light': des_light}


class Dataset_surroundings(data.Dataset): 
    def __init__(self, opt, seed=0):
        super(Dataset_surroundings, self).__init__()
        self.opt = opt
        self.image_root_path = "/data/124-1/datasets/RSR_256/"
        # self.scenes = opt["scenes"]

        self.name = opt['name']
        if self.name == "ValSet":
            self.scenes = [f'scene_{n}' for n in range(61, 71)]
        elif self.name == "TrainSet":
            self.scenes = [scene for scene in self.scenes if not (60 < int(scene.split('_')[-1]) < 72)]
        elif self.name == "TrainMiniSet":
            self.scenes = [f'scene_{n}' for n in ['01','03']]
        
        # self.imgsXscene = {scene: self.read_imgs_per_scene(scene) for scene in self.scenes}
        # self.samples = self.generate_all_samples(seed)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

    def read_imgs_per_scene(self, scene_name):
        scene_dir = os.path.join(self.image_root_path, scene_name)
        scene = {}
        # transform = T.ConvertImageDtype(torch.float32)
        for img in os.listdir(scene_dir):
            image = {}
            pan, tilt, rgb, index_view = self.get_pan_and_tilt_from_filename(img)
            
            if rgb != [255, 255, 255]:
                continue

            if index_view not in scene:
                scene[index_view] = {}
            scene[index_view][f"{pan*360}_{tilt*360}"] = os.path.join(self.image_root_path, scene_name, img)
        return scene

    def get_pan_and_tilt_from_filename(self, filename):
        filename_parts = filename.split('_')
        pan = int(filename_parts[2]) / 360
        tilt = int(filename_parts[3]) / 360
        rgb = [0, 0, 0]  # Placeholder for RGB values
        rgb[0] = int(filename_parts[4])
        rgb[1] = int(filename_parts[5])
        rgb[2] = int(filename_parts[6])
        index_scene = int(filename_parts[1])
        return pan, tilt, rgb, index_scene

    def generate_all_samples(self, seed):
        all_samples = []
        rng = random.Random(seed)
        for scene in self.scenes:
            for index_view, imgs in self.imgsXscene[scene].items():
                if len(imgs) < 2:
                    continue
                for i in range(len(imgs)):
                    for j in range(len(imgs)):
                        if i != j:
                            all_samples.append((scene, index_view, i, j))
        rng.shuffle(all_samples)
        return all_samples

    def __len__(self):
        return len(self.samples)

    def load_image(self, path): 
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        img_bytes = self.file_client.get(path, 'img')
        img = imfrombytes(img_bytes, float32=True)
            
        # reshape if it is bigger than 256 
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            
        img = img2tensor(img, bgr2rgb=True, float32=True)
        
        if self.mean is not None or self.std is not None:
            normalize(img, self.mean, self.std, inplace=True)
        
        return img

    def __getitem__(self, idx):
        scene, index_view, i1, i2 = self.samples[idx]
        lq_obj = self.imgsXscene[scene][index_view][i1]
        gt_obj = self.imgsXscene[scene][index_view][i2]
        
        des_light = torch.Tensor([gt_obj['pan'], gt_obj['tilt']])
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = gt_obj['img_path']
        lq_path = lq_obj['img_path']

        img_lq = self.load_image(lq_path)
        img_gt = self.load_image(gt_path)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'des_light': des_light}
