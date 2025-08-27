# Standard library imports
import argparse
import os
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
import lpips

# Local imports
from utils import *
from unets import UNet
from model_utils import *
from dataset_utils import *
from eval_utils import get_eval_relight_dataloder

# Metrics imports
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class deltaE00():
    def __init__(self, color_chart_area=0):
        super().__init__()
        self.color_chart_area = color_chart_area
        self.kl = 1
        self.kc = 1
        self.kh = 1
    
    def __call__(self, img1, img2):
        """ Compute the deltaE00 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE2000.py
        :param img1: numpy RGB image or pytorch tensor
        :param img2: numpy RGB image or pytorch tensor
        :return: deltaE00
        """
        if type(img1) == torch.Tensor:
            assert img1.shape[0] == 1
            img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
        if type(img2) == torch.Tensor:
            assert img2.shape[0] == 1
            img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Clamp values to [0, 1] for proper Lab conversion
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)
        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)
        # compute deltaE00
        Lstd = np.transpose(img1[:, 0])
        astd = np.transpose(img1[:, 1])
        bstd = np.transpose(img1[:, 2])
        Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
        Lsample = np.transpose(img2[:, 0])
        asample = np.transpose(img2[:, 1])
        bsample = np.transpose(img2[:, 2])
        Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
        Cabarithmean = (Cabstd + Cabsample) / 2
        G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
            Cabarithmean, 7) + np.power(25, 7))))
        apstd = (1 + G) * astd
        apsample = (1 + G) * asample
        Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
        Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
        Cpprod = (Cpsample * Cpstd)
        zcidx = np.argwhere(Cpprod == 0)
        hpstd = np.arctan2(bstd, apstd)
        hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
        hpsample = np.arctan2(bsample, apsample)
        hpsample = hpsample + 2 * np.pi * (hpsample < 0)
        hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
        dL = (Lsample - Lstd)
        dC = (Cpsample - Cpstd)
        dhp = (hpsample - hpstd)
        dhp = dhp - 2 * np.pi * (dhp > np.pi)
        dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
        dhp[zcidx] = 0
        dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
        Lp = (Lsample + Lstd) / 2
        Cp = (Cpstd + Cpsample) / 2
        hp = (hpstd + hpsample) / 2
        hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
        hp = hp + (hp < 0) * 2 * np.pi
        hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
        Lpm502 = np.power((Lp - 50), 2)
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
            0.32 * np.cos(3 * hp + np.pi / 30) \
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
        Sh = 1 + 0.015 * Cp * T
        delthetarad = (30 * np.pi / 180) * np.exp(
            - np.power((180 / np.pi * hp - 275) / 25, 2))
        Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
        RT = - np.sin(2 * delthetarad) * Rc
        klSl = self.kl * Sl
        kcSc = self.kc * Sc
        khSh = self.kh * Sh
        de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                       np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))
        return np.sum(de00) / (np.shape(de00)[0] - self.color_chart_area)


class MetricsCalculator:
    """Class to calculate various image quality metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)  # You can also use 'vgg'
        self.delta_e_fn = deltaE00()

        self.compute_ssim = SSIM(data_range=1.0).to(device)
        self.compute_psnr = PSNR(data_range=1.0).to(device)
        
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        try:
            # Replace compute_psnr with self.compute_psnr
            psnr_val = self.compute_psnr(img1, img2)
            return psnr_val.item()
        except:
            # Fallback to PSNR calculation with formula
            print("Warning: Using fallback PSNR calculation")
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            return psnr.item()
        
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        # Replace compute_ssim with self.compute_ssim
        ssim_val = self.compute_ssim(img1, img2)
        return ssim_val.item()
    
    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS between two images"""
        # LPIPS expects images in range [-1, 1]
        img1_norm = img1 * 2.0 - 1.0
        img2_norm = img2 * 2.0 - 1.0
        lpips_val = self.lpips_fn(img1_norm, img2_norm)
        return lpips_val.item()
    
    def calculate_delta_e(self, img1, img2):
        """Calculate Delta E between two images"""
        # Convert tensors to proper format for deltaE calculation
        if len(img1.shape) == 4 and img1.shape[0] == 1:
            img1 = img1.squeeze(0)
        if len(img2.shape) == 4 and img2.shape[0] == 1:
            img2 = img2.squeeze(0)
        
        # Convert from [-1, 1] to [0, 1] if needed
        if img1.min() < 0:
            img1 = (img1 + 1) / 2
        if img2.min() < 0:
            img2 = (img2 + 1) / 2
            
        delta_e_val = self.delta_e_fn(img1.unsqueeze(0), img2.unsqueeze(0))
        return delta_e_val
    
    def calculate_all_metrics(self, pred_img, gt_img):
        """Calculate all metrics for a pair of images"""
        metrics = {}
        
        # Ensure images are in [0, 1] range for most metrics
        if pred_img.min() < 0:
            pred_img_01 = (pred_img + 1) / 2
        else:
            pred_img_01 = pred_img
            
        if gt_img.min() < 0:
            gt_img_01 = (gt_img + 1) / 2
        else:
            gt_img_01 = gt_img
        
        metrics['psnr'] = self.calculate_psnr(pred_img_01, gt_img_01)
        metrics['ssim'] = self.calculate_ssim(pred_img_01, gt_img_01)
        metrics['lpips'] = self.calculate_lpips(pred_img_01, gt_img_01)
        metrics['delta_e'] = self.calculate_delta_e(pred_img_01, gt_img_01)
        
        return metrics


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract args from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
    else:
        # Default args if not saved in checkpoint
        print("Warning: args not found in checkpoint, using defaults")
        args = argparse.Namespace(
            affine_scale=5e-3,
            enable_normals_decoder_loss=True
        )
    
    # Initialize model
    model = UNet(
        img_resolution=256, 
        in_channels=3, 
        out_channels=3,
        num_blocks_list=[1, 2, 2, 4, 4, 4], 
        attn_resolutions=[0], 
        model_channels=32,
        channel_mult=[1, 2, 4, 4, 8, 16], 
        affine_scale=float(getattr(args, 'affine_scale', 5e-3))
    )
    
    model.to(device)
    
    # Load state dict
    if 'ema_state_dict' in checkpoint:
        # Use EMA weights if available
        state_dict = checkpoint['ema_state_dict']
        print("Using EMA weights")
    else:
        state_dict = checkpoint['state_dict']
        print("Using regular weights")
    
    # Remove 'module.' prefix if present (from DistributedDataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch}")
    
    return model, args


def setup_test_dataset(dataset_name, data_path, args):
    """Setup test dataset"""
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    if dataset_name == 'mit':
        test_dataset = MIT_Dataset(
            data_path, 
            [None, transform_test], 
            eval_mode=True
        )
    elif dataset_name == 'rsr_256':
        test_dataset = RSRDataset(
            root_dir=data_path, 
            img_transform=[None, transform_test], 
            use_all_scenes=True,
            is_validation=True,  # 
        )
    elif dataset_name == 'iiw':
        test_dataset = IIW2(
            root=data_path, 
            img_transform=[None, transform_test], 
            split='test', 
            return_two_images=True
        )
    elif dataset_name == 'rlsid':
        enable_normals = getattr(args, 'enable_normals_decoder_loss', False)
        test_dataset = RLSIDDataset(
            root_dir=data_path,
            validation_scenes=['00012', '00014', '00020', '00022', '00029', '00030', '00036', '00042', '00043', '00049', '00054', '00058', '00060', '00070', '00076', '00081', '00087', '00098', '00103', '00112', '00113', '00115', '00121', '00124', '00131', '00137', '00138', '00145', '00156', '00157', '00164', '00175', '00176', '00177', '00178', '00179', '00181', '00190', '00198', '00200', '00205', '00206', '00209', '00217', '00219', '00230', '00240', '00244', '00246', '00247', '00255', '00260', '00262', '00276', '00281', '00282', '00283', '00286', '00291', '00306', '00312', '00317', '00320', '00321', '00325', '00332', '00334', '00341', '00343', '00344', '00346', '00349', '00352', '00356', '00361', '00364', '00369', '00371', '00372', '00373', '00380', '00384', '00385', '00392', '00400', '00402', '00411', '00413', '00414', '00419', '00422', '00424', '00428', '00431', '00435', '00437', '00442', '00443', '00448', '00457', '00459', '00461', '00466', '00476', '00477', '00491', '00494', '00495', '00498', '00505', '00506', '00515', '00527', '00529', '00533', '00542', '00552', '00556', '00558', '00566', '00569', '00573', '00576', '00577', '00581', '00586', '00587', '00594', '00600', '00603', '00604', '00605', '00608', '00619', '00636', '00648', '00663', '00669', '00673', '00678', '00687', '00697', '00702', '00704', '00707', '00713', '00726', '00727', '00731', '00736', '00743', '00744', '00749', '00760', '00768', '00772', '00776', '00777', '00785', '00788', '00790', '00792', '00794', '00796', '00802', '00806', '00820', '00821', '00824', '00828', '00832', '00847', '00851', '00852', '00855', '00857', '00867', '00871', '00879', '00880', '00883', '00900', '00902', '00904', '00905', '00909', '00921', '00923', '00931', '00935', '00942', '00946', '00961', '00971', '00981', '00986', '00988', '01000', '01006', '01009', '01014', '01018', '01020', '01024', '01031', '01035', '01036', '01038', '01040', '01041', '01044', '01059', '01060', '01064', '01065', '01066', '01067', '01069', '01076', '01085', '01086', '01088', '01090', '01095', '01097', '01113', '01115', '01124', '01139', '01148', '01150', '01152', '01154', '01155', '01158', '01163', '01166', '01168', '01178', '01182', '01191', '01197', '01200', '01201', '01207', '01210', '01211', '01213', '01214', '01218', '01221', '01223', '01229', '01233', '01236', '01244', '01250', '01253', '01258', '01263', '01270', '01278', '01293', '01295', '01301', '01311', '01314', '01323', '01329', '01338', '01339', '01342', '01343', '01348', '01356', '01363', '01367', '01370', '01384', '01393', '01398', '01404', '01405', '01410', '01416', '01419', '01423', '01424', '01426', '01431', '01433', '01438', '01444', '01451', '01460', '01463', '01466', '01468', '01469', '01479', '01481', '01485', '01487', '01492', '01498', '01505', '01519', '01520', '01525', '01526', '01537', '01547', '01561', '01566', '01570', '01577', '01579', '01582', '01583', '01585', '01588', '01589', '01593', '01594', '01599', '01603', '01604', '01605', '01606', '01610', '01611', '01620', '01622', '01627', '01637', '01646', '01649', '01650', '01652', '01657', '01658', '01668', '01670', '01672', '01678', '01680', '01681', '01686', '01694', '01710', '01711', '01716', '01721', '01726', '01732', '01733', '01735', '01737', '01740', '01744', '01748', '01754', '01766', '01767', '01773', '01779', '01781', '01786', '01790', '01793', '01796', '01804', '01806', '01807', '01809', '01811', '01821', '01831', '01836', '01842', '01845', '01851', '01852', '01858', '01865', '01877', '01887', '01895', '01896', '01901', '01903', '01906', '01912', '01922', '01924', '01929', '01931', '01932', '01935', '01940', '01941', '01946', '01964', '01966', '01969', '01976', '01981', '01985', '01987', '01990', '01993', '01995', '02007', '02047', '02051', '02052', '02057', '02062', '02069', '02073', '02076', '02081', '02083', '02100', '02101', '02102', '02110', '02119', '02125', '02129', '02130', '02131', '02135', '02136', '02145', '02150', '02163', '02167', '02189', '02202', '02216', '02219', '02229', '02230', '02233', '02234', '02235', '02236', '02247', '02256', '02258', '02261', '02263', '02265', '02266', '02267', '02273', '02278', '02287', '02288', '02291', '02307', '02311', '02322', '02335', '02343', '02352', '02355', '02356', '02359', '02373', '02374', '02380', '02385', '02387', '02392', '02396', '02400', '02407', '02416', '02417', '02418', '02424', '02425', '02429', '02441', '02452', '02456', '02491', '02501', '02503', '02513', '02515', '02517', '02518', '02520', '02526', '02530', '02537', '02539', '02542', '02543', '02548', '02550', '02561', '02562', '02569', '02580', '02581', '02584', '02591', '02594', '02604', '02605', '02607', '02624', '02631', '02632', '02634', '02642', '02645', '02653', '02660', '02661', '02669', '02670', '02672', '02676', '02678', '02684', '02694', '02696', '02697', '02701', '02709', '02711', '02713', '02734', '02736', '02737', '02743', '02752', '02767', '02770', '02776', '02779', '02783', '02784', '02790', '02791', '02793', '02797', '02808', '02810', '02814', '02820', '02826', '02828', '02829', '02833', '02834', '02839', '02852', '02854', '02856', '02858', '02859', '02860', '02863', '02865', '02870', '02874', '02886', '02888', '02894', '02900', '02905', '02906', '02911', '02913', '02917', '02922', '02928', '02942', '02945', '02946', '02952', '02958', '02973', '02975', '02976', '02977', '02982', '02995', '02996', '02997', '03001', '03006', '03013', '03015', '03018', '03024', '03038', '03048', '03056', '03057', '03061', '03071', '03072', '03078', '03084', '03087', '03099', '03101', '03116', '03118', '03120', '03128', '03133', '03143', '03145', '03149', '03150', '03152', '03154', '03157', '03158', '03161', '03162', '03164', '03169', '03170', '03172', '03184', '03191', '03192', '03207', '03208', '03213', '03219', '03220', '03237', '03241', '03247', '03251', '03256', '03258', '03271', '03273', '03283', '03288', '03290', '03297', '03302', '03307', '03308', '03333', '03340', '03343', '03345', '03352', '03362', '03366', '03368', '03379', '03391', '03395', '03397', '03398', '03416', '03419', '03430', '03431', '03440', '03447', '03450', '03459', '03467', '03469', '03489', '03491', '03492', '03501', '03503', '03507', '03511', '03518', '03519', '03520', '03539', '03547', '03556', '03569', '03571', '03574', '03595', '03604', '03617', '03621', '03631', '03638', '03640', '03647', '03648', '03656', '03662', '03668', '03670', '03672', '03675', '03680', '03682', '03690', '03691', '03693', '03694', '03706', '03713', '03716', '03717', '03724', '03731', '03733', '03734', '03747', '03758', '03781', '03782', '03791', '03794', '03795', '03801', '03802', '03813', '03815', '03817', '03821', '03826', '03830', '03832', '03856', '03861', '03864', '03867', '03873', '03877', '03883', '03889', '03891', '03892', '03895', '03896', '03900', '03901', '03904', '03909', '03912', '03915', '03920', '03925', '03944', '03948', '03951', '03956', '03960', '03967', '03968', '03970', '03981', '03986', '03997', '03998', '04002', '04004', '04016', '04019', '04022', '04023', '04027', '04032', '04041', '04046', '04049', '04050', '04051', '04054', '04055', '04059', '04060', '04064', '04072', '04074', '04075', '04077', '04082', '04087', '04096', '04097', '04104', '04105', '04115', '04116', '04117', '04120', '04122', '04128', '04130', '04140', '04142', '04146', '04149', '04152', '04156', '04165', '04170', '04175', '04180', '04191', '04192', '04202', '04212', '04213', '04222', '04226', '04232', '04235', '04237', '04254', '04258', '04259', '04267', '04279', '04283', '04287', '04289', '04290', '04291', '04295', '04298', '04313', '04317', '04319', '04326', '04328', '04329', '04363', '04364', '04365', '04368', '04372', '04373', '04376', '04386', '04387', '04389', '04390', '04396', '04398', '04402', '04405', '04408', '04417', '04438', '04439', '04446', '04456', '04460', '04464', '04466', '04467', '04474', '04476', '04489', '04490', '04494', '04498', '04507', '04518', '04519', '04520', '04522', '04536', '04546', '04550', '04560', '04581', '04586', '04589', '04597', '04598', '04601', '04604', '04607', '04608', '04616', '04622', '04624', '04626', '04637', '04641', '04644', '04645', '04650', '04656', '04660', '04669', '04672', '04676', '04678', '04681', '04685', '04689', '04690', '04693', '04694', '04698', '04706', '04707', '04715', '04728', '04729', '04742', '04744', '04751', '04752', '04753', '04756', '04758', '04760', '04767', '04794', '04798', '04800', '04801', '04803', '04813', '04815', '04818', '04825', '04827', '04832', '04842', '04861', '04862', '04875', '04881', '04889', '04890', '04895', '04902', '04923', '04934', '04943', '04950', '04955', '04961', '04972', '04977', '04978', '04980', '04987', '04995', '04998', '05005', '05006', '05013', '05016', '05019', '05025', '05027', '05029', '05036', '05037', '05042', '05046', '05056', '05062', '05063', '05066', '05068', '05081', '05089', '05091', '05092', '05093', '05095', '05097', '05099', '05103', '05108', '05111', '05112', '05116', '05117', '05124', '05125', '05127', '05128', '05135', '05136', '05138', '05139', '05144', '05147', '05148', '17219', '17235', '17239', '17245', '17250', '17258', '17263', '17275', '17276', '17278', '17286', '17287', '17291', '17293', '17294', '17297', '17307', '17309', '17313', '17318', '17324', '17329', '17333', '17334', '17341', '17347', '17359', '17362', '17364', '17367', '17371', '17378', '17385', '17386', '17389', '17402', '17417', '17418', '17421', '17425', '17439', '17441', '17446', '17447', '17455', '17457', '17462', '17466', '17475', '17477', '17483', '17485', '17497', '17506', '17514', '17518', '17521', '17527', '17528', '17538', '17539', '17541', '17546', '17547', '17550', '17554', '17562', '17564', '17565', '17566', '17580', '17583', '17597', '17599', '17601', '17602', '17603', '17606', '17611', '17612', '17616', '17621', '17623', '17635', '17637', '17656', '17662', '17665', '17666', '17674', '17681', '17682', '17683', '17690', '17693', '17696', '17704', '17705', '17719', '17722', '17727', '17729', '17730', '17741', '17745', '17746', '17747', '17751', '17753', '17756', '17757', '17761', '17763', '17765', '17768', '17770', '17771', '17784', '17786', '17788', '17800', '17801', '17805', '17806', '17807', '17809', '17815', '17821', '17828', '17829', '17831', '17836', '17837', '17851', '17855', '17863', '17865', '17873', '17876', '17877', '17883', '17884', '17893', '17894', '17897', '17906', '17908', '17911', '17919', '17932', '17933', '17937', '17946', '17947', '17949', '17953', '17957', '17962', '17968', '17974', '17987', '17990', '17994', '17995', '18002', '18007', '18008', '18011', '18018', '18022', '18026', '18028', '18033', '18037', '18041', '18043', '18045', '18048', '18055', '18063', '18075', '18078', '18080', '18082', '18092', '18094', '18097', '18104', '18106', '18107', '18110', '18111', '18112', '18126', '18128', '18152', '18159', '18162', '18171', '18172', '18178', '18179', '18180', '18181', '18195', '18197', '18206', '18208', '18209', '18211', '18221', '18228', '18232', '18237', '18238', '18239', '18241', '18253', '18270', '18280', '18288', '18292', '18294', '18299', '18301', '18302', '18304', '18312', '18319', '18320', '18323', '18327', '18335', '18337', '18342', '18353', '18354', '18357', '18358', '18360', '18364', '18365', '18375', '18377', '18381', '18382', '18383', '18393', '18394', '18401', '18405', '18414', '18417', '18419', '18431', '18434', '18436', '18441', '18446', '18447', '18451', '18453', '18459', '18460', '18462', '18466', '18468', '18477', '18482', '18490', '18492', '18493', '18496', '18499', '18501', '18511', '18518', '18532', '18542', '18543', '18546', '18551', '18556', '18561', '18566', '18570', '18573', '18582', '18587', '18596', '18601', '18602', '18616', '18617', '18620', '18632', '18648', '18653', '18655', '18660', '18661', '18665', '18676', '18680', '18681', '18683', '18684', '18692', '18694', '18698', '18702', '18710', '18712', '18716', '18718', '18725', '18734', '18739', '18741', '18743', '18754', '18756', '18759', '18761', '18771', '18777', '18778', '18780', '18783', '18786', '18796', '18797', '18802', '18813', '18814', '18815', '18831', '18835', '18836', '18841', '18849', '18850', '18853', '18857', '18859', '18865', '18872', '18878', '18881', '18887', '18892', '18900', '18911', '18912', '18916', '18922', '18928', '18940', '18949', '18959', '18960', '18962', '18981', '18986', '18988', '18989', '18994', '19001', '19008', '19017', '19026', '19034', '19038', '19040', '19045', '19046', '19047', '19050', '19052', '19054', '19058', '19059', '19060', '19066', '19071', '19074', '19077', '19079', '19088', '19096', '19098', '19101', '19102', '19108', '19111', '19114', '19120', '19121', '19125', '19128', '19130', '19136', '19147', '19151', '19154', '19155', '19165', '19168', '19170', '19181', '19189', '19194', '19196', '19202', '19204', '19207', '19211', '19214', '19220', '19225', '19229', '19235', '19237', '19240', '19251', '19257', '19259', '19266', '19267', '19268', '19274', '19277', '19285', '19297', '19298', '19304', '19309', '19313', '19320', '19324', '19326', '19327', '19331', '19335', '19345', '19347', '19348', '19357', '19359', '19364', '19370', '19373', '19374', '19375', '19377', '19383', '19384', '19386', '19388', '19398', '19400', '19405', '19408', '19409', '19419', '19421', '19423', '19426', '19428', '19431', '19432', '19437', '19440', '19441', '19443', '19448', '19451', '19452', '19455', '19459', '19460', '19464', '19467', '19475', '19480', '19483', '19487', '19496', '19498', '19512', '19520', '19528', '19532', '19534', '19535', '19542', '19550', '19555', '19561', '19572', '19573', '19579', '19581', '19586', '19587', '19591', '19604', '19606', '19636', '19650', '19652', '19654', '19655', '19658', '19660', '19665', '19673', '19675', '19690', '19708', '19711', '19718', '19729', '19733', '19745', '19749', '19755', '19756', '19757', '19764', '19772', '19776', '19778', '19786', '19787', '19789', '19790', '19792', '19793', '19796', '19800', '19801', '19805', '19820', '19823', '19829', '19840', '19845', '19857', '19866', '19871', '19879', '19881', '19890', '19904', '19913', '19918', '19923', '19931', '19932', '19936', '19937', '19941', '19945', '19956', '19962', '19965', '19966', '19976', '19978', '19979', '19984', '19989', '19993', '19994', '19998', '20002', '20003', '20007', '20011', '20012', '20014', '20018', '20019', '20020', '20032', '20034', '20037', '20039', '20040', '20041', '20052', '20065', '20074', '20084', '20085', '20086', '20087', '20088', '20098', '20100', '20117', '20118', '20129', '20130', '20144', '20161', '20166', '20170', '20179', '20183', '20186', '20188', '20189', '20215', '20237', '20251', '20253', '20255', '20257', '20265', '20266', '20269', '20274', '20275', '20276', '20280', '20283', '20284', '20285', '20286', '20289', '20290', '20303', '20307', '20313', '20329', '20339', '20342', '20348', '20352', '20366', '20367', '20374', '20375', '20376', '20382', '20389', '20391', '20393', '20396', '20399', '20402', '20403', '20405', '20414', '20416', '20424', '20435', '20438', '20441', '20449', '20452', '20461', '20463', '20465', '20468', '20472', '20477', '20483', '20490', '20494', '20495', '20504', '20511', '20515', '20516', '20518', '20520', '20523', '20526', '20529', '20530', '20535', '20541', '20561', '20566', '20569', '20573', '20579', '20595', '20599', '20605', '20606', '20608', '20614', '20618', '20621', '20629', '20631', '20643', '20644', '20662', '20669', '20677', '20679', '20682', '20688', '20691', '20692', '20701', '20709', '20711', '20713', '20721', '20724', '20729', '20737', '20739', '20740', '20741', '20743', '20755', '20768', '20776', '20782', '20787', '20793', '20800', '20803', '20804', '20813', '20816', '20817', '20820', '20822', '20829', '20835', '20837', '20838', '20846', '20849', '20858', '20866', '20874', '20888', '20889', '20892', '20898', '20919', '20921', '20923', '20929', '20933', '20936', '20943', '20944', '20946', '20951', '20958', '20962', '20964', '20966', '20976', '20977', '20988', '20996', '21010', '21011', '21012', '21026', '21036', '21042', '21051', '21053', '21054', '21060', '21065', '21068', '21078', '21080', '21082', '21083', '21096', '21102', '21106', '21112', '21115', '21116', '21131', '21133', '21138', '21149', '21155', '21158', '21162', '21164', '21172', '21175', '21193', '21202', '21203', '21213', '21214', '21218', '21219', '21223', '21224', '21233', '21235', '21239', '21244', '21247', '21251', '21253', '21256', '21260', '21265', '21270', '21273', '21276', '21277', '21284', '21286', '21289', '21293', '21294', '21304', '21310', '21313', '21314', '21318', '21333', '21340', '21348', '21353', '21354', '21363', '21369', '21373', '21380', '21381', '21383', '21387', '21392', '21394', '21403', '21406', '21407', '21411', '21412', '21425', '21426', '21428', '21435', '21439', '21442', '21444', '21447', '21449', '21455', '21457', '21468', '21471', '21483', '21485', '21488', '21489', '21495', '21496', '21497', '21504', '21505', '21506', '21507', '21510', '21512', '21516', '21518', '21519', '21521', '21530', '21537', '21540', '21542', '21544', '21550', '21561', '21565', '21570', '21572', '21577', '21579', '21590', '21594', '21595', '21597', '21603', '21605', '21607', '21613', '21614', '21623', '21626', '21632', '21639', '21640', '21644', '21645', '21648', '21649', '21652', '21655', '21659', '21663', '21664', '21666', '21673', '21674', '21683', '21686', '21690', '21695', '21702', '21703', '21705', '21709', '21713', '21717', '21724', '21728', '21742', '21743', '21752', '21758', '21770', '21772', '21783', '21784', '21793', '21805', '21809', '21821', '21827', '21838', '21843', '21844', '21846', '21851', '21872', '21875', '21877', '21878', '21887', '21891', '21896', '21903', '21907', '21915', '21916', '21919', '21920', '21922', '21929', '21936', '21937', '21946', '21947', '21949', '21951', '21954', '21955', '21957', '21958', '21960', '21964', '21968', '21971', '21973', '21979', '21981', '21986', '21988', '21989', '21993', '21998', '22004', '22005', '22006', '22010', '22045', '22048', '22056', '22062', '22063', '22069', '22074', '22079', '22095', '22097', '22100', '22105', '22106', '22111', '22122', '22123', '22124', '22127', '22128', '22129'],
            img_transform=[None, transform_test],
            is_validation=True,
            enable_normals=enable_normals
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return test_dataset


def run_inference(model, test_loader, metrics_calculator, device, save_images=False, output_dir=None, save_every=50):
    """Run inference and calculate metrics"""
    model.eval()
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'delta_e': []
    }
    
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Running inference")):
            if len(batch) >= 3:
                input_img, ref_img, gt_img = batch[:3]
            else:
                input_img, ref_img = batch
                gt_img = ref_img  # Use ref as ground truth if no separate GT
            
            input_img = input_img.to(device)
            ref_img = ref_img.to(device)
            gt_img = gt_img.to(device)
            
            batch_size = input_img.shape[0]
            
            for b in range(batch_size):
                # Extract single images
                input_single = input_img[b:b+1]
                ref_single = ref_img[b:b+1]
                gt_single = gt_img[b:b+1]
                
                # Forward pass
                try:
                    intrinsic_input, extrinsic_input = model(input_single, run_encoder=True)
                    intrinsic_ref, extrinsic_ref = model(ref_single, run_encoder=True)
                    
                    # Generate relit image
                    if hasattr(model, 'enable_normals_decoder') and model.enable_normals_decoder:
                        relit_img, _ = model([intrinsic_input, extrinsic_ref], run_encoder=False, decode_mode='both')
                    else:
                        relit_img = model([intrinsic_input, extrinsic_ref], run_encoder=False, decode_mode='main')
                    
                    relit_img = relit_img.float()
                    
                    # Calculate metrics
                    sample_metrics = metrics_calculator.calculate_all_metrics(relit_img, gt_single)
                    
                    # Accumulate metrics
                    for key in all_metrics.keys():
                        all_metrics[key].append(sample_metrics[key])
                    
                    total_samples += 1
                    
                    # Save images if requested
                    if save_images and output_dir is not None and total_samples % save_every == 0:
                        save_comparison_images(
                            input_single, ref_single, relit_img, gt_single,
                            output_dir, f"sample_{i:04d}_{b:02d}", sample_metrics
                        )
                        
                except Exception as e:
                    print(f"Error processing sample {i}_{b}: {e}")
                    continue
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics.keys():
        if len(all_metrics[key]) > 0:
            avg_metrics[key] = {
                'mean': np.mean(all_metrics[key]),
                'std': np.std(all_metrics[key]),
                'median': np.median(all_metrics[key]),
                'min': np.min(all_metrics[key]),
                'max': np.max(all_metrics[key])
            }
        else:
            avg_metrics[key] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0
            }
    
    print(f"Processed {total_samples} samples successfully")
    return avg_metrics, all_metrics


def save_comparison_images(input_img, ref_img, relit_img, gt_img, output_dir, filename, metrics):
    """Save comparison images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy and denormalize
    def tensor_to_numpy(tensor):
        img = tensor.squeeze().cpu().detach()
        img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
        img = torch.clamp(img, 0, 1)
        return img.permute(1, 2, 0).numpy()
    
    input_np = tensor_to_numpy(input_img)
    ref_np = tensor_to_numpy(ref_img)
    relit_np = tensor_to_numpy(relit_img)
    gt_np = tensor_to_numpy(gt_img)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(input_np)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(ref_np)
    axes[1].set_title('Reference')
    axes[1].axis('off')
    
    axes[2].imshow(relit_np)
    axes[2].set_title('Relit')
    axes[2].axis('off')
    
    axes[3].imshow(gt_np)
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')
    
    # Add metrics as subtitle
    metrics_text = f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f} | ΔE: {metrics['delta_e']:.2f}"
    fig.suptitle(metrics_text, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inference script for relighting models')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['mit', 'rsr_256', 'iiw', 'rlsid'], 
                        default='mit', help='Dataset to test on')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the test dataset')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--save_images', action='store_true',
                        help='Save comparison images')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for debugging)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save images every N samples (default: 50, set to 1 to save all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, model_args = load_model_from_checkpoint(args.checkpoint_path, args.device)
    
    # Setup test dataset
    test_dataset = setup_test_dataset(args.dataset, args.data_path, model_args)
    
    # Limit samples if specified
    if args.max_samples is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(args.max_samples, len(test_dataset))))
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset: {args.dataset}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Checkpoint: {args.checkpoint_path}")
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(args.device)
    
    # Run inference
    start_time = time.time()
    avg_metrics, all_metrics = run_inference(
        model, test_loader, metrics_calculator, args.device,
        save_images=args.save_images, 
        output_dir=args.output_dir if args.save_images else None,
        save_every=args.save_every
    )
    end_time = time.time()
    
    # Print results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {os.path.basename(args.checkpoint_path)}")
    print(f"Total samples: {len(all_metrics['psnr'])}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Time per sample: {(end_time - start_time) / len(all_metrics['psnr']):.3f} seconds")
    print("\nMetrics Summary:")
    print("-" * 60)
    
    for metric_name in ['psnr', 'ssim', 'lpips', 'delta_e']:
        stats = avg_metrics[metric_name]
        print(f"{metric_name.upper():>8}: {stats['mean']:8.4f} ± {stats['std']:6.4f} "
              f"(min: {stats['min']:6.4f}, max: {stats['max']:6.4f}, median: {stats['median']:6.4f})")
    
    # Save detailed results
    results = {
        'checkpoint_path': args.checkpoint_path,
        'dataset': args.dataset,
        'data_path': args.data_path,
        'total_samples': len(all_metrics['psnr']),
        'processing_time_seconds': end_time - start_time,
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'model_args': vars(model_args) if hasattr(model_args, '__dict__') else str(model_args)
    }
    
    results_file = os.path.join(args.output_dir, f"inference_results_{args.dataset}.json")
    with open(results_file, 'w') as f:
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return obj
        
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(v) for v in data]
            else:
                return convert_numpy(data)
        
        json.dump(clean_for_json(results), f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    if args.save_images:
        print(f"Comparison images saved to: {args.output_dir}")
    
    print("="*60)


if __name__ == '__main__':
    main()