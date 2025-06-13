import numpy as np
import glob,pdb
from PIL import Image
import matplotlib.pyplot as plt
import torch
F = torch.nn.functional

def group_img_by_prefix(img_folder, img_name_list = None):
    if img_name_list is not None:
        img_list = [img_folder + '/' + img_path for img_path in img_name_list]
    else:
        img_list = sorted(glob.glob(img_folder + '/*.png'))
    img_dict = {}
    img_name_list = []
    for img_path in img_list:
        img_name_list.append(img_path.split('/')[-1])
        p1, p2, _ = img_path.split('/')[-1].split('_')
        img = Image.open(img_path)
        scene_name = p1 + '_' + p2
        if scene_name not in img_dict:
            img_dict[scene_name] = []
        else:
            img_dict[scene_name].append(np.array(img))
    for key, val in img_dict.items():
        img_dict[key] = np.stack(val, axis = 0)
    return img_dict, img_name_list

mae_img_dict, img_name_list = group_img_by_prefix('IntrinsicImageDiffusion/out_img') 
new_img_dict, _ = group_img_by_prefix('stable_exp', img_name_list)
img1 = []
img2 = []
for keys, val in new_img_dict.items():
    val = torch.from_numpy(val[:,:,:256]).permute(0,3,1,2) * 1.0
    id_val = mae_img_dict[keys].transpose(0,3,1,2)
    id_val = F.interpolate(torch.from_numpy(id_val) * 1.0, size = (256, 256), mode = 'bilinear')
    img1.append((val - val.min())/(val.max()  - val.min())) 
    img2.append((id_val - id_val.min())/(id_val.max()  - id_val.min()))
img1 = np.stack(img1, axis = 0)
img2 = np.stack(img2, axis = 0)
md1 =np.abs(img1 - img1.mean(axis = 1, keepdims= True)).mean()
md2 =np.abs(img2 - img2.mean(axis = 1, keepdims= True)).mean()
sd1 = img1.std(axis = 1).mean()
sd2 = img2.std(axis = 1).mean()
pdb.set_trace()