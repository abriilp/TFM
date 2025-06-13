import numpy as np
from PIL import Image
import pdb, torch
import glob
import matplotlib.pyplot as plt

our_result = '/net/projects/willettlab/roxie62_proj/latent_intrinsics_release/stable_exp'
li_result = '/net/projects/willettlab/roxie62_proj/latent_intrinsics_release/IntrinsicImageDiffusion/out_img'
gt_path = '/net/projects/willettlab/vdataset/mit_multiview_test'

def get_li_imgs(path):
    img_path_dict = {}
    img_list = sorted(glob.glob(path + '/*.png'))
    loaded_img_list =[]
    for img_path in img_list:
        folder_prefix = ''.join(img_path.split('/')[-1].split('_')[:2])
        if folder_prefix not in img_path_dict:
            img_path_dict[folder_prefix] = []
        img_path_dict[folder_prefix].append(np.array(Image.open(img_path)))
        loaded_img_list.append(img_path.split('/')[-1])
    for key, val in img_path_dict.items():
        img_path_dict[key] = F.interpolate(torch.from_numpy(np.stack(val, axis = 0)).permute(0,3,1,2) * 1.0, size = (256,256), mode = 'bilinear')
    return img_path_dict, loaded_img_list

F = torch.nn.functional
def get_our_imgs(path, img_name_list):
    img_path_dict = {}
    gt_path_dict = {}
    input_path_dict = {}
    for img_name in img_name_list:
        img = np.array(Image.open(path + '/' + img_name))
        folder_prefix = ''.join(img_name.split('_')[:2])
        img_1 = img[:,:256]
        img_2 = img[:,256:]
        if folder_prefix not in img_path_dict:
            img_path_dict[folder_prefix] = []
            input_path_dict[folder_prefix] = []
        img_path_dict[folder_prefix].append(img_1) 
        p1, p2, _ = img_name.split('_')
        gt_img_path = gt_path + '/' + p1 + '_' + p2 + '/thumb.jpg'
        gt_img = Image.open(gt_img_path)
        input_path_dict[folder_prefix].append(img_2)
        gt_path_dict[folder_prefix] = F.interpolate(torch.from_numpy(np.array(gt_img))[None,...].permute(0,3,1,2) * 1.0, size = (256, 256), mode = 'bilinear')
    for key, val in img_path_dict.items():
        img_path_dict[key] = F.interpolate(torch.from_numpy(np.stack(val, axis = 0)).permute(0,3,1,2) * 1.0, size = (256,256), mode = 'bilinear')
    for key, val in input_path_dict.items():
        input_path_dict[key] = F.interpolate(torch.from_numpy(np.stack(val, axis = 0)).permute(0,3,1,2) * 1.0, size = (256,256), mode = 'bilinear')
    return img_path_dict, gt_path_dict, input_path_dict

li_img, img_name = get_li_imgs(li_result)
our_img, gt_img, input_img = get_our_imgs(our_result, img_name)
#for keys in li_img.keys():
for keys in ['everettkitchen14', 'everettkitchen18']:
    fig, ax = plt.subplots(3, 5, figsize = (8, 4))
    for i in range(5):
        ax[0,i].imshow(input_img[keys][i].permute(1,2,0).to(torch.uint8))
        ax[0,i].axis("off")
        Image.fromarray(input_img[keys][i].permute(1,2,0).to(torch.uint8).cpu().data.numpy()).save(f'relight_result/{keys}_{i}.png')
    for i in range(5):
        ax[1,i].imshow(our_img[keys][i].permute(1,2,0).to(torch.uint8))
        ax[1,i].axis("off")
        Image.fromarray(our_img[keys][i].permute(1,2,0).to(torch.uint8).cpu().data.numpy()).save(f'relight_result/{keys}_our_{i}.png')
    for i in range(5):
        ax[2,i].imshow(li_img[keys][i].permute(1,2,0).to(torch.uint8))
        ax[2,i].axis("off")
        Image.fromarray(li_img[keys][i].permute(1,2,0).to(torch.uint8).cpu().data.numpy()).save(f'relight_result/{keys}_li_{i}.png')
    plt.savefig(f'relight_{keys}.png')
    plt.close()
pdb.set_trace()