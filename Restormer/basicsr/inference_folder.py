import math
import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import imageio
import random

from basicsr.models import create_model
from basicsr.utils import set_random_seed, tensor2img
from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.data.paired_image_dataset import Dataset_surroundings

PARA_NOR = {
    'pan_a': 0.5,
    'pan_b': 0.5,
    'tilt_a': 1.0,
    'tilt_b': 0.0,
}


def convert_K_to_RGB(colour_temperature):
    """
    Converts from K to RGB, algorithm courtesy of
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    # range check
    if colour_temperature < 1000:
        colour_temperature = 1000
    elif colour_temperature > 40000:
        colour_temperature = 40000

    tmp_internal = colour_temperature / 100.0

    # red
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red

    # green
    if tmp_internal <= 66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green

    # blue
    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue

    return red, green, blue


def light_condition2tensor(pan_deg, tilt_deg, color, color_type = "temperature"):
    """
    transform pan, tilt, color into the tensors for input.
    :param pan: in deg
    :param tilt: in deg
    :param color: in temperature
    :return: tensor size(7)
    """
    factor_deg2rad = math.pi / 180.0
    pan = float(pan_deg) * factor_deg2rad
    tilt = float(tilt_deg) * factor_deg2rad

    # transform light position to cos and sin
    light_position = [math.cos(pan), math.sin(pan), math.cos(tilt), math.sin(tilt)]
    # normalize the light position to [0, 1]
    light_position[:2] = [x * PARA_NOR['pan_a'] + PARA_NOR['pan_b'] for x in light_position[:2]]
    light_position[2:] = [x * PARA_NOR['tilt_a'] + PARA_NOR['tilt_b'] for x in light_position[2:]]
    # transform light temperature to RGB, and normalize it.
    if color_type == "temperature":
        color_temp = int(color)
        light_color = list(map(lambda x: x / 255.0, convert_K_to_RGB(color_temp)))
    else:
        light_color = [x/255 for x in color]
    light_position_color = light_position + light_color
    return torch.tensor(light_position_color)




class ImageDial():
    def __init__(self, dial_img_name):
        dial_img = Image.open(dial_img_name).convert('RGB')
        dial_img = np.array(dial_img)
        scale_ratio = 256 / 880
        self.dial_img = cv2.resize(dial_img, None, fx=scale_ratio, fy=scale_ratio,
                                   interpolation=cv2.INTER_CUBIC)
        self.dial_center = [int(523 * scale_ratio), int(1320 * scale_ratio)]
        self.radius = 400 * scale_ratio
        # original size is (281, 768, 3), we need to fill to (288, 768, 3) to satisfy macro_block_size=16 in imageio
        self.h_pad, self.w_pad = tuple([x if x % 16 == 0 else x + 16 - x % 16 for x in self.dial_img.shape[:2]])

    def insert_img(self, img_input, img_relit, pan, tilt):
        merged_img = np.copy(self.dial_img)
        # merged_img[-256:, :256, :] = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
        # merged_img[-256:, -256:, :] = cv2.cvtColor(img_relit, cv2.COLOR_RGB2BGR)
        merged_img[-256:, :256, :] = img_input
        merged_img[-256:, -256:, :] = img_relit
        # plot the point of pan and tilt.
        length = math.sin(tilt/180*math.pi) / math.sin(50/180*math.pi) * self.radius
        position = (int(self.dial_center[1] + math.sin(pan/180*math.pi) * length),
                    int(self.dial_center[0] + math.cos(pan/180*math.pi) * length))
        # cv2.circle(merged_img, position, 3, (0, 0, 255), -1)
        cv2.circle(merged_img, position, 3, (255, 0, 0), -1)
        # add padding
        white_padding = np.full((self.h_pad, self.w_pad, 3), 255, dtype=np.uint8)
        white_padding[:merged_img.shape[0], :merged_img.shape[1]] = merged_img
        return white_padding


def generate_path(points, steps, length):
    loop_path = []
    for i in range(len(points)-1):
        point_a = points[i]
        point_b = points[i+1]
        this_step = [step if point_b[k] > point_a[k] else -step for k, step in enumerate(steps)]
        lists = [np.arange(point_a[k], point_b[k], this_step[k]) for k in range(len(point_a))]
        for j in range(max([len(lst) for lst in lists])):
            loop_path.append([lists[k][j] if j < len(lists[k]) else point_b[k] for k in range(len(lists))])
    path = [loop_path[i % len(loop_path)] for i in range(length)]

    return path


def create_pan_tilt_temperature_seq(length, seq_type):
    # pan, tilt, temperature
    # default_steps = [2, 1, 100]
    default_start = [90, 30, 4100]
    if seq_type == "cycle_tilt":
        points = [[default_start[0], 40.0, default_start[2]],
                  [default_start[0], 0, default_start[2]],
                  [-default_start[0], 40, default_start[2]],
                  [-default_start[0], 0, default_start[2]],
                  [default_start[0], 40.0, default_start[2]]]
        steps = [float('inf'), 1, float('inf')]
    elif seq_type == "cycle_pan":
        points = [[0, default_start[1], default_start[2]],
                  [360, default_start[1], default_start[2]], ]
        steps = [2, float('inf'), float('inf')]
    elif seq_type == "cycle_temperature":
        points = [[default_start[0], default_start[1], 2300],
                  [default_start[0], default_start[1], 6400],]
        steps = [float('inf'), float('inf'), 100]
    else:
        raise Exception("seq_type wrong!")

    sequence = generate_path(points, steps, length)
    return sequence


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    return opt



if __name__ == '__main__':
    scene = "scene_01"
    
    output_pred = "/home/mpilligua/Restormer/BestModelOut_val/"
    os.makedirs(output_pred, exist_ok=True)
    
    output_gt = "/home/mpilligua/Restormer/BestModelOut_GT_val/"
    os.makedirs(output_gt, exist_ok=True)
    
    pan_tilt = [[0, 0], [90, 27], [135, 35], [180, 27], [225, 35], [315, 35], [0, 27], [45, 35]]
    opt = parse_options(is_train=True)
    dataset = Dataset_surroundings(opt['datasets']['val'])
    
    for ckpt in os.listdir('/home/mpilligua/Restormer/experiments/Finetuning_RSR_compiled_Restormer/models/'): 
        if (int(ckpt.split('_')[-1].split('.')[0]) not in [2000]): #1000, 5000, 250000]):
            continue

        dial_img_name = "/home/mpilligua/Restormer/basicsr/utils/pan_tilt_dial.png"
        img_dial = ImageDial(dial_img_name)

        data = {}
        opt['path']['pretrain_network_g'] = '/home/mpilligua/Restormer/experiments/Finetuning_RSR_compiled_Restormer/models/' + ckpt
        model = create_model(opt)
        model.net_g.eval()  # set model to eval mode
        
        for folder in os.listdir(f"/data/124-1/datasets/RSR_256"):  
            scene = dataset.read_imgs_per_scene(folder)
            for index_view in scene:
                for pan_tilt in scene[index_view].keys():
                    pan, tilt = pan_tilt.split("_")
                    pan, tilt = int(float(pan)), int(float(tilt))
                    dir1 = scene[index_view][pan_tilt]
                    
                    if not (60 < int(folder.split("_")[1]) < 71):
                        continue 
                    
                    if os.path.exists(os.path.join(output_pred, os.path.basename(dir1) + f"_{pan}_{tilt}.png")):
                        if os.path.exists(os.path.join(output_gt, os.path.basename(dir1) + f"_{pan}_{tilt}.png")):
                            print(f"Skipping {dir1} with input {pan} and {tilt}, already processed.")
                            continue
                    
                    for pan_tilt_gt in scene[index_view].keys():
                        pan_gt, tilt_gt = pan_tilt_gt.split("_")
                        pan_gt, tilt_gt = int(float(pan_gt)), int(float(tilt_gt))
                        dir_gt = scene[index_view][pan_tilt_gt]
                
                        print(f"Processing {dir1} with input {pan} and {tilt} to {pan_gt} and {tilt_gt}")
            
                        input_name = dir1
                        image = dataset.load_image(input_name)
                        
                        data2feed = {'lq': image.unsqueeze(0),
                                'des_light': torch.Tensor([pan_gt/360, tilt_gt/360]).unsqueeze(0),
                        }
                        model.feed_data(data2feed)
                        model.nonpad_test()  # inference, get the results
                        visuals = model.get_current_visuals()  # get image results

                        res_input = tensor2img(visuals['lq'][0].unsqueeze(0))
                        # res_input = cv2.cvtColor(res_input, cv2.COLOR_RGB2BGR)

                        name_file = os.path.basename(input_name)
                        cv2.imwrite(os.path.join(output_pred, name_file + f"_{pan_gt}_{tilt_gt}.png"), res_input)


                        gt_img = dir_gt
                        gt_img = dataset.load_image(gt_img)
                        cv2.imwrite(os.path.join(output_gt, name_file + f"_{pan_gt}_{tilt_gt}.png"), 
                                    tensor2img(gt_img.unsqueeze(0)))