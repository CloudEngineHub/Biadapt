import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import PIL
import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser
import os
import multiprocessing as mp
import time
from time import strftime
import h5py

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

import torch

torch.cuda.empty_cache()
import gc

gc.collect()
import time
import random
import imageio
from subprocess import call
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import math
import datetime
import h5py
from run_featurizer import transfer_affordance, transfer_affordance_w_mask
from tqdm import tqdm

import sapien as sapien
from sapien.core import Pose
from env import Env
from camera import Camera
from robots.panda_robot import Robot

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--exp_suffix', type=str, help='exp suffix')
parser.add_argument('--category_types',
                    type=str,
                    help='list all categories [Default: None, meaning all 10 categories]',
                    default=None)
parser.add_argument('--category', type=str, default='Pliers')

parser.add_argument('--cat2freq', type=str, default=None)
parser.add_argument('--primact_type', type=str)

parser.add_argument('--out_dir', type=str, default='../exp/explore')

parser.add_argument('--src_txt_path', type=str, default='../../stats/src_USB_uncapping_list.txt',
                    help='data directory')  # lyw
parser.add_argument('--src_data_dir', type=str, default='../../exp/USB_1112_TRAIN', help='data directory')  # lyw

parser.add_argument('--trg_data_dir', type=str, help='trg_data_dir')
parser.add_argument('--trg_img_dir', type=str, default='../../exp/USB_1112_TRAIN/succ_img', help='trg_txt_path')  # lyw

parser.add_argument('--img_size', type=int, default=224)

parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='closed')
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--static_friction', type=float, default=4.0)
parser.add_argument('--dynamic_friction', type=float, default=4.0)

parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

parser.add_argument('--model_id',
                    default='stabilityai/stable-diffusion-2-1',
                    type=str,
                    help='model_id of the diffusion model in huggingface')
parser.add_argument('--t', default=261, type=int, help='time step for diffusion, choose from range [0, 1000]')
parser.add_argument('--up_ft_index',
                    default=1,
                    type=int,
                    choices=[0, 1, 2, 3],
                    help='which upsampling block of U-Net to extract the feature map')
parser.add_argument('--prompt', default='', type=str, help='prompt used in the stable diffusion')
parser.add_argument('--ensemble_size',
                    default=8,
                    type=int,
                    help='number of repeated images in each batch used to get features')

args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")
device = args.device


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    out_dir = args.out_dir
    print('out_dir: ', out_dir)

    if os.path.exists(out_dir):
        response = input('Out directory "%s" already exists, continue? (y/n) ' % out_dir)
        if response != 'y' and response != 'Y':
            sys.exit()

    if not os.path.exists(out_dir):
        os.makedirs(os.path.join(out_dir))

    flog = open(os.path.join(out_dir, 'test_log.txt'), 'w')
    args.flog = flog

    src_data_dir = args.src_data_dir
    src_succ_dir = os.path.join(src_data_dir, 'succ_files')
    src_img_dir = os.path.join(src_data_dir, 'succ_img')
    if not os.path.exists(src_succ_dir):
        src_succ_dir = os.path.join(src_data_dir, 'tmp_succ_files')
        src_img_dir = os.path.join(src_data_dir, 'tmp_succ_img')

    src_txt_path = args.src_txt_path
    src_trial_list = []
    src_shape_list = []

    with open(src_txt_path, 'r') as file:

        for line in file.readlines():
            src_trial_id, src_shape_id, category = line.strip().split()

            src_trial_list.append(src_trial_id)
            src_shape_list.append(src_shape_id)
    num_src = len(src_shape_list)
    utils.printout(flog, f'num src: {num_src}')
    trg_shape_list = []

    trg_img_dir = args.trg_img_dir
    trg_img_list = os.listdir(trg_img_dir)
    num_trg = len(trg_img_list)
    utils.printout(flog, f'num trg: {num_trg}')

    ctpts_2d_dict = dict()
    # src_category = 'Laptop'
    src_category = args.category
    for j in tqdm(range(num_trg)):

        trg_img_name = trg_img_list[j]

        trg_img_path = os.path.join(trg_img_dir, trg_img_name)
        trg_img = Image.open(trg_img_path).convert('RGB')
        print(trg_img_path)
        # trg_mask_path = os.path.join(args.trg_data_dir,'interaction_mask_'+str(trg_trial_id)+'.png')

        cur_ctpts_list = []
        for idx in range(num_src):
            cur_trial_id = src_trial_list[idx]
            cur_shape_id = src_shape_list[idx]
            # if cur_trial_id not in ['100030']:
            #     continue
            cur_file_path = os.path.join(src_succ_dir, 'result_%s.json' % str(cur_trial_id))
            src_img_path = os.path.join(src_img_dir,
                                        str(cur_trial_id) + f'_{src_category}_' + str(cur_shape_id) + '_rgb.png')
            if not os.path.exists(src_img_path):
                print('No ', src_img_path)
                continue
            src_img = Image.open(src_img_path).convert('RGB')

            with open(cur_file_path, 'r') as fin:
                result_data = json.load(fin)

            src_ctpt1 = result_data['pixel_locs_1']
            src_x1, src_y1 = src_ctpt1
            src_ctpt2 = result_data['pixel_locs_2']
            src_x2, src_y2 = src_ctpt2
            out_path = os.path.join(out_dir, str(cur_trial_id) + '_' + str(j))
            # if idx <=4:
            #     src_y1 -=40
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            vis_img = utils.visualize_xy(x1=src_y1,
                                         y1=src_x1,
                                         x2=src_y2,
                                         y2=src_x2,
                                         img=src_img,
                                         rad=8,
                                         save_root=os.path.join(out_path, 'Origin_vis_xy.png'))

            try:
                cur_ctpt1 = transfer_affordance(src_img,
                                                trg_img,
                                                prompt=args.prompt,
                                                src_pos_list=[[src_y1, src_x1], [src_y2, src_x2]],
                                                save_root=out_path,
                                                id=1)
                y1, x1 = cur_ctpt1
            except Exception:
                print('DIFT ERROR')
                continue
            torch.cuda.empty_cache()

            try:
                cur_ctpt2 = transfer_affordance(src_img,
                                                trg_img,
                                                prompt=args.prompt,
                                                src_pos_list=[[src_y2, src_x2], [src_y1, src_x1]],
                                                save_root=out_path,
                                                id=2)
                y2, x2 = cur_ctpt2
            except Exception:
                print('DIFT ERROR')
                continue

            torch.cuda.empty_cache()

            print('x1, y1 : ', x1, y1)
            print('x2, y2 : ', x2, y2)

            vis_img = utils.visualize_xy(x1=y1,
                                         y1=x1,
                                         x2=y2,
                                         y2=x2,
                                         img=trg_img,
                                         rad=8,
                                         save_root=os.path.join(out_path, 'vis_xy.png'))

            cur_ctpts_list.append([x1, y1, x2, y2])
            cur_ctpts = [[x1, y1], [x2, y2]]
            with open(os.path.join(out_path, 'cur_ctpts_2d.txt'), 'w') as f:
                for point in cur_ctpts:
                    f.write(f'{point[0]} {point[1]}\n')

            utils.printout(flog, f'finished from{cur_trial_id} to  {j}')

        ctpts_2d_dict[str(j)] = cur_ctpts_list

        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, 'ctpts_2d_dict.json'), 'w') as fout:
        json.dump(ctpts_2d_dict, fout, indent=4)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
