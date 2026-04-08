import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import PIL
import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser

from pointnet2_ops.pointnet2_utils import furthest_point_sample
import math
import datetime
import h5py

from env import Env
from camera import Camera

parser = ArgumentParser()
parser.add_argument('--ctpts_2d_file_path', type=str, default='')
parser.add_argument('--ctpts_2d_file_path2', type=str, default='xxx')
parser.add_argument('--ctpts_2d_file_path3', type=str, default='xxx')

parser.add_argument('--val_data_dir', type=str, help='data directory')
parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')

parser.add_argument('--trg_txt_path', type=str, default='')
parser.add_argument('--categories', type=str, default='')
parser.add_argument('--out_dir', type=str, default='')

args = parser.parse_args()


def main():
    out_dir = args.out_dir
    print('out_dir: ', out_dir)
    if os.path.exists(out_dir):
        response = input('Out directory "%s" already exists, continue? (y/n) ' % out_dir)
        if response != 'y' and response != 'Y':
            sys.exit()

    trg_shape_list = []
    vis = dict()
    with open(args.trg_txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            trg_shape_id, category = line.strip().split()
            if category in args.categories:
                trg_shape_list.append(trg_shape_id)
                vis[str(trg_shape_id)] = False

    num_trg_shape = len(trg_shape_list)

    ctpts_2d_data = dict()
    ctpts_2d_dir_list = [args.ctpts_2d_file_path, args.ctpts_2d_file_path2, args.ctpts_2d_file_path3]

    for cur_dir in ctpts_2d_dir_list:
        if not os.path.exists(cur_dir):
            continue
        with open(cur_dir, 'r') as f:
            cur_ctpts_2d_data = json.load(f)

        for shape_id, ctpts_2d_list in cur_ctpts_2d_data.items():
            if shape_id not in ctpts_2d_data:
                ctpts_2d_data[shape_id] = []
                ctpts_2d_data[shape_id].extend(ctpts_2d_list)

    dir_list = [
        args.val_data_dir, args.val_data_dir2, args.val_data_dir3, args.val_data_dir4, args.val_data_dir5,
        args.val_data_dir6
    ]
    dir_name = 'succ_files'
    out_data = dict()
    cnt_shape = 0
    # trg_shape_list = ['100191']  #TODO
    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            continue
        for file in sorted(os.listdir(os.path.join(cur_dir, dir_name))):
            if file[-4:] != 'json':
                continue
            file_id = int(file.split('.')[0].split('_')[1])
            with open(os.path.join(cur_dir, dir_name, file), 'r') as fin:
                result_data = json.load(fin)

            cur_cat, cur_shape_id = result_data['category'], result_data['shape_id']
            if cur_shape_id in trg_shape_list:
                if vis[str(cur_shape_id)]:
                    continue
            else:
                continue

            vis[str(cur_shape_id)] = True
            ctpts_2d_pairs = ctpts_2d_data[str(cur_shape_id)]
            num_pairs = len(ctpts_2d_pairs)
            file_dir = os.path.join(cur_dir, dir_name)

            with h5py.File(os.path.join(file_dir, 'cam_XYZA_%d.h5' % file_id), 'r') as fin:
                cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                cam_XYZA_pts = fin['pc'][:].astype(np.float32)

            camera_metadata = result_data['camera_metadata']
            mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
            cam_XYZA = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)

            ctpts_world_list = []
            ctpts_cam_list = []
            for idxx in range(num_pairs):
                ctpts_2d = ctpts_2d_pairs[idxx]
                print(cur_shape_id,num_pairs,ctpts_2d)
                x1, y1, x2, y2 = ctpts_2d
                position_world1, position_cam1, _ = utils.get_ctpt_from_cam(mat44, cam_XYZA, x1, y1)
                position_world2, position_cam2, _ = utils.get_ctpt_from_cam(mat44, cam_XYZA, x2, y2)

                ctpts_world_list.append([position_world1.tolist(), position_world2.tolist()])
                ctpts_cam_list.append([position_cam1.tolist(), position_cam2.tolist()])
                # ctpts_world_list.append([position_world1, position_world2])
                # ctpts_cam_list.append([position_cam1, position_cam2])

            out_data[str(cur_shape_id)] = dict()
            out_data[str(cur_shape_id)]['ctpts_world'] = ctpts_world_list
            out_data[str(cur_shape_id)]['ctpts_cam'] = ctpts_cam_list
            # out_data[str(cur_shape_id)]['ctpts_2d'] = ctpts_2d
            cnt_shape += 1
            if cnt_shape >= num_trg_shape:
                break
    with open(out_dir, 'w') as f:
        json.dump(out_data, f, indent=4)

    print(f'saving to{out_dir}')


if __name__ == '__main__':
    sys.exit(main())
