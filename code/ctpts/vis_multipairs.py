import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
import numpy as np
# import sapien.core as sapien
import shutil
import sapien as sapien
from sapien.core import Pose
from PIL import Image
from env import Env, ContactError, SVDError
from camera import Camera
from robots.panda_robot import Robot
import random
import imageio
import utils as utils
import json
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import cv2

from argparse import ArgumentParser


def check_path(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def visualize_xy_pairs(ctpt1_list,
                       ctpt2_list,
                       img,
                       save_root=None,
                       rad=10,
                       base_c1=(0, 0, 255),   # 红 (BGR!)
                       base_c2=(255, 0, 0)):  # 蓝 (BGR!)
    img = np.array(img)

    # 确保是 BGR
    if img.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img.copy()

    num_pairs = len(ctpt1_list)
    overlay = img_bgr.copy()
    for i in range(num_pairs):
        if i > 4:
            break
        ctpt1 = tuple(map(int, ctpt1_list[i]))
        ctpt2 = tuple(map(int, ctpt2_list[i]))
        # # alpha = 1 - 0.35 * i / num_pairs  # 越靠后的点越浅
        # alpha = 1 - (i / num_pairs)**2 
        # # 生成红蓝不同深浅
        # color1 = tuple(int(c * alpha ) for c in base_c1)
        # color2 = tuple(int(c * alpha) for c in base_c2)
        # 线性插值到白色，越靠后的点越浅
        alpha = 0.6 + 0.4*(1 - i/(num_pairs-1))  # 0.3~1
        color1 = tuple(int(c*alpha + 255*(1-alpha)) for c in base_c1)
        color2 = tuple(int(c*alpha + 255*(1-alpha)) for c in base_c2)

        cv2.circle(overlay, ctpt1, rad, color1, -1)
        cv2.circle(overlay, ctpt2, rad, color2, -1)

    # 直接返回 overlay（点叠在原图上）
    colored_img = overlay

    # 转回 RGB 输出
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)

    if save_root is not None:
        cv2.imwrite(save_root, cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR))
        print(f'Vis Image saved to {save_root}')

    return colored_img


parser = ArgumentParser()
parser.add_argument('--ctpts_2d_file_path', type=str, default='')
parser.add_argument('--ctpts_3d_file_path', type=str, default='../stats/ctpts_3d_Box.json')

parser.add_argument('--shape_setting_path', type=str, default='../stats/shape_setting.json')

# parser.add_argument('--resuld_dir', type=str, help='data directory')
parser.add_argument('--shape_id', type=str, help='shape_id', default=102379)
# parser.add_argument('--trial_id', type=str, help='trial_id')

parser.add_argument('--category', type=str, default='Box')

parser.add_argument('--primact_type', type=str, default='capping')
parser.add_argument('--out_dir', type=str, default='../expdata/Vis_Pairs')
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

args = parser.parse_args()

# result_dir = args.result_dir
# trial_id = args.trial_id
shape_id = args.shape_id
category = args.category
out_dir = args.out_dir
check_path(out_dir)
out_logs = os.path.join(out_dir, 'logs')
out_path = os.path.join(out_logs, category + '_' + str(shape_id))

with open(args.shape_setting_path, 'r') as file:
    shape_setting_data = json.load(file)
if not os.path.exists(out_logs):
    os.mkdir(out_logs)
if os.path.exists(out_path):
    if os.path.exists(os.path.join(out_path, 'result.json')):
        print('well')

    shutil.rmtree(out_path)
os.mkdir(out_path)
flog = open(os.path.join(out_path, 'log.txt'), 'w')
out_info = dict()

density = shape_setting_data[str(shape_id)]["density"]
mu = shape_setting_data[str(shape_id)]["mu"]
friction = shape_setting_data[str(shape_id)]["friction"]
damping = shape_setting_data[str(shape_id)]["damping"]
object_scale = shape_setting_data[str(shape_id)]["obj_scale"]
robot_scale = shape_setting_data[str(shape_id)]["robot_scale"]
object_state = shape_setting_data[str(shape_id)]["state"]
urdf_type = shape_setting_data[str(shape_id)]["urdf_type"]
euler = shape_setting_data[str(shape_id)]['euler']

with open(args.ctpts_3d_file_path, 'r') as f:
    ctpts_3d_data = json.load(f)
# setup env

ctpts_3d_pairs = ctpts_3d_data[str(shape_id)]
num_pairs = len(ctpts_3d_pairs['ctpts_world'])
ctpt1_list = []
ctpt2_list = []
for idxx in range(num_pairs):
    ctpts_3d_world = ctpts_3d_pairs['ctpts_world'][idxx]
    ctpts_3d_cam = ctpts_3d_pairs['ctpts_cam'][idxx]
    cur_ctpt1, cur_ctpt2 = ctpts_3d_world
    ctpt1_list.append(np.array(cur_ctpt1))
    ctpt2_list.append(np.array(cur_ctpt2))

print("creating env")

env = Env(flog=flog, show_gui=(not args.no_gui), set_ground=True)
# setup camera
cam = Camera(env,
             near=0.1,
             far=100,
             dist=3,
             image_size1=1024,
             image_size2=1024,
             fixed_position=True,
             object_centered=True,
             restrict_dir=True)  # [0.5π, 1.5π]
print("camera created")
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
if urdf_type in ["processed"]:
    object_urdf_fn = '../data/dataset/%s/mobility_vhacd.urdf' % str(shape_id)
else:
    object_urdf_fn = '../data/dataset/%s/mobility.urdf' % str(shape_id)

quaternion = utils.euler2quaternion(roll=euler[0], pitch=euler[1], yaw=euler[2])
given_pose = Pose([0, 0, 0], quaternion)
object_material = [mu, mu, 0.01]
if args.primact_type in ['capping','closing']:
    object_state='open'
joint_angles = env.load_object(object_urdf_fn,
                               object_material,
                               state=object_state,
                               target_part_id=-1,
                               scale=object_scale,
                               density=density,
                               damping=damping,
                               given_pose=given_pose)

# wait for the object's still
still_timesteps = utils.wait_for_object_still(env)

### use the GT vision
rgb, depth = cam.get_observation()

rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))

rgb_img.save(os.path.join(out_path, 'rgb.png'))
img_path = os.path.join(out_path, 'rgb.png')
print(f'rgb_img save to {img_path}')
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]

object_movable_link_ids = env.movable_link_ids
object_all_link_ids = env.all_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_movable_link_ids)  # (448, 448), 0(unmovable) - id(movable)
gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)

mat44 = cam.get_metadata()['mat44']

ctpt1_2d_list = []
ctpt2_2d_list = []

for i in range(num_pairs):
    position_world1 = ctpt1_list[i]
    position_world2 = ctpt2_list[i]
    ctpt1_2d = utils.get_ctpt_from_world(mat44, cam_XYZA, position_world1)
    ctpt2_2d = utils.get_ctpt_from_world(mat44, cam_XYZA, position_world2)
    x1, y1 = ctpt1_2d
    x2, y2 = ctpt2_2d
    ctpt1_2d = [y1, x1]
    ctpt2_2d = [y2, x2]
    ctpt1_2d_list.append(ctpt1_2d)
    ctpt2_2d_list.append(ctpt2_2d)

visualize_xy_pairs(ctpt1_2d_list, ctpt2_2d_list, rgb_img, rad=15, save_root=os.path.join(out_path, 'Vis_Pairs.png'))

env.scene.remove_articulation(env.object)
env.close()
