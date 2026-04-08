import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser
from sapien.core import Pose
from env import Env, ContactError, SVDError
from camera import Camera
from robots.panda_robot import Robot
import random
import imageio
import math

parser = ArgumentParser()
parser.add_argument('--file_dir', type=str)
parser.add_argument('--file_id', type=str)
parser.add_argument('--repeat_id', type=int)
parser.add_argument('--shape_setting_path', type=str, default='../stats/shape_setting.json')

parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--stiffness', type=int, default=10)
parser.add_argument('--damping', type=int, default=100)
parser.add_argument('--target_part_state', type=str, default='closed')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--static_friction', type=float, default=4.0)
parser.add_argument('--dynamic_friction', type=float, default=4.0)
parser.add_argument('--scene_static_friction', type=float, default=0.30)
parser.add_argument('--scene_dynamic_friction', type=float, default=0.30)
parser.add_argument('--gripper_scale', type=float, default=2.0)

parser.add_argument('--position_world1', type=str)
parser.add_argument('--up_world1', type=str)
parser.add_argument('--forward_world1', type=str)
parser.add_argument('--position_world2', type=str)
parser.add_argument('--up_world2', type=str)
parser.add_argument('--forward_world2', type=str)

parser.add_argument('--task_threshold', type=int, default=15)
parser.add_argument('--euler_threshold', type=int, default=5)
parser.add_argument('--check_contactError', action='store_true', default=False)
parser.add_argument('--out_dir', type=str, default='./out_dir')
parser.add_argument('--save_results', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

parser.add_argument('--random_seed', type=int, default=None)
args = parser.parse_args()

file_id = args.file_id
repeat_id = args.repeat_id
out_dir = os.path.join(args.out_dir, args.file_dir.split('/')[-2])
out_path = os.path.join(out_dir, 'logs')
with open(args.shape_setting_path, 'r') as file:
    shape_setting_data = json.load(file)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_path):
    os.makedirs(out_path)
if args.save_results and (not os.path.exists(os.path.join(out_dir, 'succ_files'))):
    os.makedirs(os.path.join(out_dir, 'succ_files')), os.makedirs(os.path.join(out_dir, 'succ_gif'))
    os.makedirs(os.path.join(out_dir, 'fail_files')), os.makedirs(os.path.join(out_dir, 'fail_gif'))
    os.makedirs(os.path.join(out_dir, 'invalid_files')), os.makedirs(os.path.join(out_dir, 'invalid_gif'))

with open(os.path.join(args.file_dir, 'result_%s.json' % args.file_id), 'r') as fin:
    result_data = json.load(fin)

shape_id = result_data['shape_id']
category = result_data['category']
primact_type = result_data['primact_type']
if 'random_seed' in result_data:
    random_seed = result_data['random_seed']
    np.random.seed(random_seed)
elif args.random_seed is not None:
    random_seed = np.random.seed(args.random_seed)
    np.random.seed(random_seed)

object_scale = shape_setting_data[str(shape_id)]["obj_scale"]
robot_scale = shape_setting_data[str(shape_id)]["robot_scale"]
urdf_type = shape_setting_data[str(shape_id)]["urdf_type"]

start_dist1 = shape_setting_data[str(shape_id)]["start_dist1"]
displacement1 = shape_setting_data[str(shape_id)]["displacement1"]
maneuver_dist1 = shape_setting_data[str(shape_id)]["maneuver_dist1"]
start_dist2 = shape_setting_data[str(shape_id)]["start_dist2"]
displacement2 = shape_setting_data[str(shape_id)]["displacement2"]
maneuver_dist2 = shape_setting_data[str(shape_id)]["maneuver_dist2"]

density = shape_setting_data[str(shape_id)]["density"]
mu = shape_setting_data[str(shape_id)]["mu"]
friction = shape_setting_data[str(shape_id)]["friction"]
damping = shape_setting_data[str(shape_id)]["damping"]
target_part_state = shape_setting_data[str(shape_id)]["state"]
if primact_type in ['capping', 'closing']:
    target_part_state = args.target_part_state
euler = shape_setting_data[str(shape_id)]['euler']


def swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b


def save_succ_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs):
    utils.save_data(os.path.join(out_dir, 'succ_files'),
                    file_id,
                    out_info,
                    cam_XYZA_list,
                    gt_target_link_mask,
                    repeat_id=repeat_id)
    imageio.mimsave(os.path.join(out_dir, 'succ_gif', '%s_%d_%s_%s.gif' % (file_id, repeat_id, category, shape_id)),
                    gif_imgs)


def save_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs):
    utils.save_data(os.path.join(out_dir, 'fail_files'),
                    file_id,
                    out_info,
                    cam_XYZA_list,
                    gt_target_link_mask,
                    repeat_id=repeat_id)
    imageio.mimsave(os.path.join(out_dir, 'fail_gif', '%s_%d_%s_%s.gif' % (file_id, repeat_id, category, shape_id)),
                    gif_imgs)


def save_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg):
    utils.save_data(os.path.join(out_dir, 'invalid_files'),
                    file_id,
                    out_info,
                    cam_XYZA_list,
                    gt_target_link_mask,
                    repeat_id=repeat_id)
    # fimg.save(os.path.join(out_dir, 'invalid_gif', '%s_%d_%s_%s.png' % (file_id, repeat_id, category, shape_id)))

    # 假设 fimg 是一个 numpy 数组，形状为 (H, W, 3) 或 (H, W)
    fimg_pil = Image.fromarray(fimg)
    fimg_pil.save(os.path.join(out_dir, 'invalid_gif', '%s_%d_%s_%s.png' % (file_id, repeat_id, category, shape_id)))


out_info = dict()
gif_imgs = []
fimg = None
success = False

# setup env
print("creating env")
env = Env(show_gui=(not args.no_gui),
          set_ground=True,
          static_friction=args.scene_static_friction,
          dynamic_friction=args.scene_dynamic_friction)

# setup camera
cam_theta, cam_phi = result_data['camera_metadata']['theta'], result_data['camera_metadata']['phi']
cam = Camera(env, phi=cam_phi, theta=cam_theta, random_position=False, restrict_dir=True)  # [0.5π, 1.5π]
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
print("camera created")

if urdf_type in ["processed"]:
    object_urdf_fn = '../data/dataset/%s/mobility_vhacd.urdf' % str(shape_id)
else:
    object_urdf_fn = '../data/dataset/%s/mobility.urdf' % str(shape_id)

quaternion = utils.euler2quaternion(roll=euler[0], pitch=euler[1], yaw=euler[2])
given_pose = Pose([0, 0, 0], quaternion)
object_material = [mu, mu, 0.01]
joint_angles = result_data['joint_angles']

joint_angles = env.load_object(object_urdf_fn,
                               object_material,
                               state=target_part_state,
                               target_part_id=-1,
                               scale=object_scale,
                               density=density,
                               damping=damping,
                               given_pose=given_pose,
                               given_joint_angles=joint_angles)

# wait for the object's still
still_timesteps = utils.wait_for_object_still(env)

### use the GT vision
rgb, depth = cam.get_observation()

rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]

object_movable_link_ids = env.movable_link_ids
object_all_link_ids = env.all_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_movable_link_ids)  # (448, 448), 0(unmovable) - id(movable)
gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)

position_world1 = np.array([float(i) for i in args.position_world1[3:].split(',')], dtype=np.float32)
up_world1 = np.array([float(i) for i in args.up_world1[3:].split(',')], dtype=np.float32)
forward_world1 = np.array([float(i) for i in args.forward_world1[3:].split(',')], dtype=np.float32)
position_world2 = np.array([float(i) for i in args.position_world2[3:].split(',')], dtype=np.float32)
up_world2 = np.array([float(i) for i in args.up_world2[3:].split(',')], dtype=np.float32)
forward_world2 = np.array([float(i) for i in args.forward_world2[3:].split(',')], dtype=np.float32)

action_dir1, action_dir2 = utils.get_action_dir(primact_type, up_world1, forward_world1, up_world2, forward_world2)
if_inverse = True if primact_type in ['folding', 'capping', 'closing'] else False
# sample a pixel on target part
mat44 = cam.get_metadata()['mat44']
src_ctpt1_2d = result_data['pixel_locs_1']
src_ctpt2_2d = result_data['pixel_locs_2']

ctpt1_2d = utils.get_ctpt_from_world(mat44, cam_XYZA, position_world1)
ctpt2_2d = utils.get_ctpt_from_world(mat44, cam_XYZA, position_world2)

x1, y1 = ctpt1_2d
x2, y2 = ctpt2_2d
print('ctpt1_2d:', ctpt1_2d)
print('ctpt2_2d:', ctpt2_2d)
print('src_ctpt1_2d:', src_ctpt1_2d)
print('src_ctpt2_2d:', src_ctpt2_2d)

xs, ys = np.where(gt_all_link_mask > 0)
target_part1_id = object_all_link_ids[gt_all_link_mask[x1, y1] - 1]
target_part2_id = object_all_link_ids[gt_all_link_mask[x2, y2] - 1]

if target_part1_id > target_part2_id and primact_type in ['uncapping']:
    x1, y1 = ctpt2_2d
    x2, y2 = ctpt1_2d
    position_world1, position_world2 = swap(position_world1, position_world2)
    target_part1_id, target_part2_id = swap(target_part1_id, target_part2_id)
    up_world1, up_world2 = swap(up_world1, up_world2)
    forward_world1, forward_world2 = swap(forward_world1, forward_world2)

vis_xy_img = utils.visualize_xy(x1=y1, y1=x1, x2=y2, y2=x2, img=rgb_img, save_root=os.path.join(out_path, 'Vis_XY.png'))

out_info['pixel_locs_1'] = [int(x1), int(y1)]
out_info['pixel_locs_2'] = [int(x2), int(y2)]

out_info['camera_metadata'] = cam.get_metadata_json()
out_info['object_state'] = args.target_part_state
out_info['joint_angles'] = joint_angles
out_info['shape_id'] = shape_id
out_info['category'] = category
out_info['primact_type'] = primact_type
out_info['success'] = 'False'
out_info['result'] = 'INVALID'
print('target_parts_id:', [target_part1_id, target_part2_id])
if target_part1_id == target_part2_id:
    env.scene.remove_articulation(env.object)
    save_invalid_data(out_info, cam_XYZA_list, gt_all_link_mask, vis_xy_img)
    env.close()
    exit(2)
#TODO
target_part1_id, target_part2_id = 8, 9
env.set_target_object_part_actor_ids(target_part1_id, target_part2_id)
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
out_info['target_object_part_actor_id_1'] = env.target_object_part_actor_id_1
out_info['target_object_part_actor_id_2'] = env.target_object_part_actor_id_2  #set joint friction
out_info['result'] = 'VALID'
# for joint in env.object.get_joints():
#     print(joint.get_friction())
env.render()
if not args.no_gui:
    env.wait_to_start()

pre_pose1, pre_rotmat1, start_pose1, start_rotmat1, final_pose1, final_rotmat1 = utils.get_rotmat_pre(
    cam,
    position_world1,
    up_world1,
    forward_world1,
    number='1',
    out_info=out_info,
    start_dist=start_dist1,
    displacement=displacement1,
    maneuver_dist=maneuver_dist1,
    is_given_action=True if action_dir1 is not None else False,
    action_direction_world=action_dir1,
    if_inverse=if_inverse)
pre_pose2, pre_rotmat2, start_pose2, start_rotmat2, final_pose2, final_rotmat2 = utils.get_rotmat_pre(
    cam,
    position_world2,
    up_world2,
    forward_world2,
    number='2',
    out_info=out_info,
    start_dist=start_dist2,
    displacement=displacement2,
    maneuver_dist=maneuver_dist2,
    is_given_action=True if action_dir2 is not None else False,
    action_direction_world=action_dir2,
    if_inverse=if_inverse)

# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = [args.static_friction, args.dynamic_friction, 0.01]

open_gripper = True if primact_type in ['opening', 'closing', 'uncapping', 'capping'] else False
robot1 = Robot(env, robot_urdf_fn, robot_material, open_gripper=open_gripper, scale=robot_scale)
robot2 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)

robot1.robot.set_root_pose(final_pose1)
robot2.robot.set_root_pose(final_pose2)
env.render()
rgb_final_pose, _ = cam.get_observation()
final_pose_img = Image.fromarray((rgb_final_pose * 255).astype(np.uint8))
final_pose_img.save(os.path.join(out_path, 'viz_target_pose.png'))

robot1.robot.set_root_pose(pre_pose1)
robot2.robot.set_root_pose(pre_pose2)
env.render()
rgb_pre_pose, _ = cam.get_observation()
pre_pose_img = Image.fromarray((rgb_pre_pose * 255).astype(np.uint8))
pre_pose_img.save(os.path.join(out_path, 'viz_pre_pose.png'))

# move back
robot1.robot.set_root_pose(start_pose1)
robot2.robot.set_root_pose(start_pose2)
env.render()
rgb_start_pose, _ = cam.get_observation()
start_pose_img = Image.fromarray((rgb_start_pose * 255).astype(np.uint8))
start_pose_img.save(os.path.join(out_path, 'viz_start_pose.png'))

if not args.no_gui:
    env.wait_to_start()

# calculate position    world = trans @ local
obj_part1_pose = env.get_target_part1_pose()
obj_part2_pose = env.get_target_part2_pose()
start_joint_angle = env.get_target_part_qpos() / np.pi * 180.0

target_link1_mat44 = obj_part1_pose.to_transformation_matrix()  # local2world
target_link2_mat44 = obj_part2_pose.to_transformation_matrix()  # local2world
out_info['start_target_link1_mat44'] = target_link1_mat44.tolist()
out_info['start_target_link2_mat44'] = target_link2_mat44.tolist()

prev_link1_world_xyz1 = target_link1_mat44 @ np.array([0, 0, 0, 1])
prev_link1_world_xyz = prev_link1_world_xyz1[:3]

prev_link2_world_xyz1 = target_link2_mat44 @ np.array([0, 0, 0, 1])
prev_link2_world_xyz = prev_link2_world_xyz1[:3]

out_info['success'] = 'False'
out_info['result'] = 'VALID'

grasp_fail = True
success = True
gif_imgs = []
print('here')
try:

    try:
        # activate contact checking
        env.dual_start_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, robot2.hand_actor_id,
                                        robot2.gripper_actor_ids, True)

        robot1.robot.set_root_pose(pre_pose1)
        robot2.robot.set_root_pose(pre_pose2)

        # save img
        env.render()
        rgb_pose, _ = cam.get_observation()
        fimg = (rgb_pose * 255).astype(np.uint8)
        fimg = Image.fromarray(fimg)

        env.step()

    except Exception as e:
        print('contact error')
        print(e)
        out_info['result'] = 'INVALID'
        save_invalid_data(out_info, cam_XYZA_list, gt_all_link_mask, fimg)
        env.scene.remove_articulation(env.object)
        env.scene.remove_articulation(robot1.robot)
        env.scene.remove_articulation(robot2.robot)
        env.close()
        exit(2)
    env.dual_end_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, robot2.hand_actor_id,
                                  robot2.gripper_actor_ids, False)
    if open_gripper:
        robot1.close_gripper()

    robot2.close_gripper()

    env.step()
    env.render()

    imgs = utils.dual_gripper_move_to_target_pose(robot1,
                                                  robot2,
                                                  final_rotmat1,
                                                  final_rotmat2,
                                                  num_steps=args.move_steps,
                                                  cam=cam,
                                                  vis_gif=True)
    gif_imgs.extend(imgs)

    imgs = utils.dual_gripper_wait_n_steps(robot1, robot2, n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    imageio.mimsave(os.path.join(out_path, 'dual_move.gif'), gif_imgs)

except Exception:
    success = False
    out_info['result'] = 'INVALID'
    print('Invalid')
    env.scene.remove_articulation(env.object)
    env.scene.remove_articulation(robot1.robot)
    env.scene.remove_articulation(robot2.robot)
    env.close()
    exit(2)

next_link1_pose = env.get_target_part1_pose()
next_link2_pose = env.get_target_part2_pose()

target_part1_trans = next_link1_pose.to_transformation_matrix(
)  # world coordinate -> target part transformation matrix 4*4 SE3
target_part2_trans = next_link2_pose.to_transformation_matrix()

transition1 = np.linalg.inv(target_part1_trans) @ target_link1_mat44
transition2 = np.linalg.inv(target_part2_trans) @ target_link2_mat44

alpha, beta, gamma = utils.rotationMatrixToEulerAngles(
    transition1)  # eulerAngles(trans) = eulerAngles(prev_mat44) - eulerAngle(then_mat44)
out_info['target_part_trans'] = target_part1_trans.tolist()
out_info['transition'] = transition1.tolist()
out_info['alpha'] = alpha.tolist()
out_info['beta'] = beta.tolist()
out_info['gamma'] = gamma.tolist()

# calculate displacement
next_link1_world_xyz1 = target_part1_trans @ np.array([0, 0, 0, 1])
next_link1_world_xyz = next_link1_world_xyz1[:3]
trajectory1 = next_link1_world_xyz - prev_link1_world_xyz

next_link2_world_xyz1 = target_part2_trans @ np.array([0, 0, 0, 1])
next_link2_world_xyz = next_link2_world_xyz1[:3]
trajectory2 = next_link2_world_xyz - prev_link2_world_xyz

out_info['start_pos1'] = prev_link1_world_xyz.tolist()
out_info['final_pos1'] = next_link1_world_xyz.tolist()
out_info['start_pos2'] = prev_link2_world_xyz.tolist()
out_info['final_pos2'] = next_link2_world_xyz.tolist()

out_info['trajectory1'] = trajectory1.tolist()
out_info['trajectory2'] = trajectory2.tolist()

final_joint_angle = env.object.get_qpos()[env.target_object_part_joint_id] / np.pi * 180.0
out_info['start_joint_angle'] = float(start_joint_angle)
out_info['final_joint_angle'] = float(final_joint_angle)

joint_angle_lower = env.joint_angles_lower[env.target_object_part_joint_id] / np.pi * 180.0
joint_angle_upper = env.joint_angles_upper[env.target_object_part_joint_id] / np.pi * 180.0
out_info['target_joint_angle_lower'] = float(joint_angle_lower)
out_info['target_joint_angle_upper'] = float(joint_angle_upper)

success, is_open, no_move = utils.check_success(start_joint_angle=start_joint_angle,
                                                final_joint_angle=final_joint_angle,
                                                limit_l=joint_angle_lower,
                                                limit_r=joint_angle_upper,
                                                trajectoty=trajectory1,
                                                state=target_part_state,
                                                primact_type=primact_type)

env.scene.remove_articulation(robot1.robot)
env.scene.remove_articulation(robot2.robot)
env.scene.remove_articulation(env.object)
env.close()

if success:
    if args.save_results:
        save_succ_data(out_info, cam_XYZA_list, gt_all_link_mask, gif_imgs)
        exit(0)
else:
    if args.save_results:
        save_fail_data(out_info, cam_XYZA_list, gt_all_link_mask, gif_imgs)
        exit(1)
