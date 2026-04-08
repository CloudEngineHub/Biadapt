from argparse import ArgumentParser
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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


def check_path(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def save_succ_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs, rgb_img=None):
    utils.save_data(os.path.join(out_dir, 'tmp_succ_files'), trial_id, out_info, cam_XYZA_list, gt_target_link_mask)
    check_path(os.path.join(out_dir, 'tmp_succ_gif'))
    imageio.mimsave(os.path.join(out_dir, 'tmp_succ_gif', '%d_%s_%s.gif' % (trial_id, category, shape_id)), gif_imgs)
    if rgb_img is not None:
        check_path(os.path.join(out_dir, 'tmp_succ_img'))
        rgb_img.save(os.path.join(out_dir, 'tmp_succ_img', '%d_%s_%s_rgb.png' % (trial_id, category, shape_id)))


def save_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs, rgb_img=None):
    if args.save_data:
        utils.save_data(os.path.join(out_dir, 'fail_files'), trial_id, out_info, cam_XYZA_list, gt_target_link_mask)
        check_path(os.path.join(out_dir, 'fail_gif'))
        imageio.mimsave(os.path.join(out_dir, 'fail_gif', '%d_%s_%s.gif' % (trial_id, category, shape_id)), gif_imgs)

        if rgb_img is not None:
            check_path(os.path.join(out_dir, 'fail_img'))
            rgb_img.save(os.path.join(out_dir, 'fail_img', '%d_%s_%s_rgb.png' % (trial_id, category, shape_id)))


def save_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg, rgb_img):
    if args.save_data:
        utils.save_data(os.path.join(out_dir, 'invalid_files'), trial_id, out_info, cam_XYZA_list, gt_target_link_mask)
        # check_path(os.path.join(out_dir, 'invalid_gif'))
        fimg.save(os.path.join(out_dir, 'invalid_gif', '%d_%s_%s.png' % (trial_id, category, shape_id)))
        if rgb_img is not None:
            check_path(os.path.join(out_dir, 'invalid_img'))
            rgb_img.save(os.path.join(out_dir, 'invalid_img', '%d_%s_%s_rgb.png' % (trial_id, category, shape_id)))


parser = ArgumentParser()
parser.add_argument('--trial_id', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--shape_id', type=str, default='10537')
parser.add_argument('--category', type=str, default='Scissors')

parser.add_argument('--primact_type', type=str, default='opening')
parser.add_argument('--out_dir', type=str, default='../expdata/Scissors')

parser.add_argument('--object_scale', type=float, default=0.8)
parser.add_argument('--robot_scale', type=float, default=1.8)
parser.add_argument('--density', type=float, default=0.5)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='closed')
parser.add_argument('--start_dist', type=float, default=0.15)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--threshold', type=int, default=3)

parser.add_argument('--add_table', action='store_true', default=False)

parser.add_argument('--save_data', action='store_true', default=True)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

args = parser.parse_args()

shape_id = args.shape_id
category = args.category
primact_type = args.primact_type
trial_id = args.trial_id
out_dir = args.out_dir
check_path(out_dir)
out_logs = os.path.join(out_dir, 'logs')
out_path = os.path.join(out_logs, category + '_' + shape_id + '_' + str(trial_id))

if not os.path.exists(out_logs):
    os.mkdir(out_logs)

if os.path.exists(out_path):
    if os.path.exists(os.path.join(out_path, 'result.json')):
        print('well')

    shutil.rmtree(out_path)
os.mkdir(out_path)

flog = open(os.path.join(out_path, 'log.txt'), 'w')
out_info = dict()
gif_imgs = []
fimg = None
success = False

out_info['shape_id'] = shape_id
out_info['category'] = category
out_info['primact_type'] = primact_type
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

# setup env
print("creating env")

env = Env(flog=flog, show_gui=(not args.no_gui), set_ground=True)

cam = Camera(env, fixed_position=True, restrict_dir=True)  # [0.5π, 1.5π] TODO
# cam = Camera(env,
#              near=0.1,
#              far=100,
#              dist=2,
#              image_size1=1024,
#              image_size2=1024,
#              fixed_position=True,
#              object_centered=True,
#              restrict_dir=True)  # [0.5π, 1.5π]
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

out_info['camera_metadata'] = cam.get_metadata_json()
object_urdf_fn = '../data/dataset/%s/mobility.urdf' % str(shape_id)
# object_urdf_fn = '../data/dataset/%s/mobility_origin.urdf' % str(shape_id)
###PS：scissors :pi,Pilers:-2/3 pi
quaternion = utils.euler2quaternion(roll=0, pitch=0, yaw=5 * np.pi / 6)
x_offset = 0
y_offset = 0
z_offset = 0
print('offset position:', x_offset, y_offset, z_offset)
given_pose = Pose([0, 0, 0], quaternion)

flog.write('object_urdf_fn: %s\n' % object_urdf_fn)

if not os.path.exists(object_urdf_fn):
    print('No URDF PATH')
    flog.write('No URDF PATH')
    flog.close()
    env.close()
    exit(1)

object_material = [1.0, 1.0, 0.01]  ###
target_joint_name = 'joint_0'
try:
    joint_angles = env.load_object(object_urdf_fn,
                                   object_material,
                                   state=args.target_part_state,
                                   target_part_id=-1,
                                   scale=args.object_scale,
                                   density=args.density,
                                   damping=args.damping,
                                   given_pose=given_pose,
                                   target_joint_name=target_joint_name)

except Exception:
    print('error while load object')
    env.close()
    exit(3)

print('object loaded successfully')

out_info['object_state'] = args.target_part_state
out_info['joint_angles'] = joint_angles
still_imgs = []
still_timesteps = utils.wait_for_object_still(env)
if still_timesteps < 5000:
    flog.write('Object Not Still!\n')
    flog.close()
    env.close()
    exit(1)

### use the GT vision
env.scene.step()
env.render()
print('observe!')
rgb, depth = cam.get_observation()
rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))
rgb_img.save(os.path.join(out_path, 'rgb.png'))
img_path = os.path.join(out_path, 'rgb.png')
print(f'rgb_img save to {img_path}')
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]

object_movable_link_ids = env.movable_link_ids  #list
object_all_link_ids = env.all_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_movable_link_ids)  # (448, 448), 0(unmovable) - id(movable)
movable_mask_img = Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255)
Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(os.path.join(out_path, 'movable_mask.png'))
gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)
all_mask_img = Image.fromarray((gt_all_link_mask > 0).astype(np.uint8) * 255)

xs, ys = np.where(gt_movable_link_mask > 0)
if len(xs) == 0:
    flog.write('No Movable Pixel! Quit!\n')
    flog.close()
    env.scene.remove_articulation(env.object)
    env.close()
    exit(3)

# link_id1 = 8
# link_id2 = 9
# if shape_id in ['10889', '10960', '10973']:
#     link_id1 = 9
#     link_id2 = 8
# gt_link1_mask = cam.get_movable_link_mask([link_id1])
# gt_link2_mask = cam.get_movable_link_mask([link_id2])
# link1_mask_img = Image.fromarray((gt_link1_mask > 0).astype(np.uint8) * 255)
# link2_mask_img = Image.fromarray((gt_link2_mask > 0).astype(np.uint8) * 255)

# x1_l, x1_r = 180, 240
# y1_l, y1_r = 200, 260
# limit_mask1 = cam.get_limit_mask(x1_l, x1_r, y1_l, y1_r)
# limit_mask1_img = Image.fromarray((limit_mask1 > 0).astype(np.uint8) * 255)
# conbine_mask1 = cam.get_conbine_mask(limit_mask1, gt_link1_mask)
# conbine_mask1_img = Image.fromarray((conbine_mask1 > 0).astype(np.uint8) * 255)

# xs, ys = np.where(gt_link1_mask > 0)
# idx1 = np.random.randint(len(xs))

# x1, y1 = xs[idx1], ys[idx1]  #target1
# target_part1_id = link_id1  #movable

# x2_l, x2_r = 230, 300
# y2_l, y2_r = 220, 260
# if shape_id in ['10567']:
#     x2_l, x2_r = 170, 220
#     y2_l, y2_r = 200, 260
# limit_mask2 = cam.get_limit_mask(x2_l, x2_r, y2_l, y2_r)
# limit_mask2_img = Image.fromarray((limit_mask2 > 0).astype(np.uint8) * 255)

# conbine_mask2 = cam.get_conbine_mask(limit_mask2, gt_link2_mask)
# conbine_mask2_img = Image.fromarray((conbine_mask2 > 0).astype(np.uint8) * 255)

# xs, ys = np.where(gt_link2_mask > 0)
# if len(xs) == 0:
#     env.scene.remove_articulation(env.object)
#     env.close()
# #     exit(3)

# #if use edge
# if_edge = np.random.uniform() < 0.2
# if if_edge:
#     x2, y2, top_10 = utils.find_edge_point(xs, ys)
#     chosen_index = np.random.choice(top_10)
#     x2, y2 = xs[chosen_index], ys[chosen_index]
# else:
#     idx2 = np.random.randint(len(xs))
#     x2, y2 = xs[idx2], ys[idx2]  #target2

# target_part2_id = link_id2  # unmovable

xs, ys = np.where(gt_all_link_mask > 0)
idx1 = np.random.randint(len(xs))
idx2 = np.random.randint(len(xs))
x1, y1 = xs[idx1], ys[idx1]  #target1
x2, y2 = xs[idx2], ys[idx2]
target_part1_id = object_all_link_ids[gt_all_link_mask[x1, y1] - 1]

target_part2_id = object_all_link_ids[gt_all_link_mask[x2, y2] - 1]
tot_trial = 0
while tot_trial < 50 and (target_part1_id == target_part2_id):
    tot_trial += 1
    x2, y2 = xs[idx2], ys[idx2]
    target_part2_id = object_all_link_ids[gt_all_link_mask[x2, y2] - 1]

if target_part1_id == target_part2_id:
    print(f'same part {target_part1_id}')
    env.scene.remove_articulation(env.object)
    env.close()
    exit(3)

print('x1,y1:', x1, y1, target_part1_id)
print('x2,y2:', x2, y2, target_part2_id)

utils.visualize_xy(save_root=os.path.join(out_path, 'vis_xy.png'), x1=y1, y1=x1, x2=y2, y2=x2, img=rgb_img)
env.set_target_object_part_actor_ids(target_part1_id, target_part2_id)

out_info['pixel_locs_1'] = [int(x1), int(y1)]
out_info['pixel_locs_2'] = [int(x2), int(y2)]

out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
out_info['target_object_part_actor_id_1'] = env.target_object_part_actor_id_1
out_info['target_object_part_actor_id_2'] = env.target_object_part_actor_id_2

start_dist1 = 0.12
displacement1 = 0.15
maneuver_dist1 = 0
start_dist2 = 0.12
displacement2 = 0.15
maneuver_dist2 = 0

given_left = np.array([0, 1, 0])
# given_left = given_left / np.linalg.norm(given_left)
if_given_left = True
if_constraint_up = if_given_left
# print(if_given_left, given_left)
if_given_up = False
given_up = None

given_forward1 = utils.get_near_vector(np.array([-1, 0, 0]), 20)
given_forward2 = utils.get_near_vector(np.array([1, 0, 0]), 20)
given_up1 = utils.get_near_vector(np.array([0, 0, -1]), 15)
given_up2 = utils.get_near_vector(np.array([0, 0, -1]), 15)
# given_forward = np.array([0, 1, 0])
if_inverse = False
if primact_type in ['closing', 'folding', 'capping']:
    if_inverse = True

pre_pose1, pre_rotmat1, start_pose1, start_rotmat1, final_pose1, final_rotmat1, up1, forward1 = utils.cal_pre_final_pose(
    cam,
    cam_XYZA,
    x1,
    y1,
    number='1',
    out_info=out_info,
    start_dist=start_dist1,
    maneuver_dist=maneuver_dist1,
    displacement=displacement1,
    if_given_forward=False,  # True
    given_forward=given_forward1,
    if_given_left=False,
    given_left=given_left,
    if_given_action_direction=False,  #left
    if_constraint_up=if_constraint_up,
    given_up=given_up1,
    if_given_up=True,
    if_inverse=if_inverse)  # True
pre_pose2, pre_rotmat2, start_pose2, start_rotmat2, final_pose2, final_rotmat2, up2, forward2 = utils.cal_pre_final_pose(
    cam,
    cam_XYZA,
    x2,
    y2,
    number='2',
    out_info=out_info,
    start_dist=start_dist2,
    maneuver_dist=maneuver_dist2,
    displacement=displacement2,
    if_given_forward=True,
    given_forward=given_forward2,
    if_given_left=False,
    given_left=given_left,
    if_given_action_direction=False,  #left
    action_direction_world=-given_forward2,
    if_constraint_up=if_constraint_up,
    given_up=given_up2,
    if_given_up=True,
    if_inverse=if_inverse)  # True
# utils.get_near_vector(, 20)
robot_urdf_fn = '/home/zjinxian/Biadapt/code/robots/panda_gripper.urdf'
robot_material = [3, 3, 0.01]
robot_scale = args.robot_scale

robot1 = Robot(env, robot_urdf_fn, robot_material, id=1, open_gripper=True, scale=robot_scale, name='robot1')
robot2 = Robot(env, robot_urdf_fn, robot_material, id=2, open_gripper=True, scale=robot_scale, name='robot2')

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

env.step()
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

        robot1.robot.set_root_pose(start_pose1)
        robot2.robot.set_root_pose(start_pose2)

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

except Exception as e:
    print("Contact Error!")
    print(e)
    out_info['result'] = 'INVALID'
    save_invalid_data(out_info, cam_XYZA_list, gt_all_link_mask, fimg)
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
flog.write('alpha : %s\n' % str(alpha))
flog.write('beta : %s\n' % str(beta))
flog.write('gamma : %s\n' % str(gamma))

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
flog.write('start_joint_angle:%s\n' % str(start_joint_angle))
flog.write('final_joint_angle:%s\n' % str(final_joint_angle))

joint_angle_lower = env.joint_angles_lower[env.target_object_part_joint_id] / np.pi * 180.0
joint_angle_upper = env.joint_angles_upper[env.target_object_part_joint_id] / np.pi * 180.0
out_info['target_joint_angle_lower'] = float(joint_angle_lower)
out_info['target_joint_angle_upper'] = float(joint_angle_upper)
flog.write('target_joint_angle_lower:%s\n' % str(joint_angle_lower))
flog.write('target_joint_angle_upper:%s\n' % str(joint_angle_upper))

success, is_open, no_move = utils.check_success(
    start_joint_angle=start_joint_angle,
    final_joint_angle=final_joint_angle,
    limit_l=joint_angle_lower,
    limit_r=joint_angle_upper,
    trajectoty=trajectory1,
    primact_type=primact_type,
    state=args.target_part_state,
)

out_info['success'] = 'True' if success else 'False'

flog.write('result: %s\n' % str(success))
flog.write('is_open: %s\n' % str(is_open))
flog.write('no_move: %s\n' % str(no_move))

flog.write('after check success')

env.scene.remove_articulation(robot1.robot)
env.scene.remove_articulation(robot2.robot)
env.scene.remove_articulation(env.object)
env.close()
# save results
with open(os.path.join(out_path, 'result.json'), 'w') as fout:
    print(f'save to {out_path}')
    json.dump(out_info, fout, indent=4)

if success:
    save_succ_data(out_info, cam_XYZA_list, gt_all_link_mask, gif_imgs, rgb_img)
    exit(0)
else:
    save_fail_data(out_info, cam_XYZA_list, gt_all_link_mask, gif_imgs, rgb_img)
    exit(1)
