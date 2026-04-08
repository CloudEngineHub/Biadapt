import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d

print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser
import torch
import time
import random
import imageio
from subprocess import call
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import math
import datetime
import h5py

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--exp_suffix', type=str, help='exp suffix')
parser.add_argument('--categories',
                    type=str,
                    help='list all categories [Default: None, meaning all 10 categories]',
                    default=None)
parser.add_argument('--cat2freq', type=str, default=None)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--val_data_dir', type=str)
parser.add_argument('--val_buffer_max_num', type=int, default=100)
parser.add_argument('--out_folder', type=str, default='xxx')

parser.add_argument('--coordinate_system', type=str, default='cam')  #TODO
parser.add_argument('--repeat_num', type=int, default=1)

parser.add_argument('--aff1_version', type=str, default=None)
parser.add_argument('--aff1_path', type=str, default=None)
parser.add_argument('--aff1_eval_epoch', type=str, default=None)
parser.add_argument('--actor1_version', type=str, default=None, help='model def file')
parser.add_argument('--actor1_path', type=str)
parser.add_argument('--actor1_eval_epoch', type=str)
parser.add_argument('--critic1_version', type=str, default=None, help='model def file')
parser.add_argument('--critic1_path', type=str)
parser.add_argument('--critic1_eval_epoch', type=str)

parser.add_argument('--aff2_version', type=str, default=None)
parser.add_argument('--aff2_path', type=str, default=None)
parser.add_argument('--aff2_eval_epoch', type=str, default=None)
parser.add_argument('--actor2_version', type=str, default=None, help='model def file')
parser.add_argument('--actor2_path', type=str)
parser.add_argument('--actor2_eval_epoch', type=str)
parser.add_argument('--critic2_version', type=str, default=None, help='model def file')
parser.add_argument('--critic2_path', type=str)
parser.add_argument('--critic2_eval_epoch', type=str)

parser.add_argument('--use_CA', action='store_true', default=False)
parser.add_argument('--CA_path', type=str)
parser.add_argument('--CA_eval_epoch', type=str)

parser.add_argument('--rv_cnt', type=int, default=100)
parser.add_argument('--rvs_ctpt', type=int, default=10)
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--task_feat_dim', type=int, default=32)
parser.add_argument('--cp_feat_dim', type=int, default=32)
parser.add_argument('--dir_feat_dim', type=int, default=32)
parser.add_argument('--num_point_per_shape', type=int, default=8192)

parser.add_argument('--aff1_topk', type=float, default=0.1)
parser.add_argument('--aff2_topk', type=float, default=0.1)
parser.add_argument('--aff_topk', type=float, default=0.1)
parser.add_argument('--critic_topk1', type=float, default=0.01)
parser.add_argument('--critic_topk', type=float, default=0.01)
parser.add_argument('--num_ctpt1', type=int, default=10)
parser.add_argument('--num_ctpt2', type=int, default=10)
parser.add_argument('--rv1', type=int, default=100)
parser.add_argument('--rv2', type=int, default=100)
parser.add_argument('--num_pair1', type=int, default=10)
parser.add_argument('--num_ctpts', type=int, default=10)
parser.add_argument('--rvs', type=int, default=100)

parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--static_friction', type=float, default=4.0)
parser.add_argument('--dynamic_friction', type=float, default=4.0)

parser.add_argument('--draw_aff_map', action='store_true', default=False)

parser.add_argument('--vis_action', action='store_true', default=False)
parser.add_argument('--num_draw', type=int, default=1)
parser.add_argument('--num_processes', type=int, default=1)

parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")

K = [[918.583046944134, 0, 648.5641804535674], [0, 918.583046944134, 348.43171438920723], [0, 0, 1]]

CAM_In_mat = np.array(K)
K = CAM_In_mat
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
cam_w = 2 * cx
cam_h = 2 * cy

file_path = "/home/zjinxian/Biadapt/RealData/Real_Obj_mesh/Box_02/Box_02/optimized_X_BaseCamera.npy"  # 替换为你的文件路径
extrinsics = np.load(file_path)


def pos_world2cam(position):
    position = np.array(position)
    position = np.append(position, 1.0)
    position_cam = extrinsics @ position
    position_cam = position_cam[:3]

    return position_cam


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def downsample_points(points, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)


def upsample_points(points, target_num):
    """
    对点云进行上采样，使其达到目标数量。
    :param points: (N, 3) 原始点云
    :param target_num: int 目标点数
    :return: (target_num, 3) 上采样后的点云
    """
    current_num = points.shape[0]
    if current_num >= target_num:
        return points  # 如果点数已足够，直接返回

    # 随机采样索引来重复点
    idx = np.random.choice(current_num, target_num - current_num, replace=True)
    extra_points = points[idx]

    # 拼接原始点和补充点
    upsampled_points = np.vstack([points, extra_points])
    return upsampled_points


def cal_rotmat(up, forward):
    rotmat = np.eye(3).astype(np.float32)
    up /= np.linalg.norm(up)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    return np.array(rotmat)


# def visualize_action(pcs, position1, up1, forward1, position2, up2, forward2, save_dir=None):

#     # 构建 Open3D 点云
#     pcd = o3d.geometry.PointCloud()
#     # 确保点云数据是 NumPy 格式
#     if isinstance(pcs, torch.Tensor):
#         pcs = pcs.cpu().numpy()  # 转换为 NumPy 数组

#     pcd.points = o3d.utility.Vector3dVector(pcs)

#     rotmat1 = cal_rotmat(up1, forward1)
#     print('rotmat1:', rotmat1)
#     # print()
#     rotmat2 = cal_rotmat(up2, forward2)

#     arrow_forward1 = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.01,
#                                                             cone_height=0.03,
#                                                             cylinder_radius=0.005,
#                                                             cylinder_height=0.1)
#     print('forward1 OK')
#     arrow_up1 = arrow_forward1.translate([0, 0, 0])  # 复制箭头
#     print('up1 OK')
#     arrow_forward1.paint_uniform_color([1, 0, 0])  # Forward: 红色
#     arrow_up1.paint_uniform_color([0, 1, 0])  # Up: 绿色

#     # 调整箭头方向
#     # 旋转箭头，使其朝向forward和up方向
#     arrow_forward1.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
#         np.cross([0, 0, 1], forward1) * np.arccos(np.dot([0, 0, 1], forward1))),
#                           center=(0, 0, 0))
#     arrow_up1.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
#         np.cross([0, 0, 1], up1) * np.arccos(np.dot([0, 0, 1], up1))),
#                      center=(0, 0, 0))
#     # 平移到物体点位置
#     arrow_forward1.translate(position1)
#     arrow_up1.translate(position1)

#     arrow_forward2 = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.01,
#                                                             cone_height=0.03,
#                                                             cylinder_radius=0.005,
#                                                             cylinder_height=0.1)
#     arrow_up2 = arrow_forward1.translate([0, 0, 0])  # 复制箭头

#     arrow_forward2.paint_uniform_color([1, 0, 0])  # Forward: 红色
#     arrow_up2.paint_uniform_color([0, 1, 0])  # Up: 绿色
#     print('forward2 up2 OK')
#     # 调整箭头方向
#     # 旋转箭头，使其朝向forward和up方向
#     arrow_forward2.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
#         np.cross([0, 0, 1], forward2) * np.arccos(np.dot([0, 0, 1], forward2))),
#                           center=(0, 0, 0))
#     arrow_up2.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
#         np.cross([0, 0, 1], up2) * np.arccos(np.dot([0, 0, 1], up2))),
#                      center=(0, 0, 0))
#     # 平移到物体点位置
#     arrow_forward2.translate(position2)
#     arrow_up2.translate(position2)
#     print('here')
#     # 使用 Visualizer 渲染
#     vis = o3d.visualization.Visualizer()
#     # vis.create_window()

#     # 添加几何体
#     vis.add_geometry(pcd)
#     vis.add_geometry(arrow_forward1)
#     vis.add_geometry(arrow_up1)
#     vis.add_geometry(arrow_forward2)
#     vis.add_geometry(arrow_up2)

#     # 渲染并保存截图
#     if save_dir is not None:
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)
#         vis.poll_events()
#         vis.update_renderer()
#         vis.capture_screen_image(os.path.join(save_dir, "visualized_image.png"))

#     # 关闭窗口
#     vis.run()
#     vis.destroy_window()

#     # 保存点云到文件
#     if save_dir is not None:
#         o3d.io.write_point_cloud(os.path.join(save_dir, "Point_with_arrows.ply"), pcd)


def visualize_action(pcs, position1, up1, forward1, position2, up2, forward2, save_dir=None):
    # 将点云转换为 NumPy 数组
    if isinstance(pcs, torch.Tensor):
        pcs = pcs.cpu().numpy()  # 转换为 NumPy 数组

    # 创建一个 3D 图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c='b', marker='o', s=1, label='Point Cloud')

    # 绘制第一个箭头（forward 和 up）
    ax.quiver(position1[0],
              position1[1],
              position1[2],
              forward1[0],
              forward1[1],
              forward1[2],
              color='r',
              length=0.1,
              label='Forward 1')
    ax.quiver(position1[0], position1[1], position1[2], up1[0], up1[1], up1[2], color='g', length=0.1, label='Up 1')

    # # 绘制第二个箭头（forward 和 up）
    # ax.quiver(position2[0],
    #           position2[1],
    #           position2[2],
    #           forward2[0],
    #           forward2[1],
    #           forward2[2],
    #           color='r',
    #           length=0.1,
    #           label='Forward 2')
    # ax.quiver(position2[0], position2[1], position2[2], up2[0], up2[1], up2[2], color='g', length=0.1, label='Up 2')

    # 设置图形的标签和显示
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud and Arrows')

    # 显示图例
    ax.legend()
    plt.show()
    # 保存图像
    plt.savefig(f'{save_dir}/visualized_action.png')

    # 关闭图形
    plt.close()


def run_jobs(idx_process, args, trans_q, cur_file_list, out_dir):
    device = args.device if torch.cuda.is_available() else "cpu"

    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)

    primact_type = args.primact_type
    task_input_dim = 1

    # load models
    aff1_def = utils.get_model_module(args.aff1_version)
    affordance1 = aff1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)
    actor1_def = utils.get_model_module(args.actor1_version)
    actor1 = actor1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim)
    critic1_def = utils.get_model_module(args.critic1_version)
    critic1 = critic1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)

    aff2_def = utils.get_model_module(args.aff2_version)
    affordance2 = aff2_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)
    actor2_def = utils.get_model_module(args.actor2_version)
    actor2 = actor2_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim)
    critic2_def = utils.get_model_module(args.critic2_version)
    critic2 = critic2_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)

    if not args.use_CA:
        affordance1.load_state_dict(
            torch.load(os.path.join(args.aff1_path, 'ckpts', '%s-network.pth' % args.aff1_eval_epoch)))
        actor1.load_state_dict(
            torch.load(os.path.join(args.actor1_path, 'ckpts', '%s-network.pth' % args.actor1_eval_epoch)))
        critic1.load_state_dict(
            torch.load(os.path.join(args.critic1_path, 'ckpts', '%s-network.pth' % args.critic1_eval_epoch)))
        affordance2.load_state_dict(
            torch.load(os.path.join(args.aff2_path, 'ckpts', '%s-network.pth' % args.aff2_eval_epoch)))
        actor2.load_state_dict(
            torch.load(os.path.join(args.actor2_path, 'ckpts', '%s-network.pth' % args.actor2_eval_epoch)))
        critic2.load_state_dict(
            torch.load(os.path.join(args.critic2_path, 'ckpts', '%s-network.pth' % args.critic2_eval_epoch)))
    else:
        affordance1.load_state_dict(
            torch.load(os.path.join(args.CA_path, 'ckpts', '%s-affordance1.pth' % args.CA_eval_epoch)))
        actor1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-actor1.pth' % args.CA_eval_epoch)))
        critic1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic1.pth' % args.CA_eval_epoch)))
        affordance2.load_state_dict(
            torch.load(os.path.join(args.CA_path, 'ckpts', '%s-affordance2.pth' % args.CA_eval_epoch)))
        actor2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-actor2.pth' % args.CA_eval_epoch)))
        critic2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic2.pth' % args.CA_eval_epoch)))

    affordance1.to(device).eval()
    actor1.to(device).eval()
    critic1.to(device).eval()
    affordance2.to(device).eval()
    actor2.to(device).eval()
    critic2.to(device).eval()
    for file in cur_file_list:
        for repeat_i in range(args.repeat_num):
            torch.cuda.empty_cache()
            batch_size = 1

            # file_dir, file_id = file
            obj_path = os.path.join(args.val_data_dir, file)
            file_dir = args.val_data_dir
            file_id = file.split('.')[0]
            category = 'Box'
            shape_id = '02'

            mesh = trimesh.load(obj_path)
            points = np.array(mesh.vertices)  #TODO
            print('points shape', points.shape)
            target_num = 448 * 448  # 需要的总点数
            upsampled_points = upsample_points(points, target_num)

            # 确认点数
            print("Upsampled points shape:", upsampled_points.shape)  # (200704, 3)
            points = upsampled_points
            grid_size = int(np.sqrt(points.shape[0]))  # 假设网格接近正方形
            if grid_size * grid_size != points.shape[0]:
                raise ValueError("点云数量无法直接映射到二维网格，请调整方法。")

            cam_XYZA_id1, cam_XYZA_id2 = np.divmod(np.arange(points.shape[0]), grid_size)
            cam_XYZA_pts = points

            save_aff_dir = os.path.join(out_dir, 'affordance_maps')
            # pcs, pc_center, pc_numpy = utils.get_part_pc(cam_XYZA_id1,
            #                                              cam_XYZA_id2,
            #                                              cam_XYZA_pts,
            #                                              coordinate_system=args.coordinate_system,
            #
            #  mat44=extrinsics)
            pcs, _, _ = utils.get_real_pc(cam_XYZA_id1,
                                          cam_XYZA_id2,
                                          cam_XYZA_pts,
                                          coordinate_system=args.coordinate_system,
                                          mat44=extrinsics)
            # torch.from_numpy(pcs).unsqueeze(0)
            pcs = pcs.float().contiguous().to(device)
            input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(
                -1)  # BN
            input_pcid2 = furthest_point_sample(pcs, args.num_point_per_shape).long().reshape(-1)  # BN
            pcs = pcs[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
            ''' inference '''
            num_ctpt1, num_ctpt2, rv1, rv2 = args.num_ctpt1, args.num_ctpt2, args.rv1, args.rv2
            num_pair1 = args.num_pair1
            print('here inference')
            # with torch.no_grad():
            #     # print('aff1')
            #     aff_scores1 = affordance1.inference_whole_pc(pcs).view(batch_size, args.num_point_per_shape)  # B * N
            #     aff_sorted_idx = torch.argsort(aff_scores1, dim=1,
            #                                    descending=True).view(batch_size, args.num_point_per_shape)
            #     batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
            #     selected_idx_idx = torch.randint(0,
            #                                      int(args.num_point_per_shape * args.aff_topk),
            #                                      size=(batch_size, num_ctpt1))
            #     selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]
            #     position1s = pcs.clone()[batch_idx, selected_idx].view(batch_size * num_ctpt1, 3)

            #     # print('actor1')
            #     dir1s = actor1.actor_sample_n_finetune(pcs, position1s, rvs_ctpt=num_ctpt1,
            #                                            rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, 6)

            #     # print('critic1')
            #     critic_scores = critic1.forward_n_finetune(pcs, position1s, dir1s, rvs_ctpt=num_ctpt1,
            #                                                rvs=rv1).view(batch_size, num_ctpt1 * rv1)
            #     critic_sorted_idx = torch.argsort(critic_scores, dim=1,
            #                                       descending=True).view(batch_size, num_ctpt1 * rv1)
            #     batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
            #     selected_idx_idx = torch.randint(0,
            #                                      int(num_ctpt1 * rv1 * args.critic_topk1),
            #                                      size=(batch_size, num_pair1))
            #     selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
            #     position1 = position1s.view(batch_size, num_ctpt1,
            #                                 3)[batch_idx, selected_idx // rv1].view(batch_size * num_pair1, 3)
            #     dir1 = dir1s.view(batch_size, num_ctpt1 * rv1, 6)[batch_idx,
            #                                                       selected_idx].view(batch_size * num_pair1, 6)

            #     # print('aff2')
            #     aff_scores = affordance2.inference_whole_pc_finetune(pcs, position1, dir1, rvs_ctpt=num_pair1).view(
            #         batch_size * num_pair1, args.num_point_per_shape)
            #     aff_sorted_idx = torch.argsort(aff_scores, dim=1,
            #                                    descending=True).view(batch_size * num_pair1, args.num_point_per_shape)
            #     batch_idx = torch.tensor(range(batch_size * num_pair1)).view(batch_size * num_pair1, 1)
            #     selected_idx_idx = torch.randint(0,
            #                                      int(args.num_point_per_shape * args.aff_topk),
            #                                      size=(batch_size * num_pair1, num_ctpt2))
            #     selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]
            #     expanded_pcs = pcs.repeat(1, num_pair1, 1).reshape(batch_size * num_pair1, args.num_point_per_shape, 3)
            #     position2 = expanded_pcs.clone()[batch_idx, selected_idx].view(batch_size * num_pair1 * num_ctpt2, 3)

            #     # print('actor2')
            #     expanded_position1 = position1.unsqueeze(dim=1).repeat(1, num_ctpt2,
            #                                                            1).reshape(batch_size * num_pair1 * num_ctpt2, 3)
            #     expanded_dir1s = dir1.unsqueeze(dim=1).repeat(1, num_ctpt2,
            #                                                   1).reshape(batch_size * num_pair1 * num_ctpt2, 6)
            #     dir2s = actor2.actor_sample_n_finetune(pcs,
            #                                            expanded_position1,
            #                                            position2,
            #                                            expanded_dir1s,
            #                                            rvs_ctpt=num_pair1 * num_ctpt2,
            #                                            rvs=rv2).contiguous().view(
            #                                                batch_size * num_pair1 * num_ctpt2 * rv2, 6)

            #     # print('critic2')
            #     expanded_expanded_dir1s = expanded_dir1s.unsqueeze(dim=1).repeat(1, rv2, 1).reshape(
            #         batch_size * num_pair1 * num_ctpt2 * rv2, 6)
            #     critic_scores = critic2.forward_n_finetune(pcs,
            #                                                expanded_position1,
            #                                                position2,
            #                                                expanded_expanded_dir1s,
            #                                                dir2s,
            #                                                rvs_ctpt=num_pair1 * num_ctpt2,
            #                                                rvs=rv2).view(batch_size, num_pair1 * num_ctpt2 * rv2)
            #     critic_sorted_idx = torch.argsort(critic_scores, dim=1,
            #                                       descending=True).view(batch_size, num_pair1 * num_ctpt2 * rv2)
            #     batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
            #     selected_idx_idx = torch.randint(0,
            #                                      int(num_pair1 * num_ctpt2 * rv2 * args.critic_topk),
            #                                      size=(batch_size, 1))
            #     selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]

            #     pred_scores = critic_scores[batch_idx, selected_idx]

            #     position1 = expanded_position1.reshape(batch_size, num_pair1 * num_ctpt2,
            #                                            3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
            #     position2 = position2.reshape(batch_size, num_pair1 * num_ctpt2,
            #                                   3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
            #     dir1 = expanded_expanded_dir1s.view(batch_size, num_pair1 * num_ctpt2 * rv2,
            #                                         6)[batch_idx, selected_idx].view(batch_size, 6)
            #     dir2 = dir2s.view(batch_size, num_pair1 * num_ctpt2 * rv2, 6)[batch_idx,
            #                                                                   selected_idx].view(batch_size, 6)

            raw_pair = 1
            ctpt1_world = [0.6688488916024606, -0.06012317213691065, 0.0439701776764304]
            ctpt2_world = [0.5495145022896135, -0.036597775960899526, 0.1202538848978802]

            ctpt1_cam = pos_world2cam(ctpt1_world)
            print('ctpt1_cam:', ctpt1_cam)
            print('ctpt1_world:', ctpt1_world)
            ctpt2_cam = pos_world2cam(ctpt2_world)
            print('ctpt2_cam:', ctpt2_cam)
            print('ctpt2_world:', ctpt2_world)

            ctpt1_list = [ctpt1_world]
            ctpt2_list = [ctpt2_world]

            ctpt1 = torch.tensor(np.array(ctpt1_list)).float().reshape(batch_size * raw_pair, -1).to(args.device)
            ctpt2 = torch.tensor(np.array(ctpt2_list)).float().reshape(batch_size * raw_pair, -1).to(args.device)
            ''' inference '''
            num_ctpt1, num_ctpt2, rv1, rv2 = raw_pair, raw_pair, args.rv1, args.rv2
            num_pair1 = args.num_pair1

            with torch.no_grad():

                position1s = ctpt1.reshape(batch_size, -1, 3).view(batch_size * num_ctpt1, 3)
                position2s = ctpt2.reshape(batch_size, -1, 3).view(batch_size * num_ctpt1, 3)

                #actor1
                dir1s = actor1.actor_sample_n_finetune(pcs, position1s, rvs_ctpt=num_ctpt1,
                                                       rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, 6)
                #critic1
                critic_scores = critic1.forward_n_finetune(pcs, position1s, dir1s, rvs_ctpt=num_ctpt1,
                                                           rvs=rv1).view(batch_size, num_ctpt1 * rv1)
                critic_sorted_idx = torch.argsort(critic_scores, dim=1,
                                                  descending=True).view(batch_size, num_ctpt1 * rv1)
                batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
                selected_idx_idx = torch.randint(0,
                                                 int(num_ctpt1 * rv1 * args.critic_topk1),
                                                 size=(batch_size, num_pair1))
                selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
                # print('position1s:', position1s.shape)
                position1 = position1s.view(batch_size, num_ctpt1,
                                            3)[batch_idx, selected_idx // rv1].view(batch_size * num_pair1, 3)
                position2 = position2s.view(batch_size, num_ctpt1,
                                            3)[batch_idx, selected_idx // rv1].view(batch_size * num_pair1, 3)

                dir1 = dir1s.view(batch_size, num_ctpt1 * rv1, 6)[batch_idx,
                                                                  selected_idx].view(batch_size * num_pair1, 6)

                expanded_dir1s = dir1.unsqueeze(dim=1).repeat(1, rv2, 1).reshape(batch_size * num_pair1 * rv2, 6)
                expanded_position1 = position1.unsqueeze(dim=1).repeat(1, rv2,
                                                                       1).reshape(batch_size * num_pair1 * rv2, 3)
                expanded_position2 = position2.unsqueeze(dim=1).repeat(1, rv2,
                                                                       1).reshape(batch_size * num_pair1 * rv2, 3)

                #actor2
                dir2s = actor2.actor_sample_n_finetune(pcs, position1, position2, dir1, rvs_ctpt=num_pair1,
                                                       rvs=rv2).contiguous().view(batch_size * num_pair1 * rv2, 6)
                # print('position1:', position1.shape, position1)
                # print('critic2')
                critic_scores = critic2.forward_n_finetune(pcs,
                                                           position1,
                                                           position2,
                                                           expanded_dir1s,
                                                           dir2s,
                                                           rvs_ctpt=num_pair1,
                                                           rvs=rv2).view(batch_size, num_pair1 * rv2)
                critic_sorted_idx = torch.argsort(critic_scores, dim=1,
                                                  descending=True).view(batch_size, num_pair1 * rv2)
                batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
                selected_idx_idx = torch.randint(0,
                                                 max(1, int(num_ctpt2 * rv2 * args.critic_topk)),
                                                 size=(batch_size, 1))
                selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
                pred_scores = critic_scores[batch_idx, selected_idx]

            with torch.no_grad():
                position1 = expanded_position1.view(batch_size, num_pair1 * rv2, 3)[batch_idx,
                                                                                    selected_idx].view(batch_size, 3)
                position2 = expanded_position2.view(batch_size, num_pair1 * rv2, 3)[batch_idx,
                                                                                    selected_idx].view(batch_size, 3)
                dir1 = expanded_dir1s.view(batch_size, num_pair1 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)
                dir2 = dir2s.view(batch_size, num_pair1 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)

            with torch.no_grad():
                print('here draw')
                if args.draw_aff_map and repeat_i < args.num_draw:
                    aff_scores = affordance1.inference_whole_pc(pcs).detach().cpu().numpy().reshape(-1)
                    aff_scores = 1 / (1 + np.exp(-(aff_scores - 0.5) * 15))
                    fn = os.path.join(save_aff_dir, '%s_%s_%s_%s' % (str(repeat_i), category, str(shape_id), 'map1'))
                    utils.draw_real_aff_map(
                        fn,
                        extrinsics,
                        pcs[0].detach().cpu().numpy(),
                        aff_scores,
                        coordinate_system=args.coordinate_system,
                        ctpt1=np.array(ctpt1_world),
                        # ctpt1=position1[0].detach().cpu().numpy(),
                        type='0')
                    aff_scores = affordance2.inference_whole_pc(pcs, position1,
                                                                dir1).detach().cpu().numpy().reshape(-1)  # B * N * 1
                    aff_scores = 1 / (1 + np.exp(-(aff_scores - 0.5) * 15))
                    fn = os.path.join(save_aff_dir, '%s_%s_%s_%s' % (str(repeat_i), category, str(shape_id), 'map2'))
                    utils.draw_real_aff_map(
                        fn,
                        extrinsics,
                        pcs[0].detach().cpu().numpy(),
                        aff_scores,
                        coordinate_system=args.coordinate_system,
                        ctpt1=np.array(ctpt1_world),
                        ctpt2=np.array(ctpt2_world),
                        # ctpt1=position1[0].detach().cpu().numpy(),
                        # ctpt2=position2[0].detach().cpu().numpy(),
                        type='2')

                dir1 = dir1.view(6).detach().cpu().numpy()
                dir2 = dir2.view(6).detach().cpu().numpy()

            position1 = position1.view(3).detach().cpu().numpy()
            position2 = position2.view(3).detach().cpu().numpy()
            up1, forward1 = dir1[0:3], dir1[3:6]
            up2, forward2 = dir2[0:3], dir2[3:6]
            print('position1:', position1, 'ctpt1_world:', ctpt1_world)
            print('up1, forward1 :', up1, forward1)
            print('up2, forward2 :', up2, forward2)
            if args.vis_action:
                visualize_action(pcs[0].detach().cpu().numpy(),
                                 position1,
                                 up1,
                                 forward1,
                                 position2,
                                 up2,
                                 forward2,
                                 save_dir=args.out_dir)
            trans_q.put([1, file_id, repeat_i])


def main():
    if args.use_CA:
        out_dir = os.path.join(args.CA_path, args.out_folder)
    else:
        out_dir = os.path.join(args.actor2_path, args.out_folder)
    print('out_dir: ', out_dir)
    os.makedirs(os.path.join(out_dir, 'affordance_maps'), exist_ok=True)
    args.out_dir = out_dir
    file_list = []
    total_file = 0

    #TODO
    file = 'combined_point_cloud.ply'
    file_list = [file]
    total_file = 1

    # for file in os.listdir(args.val_data_dir):
    #     if file.endswith(".obj"):
    #         file_list.append(file)
    #         total_file += 1
    #     else:
    #         continue
    #     if total_file >= args.val_buffer_max_num:
    #         break
    num_file_per_process = total_file // args.num_processes + 1
    trans_q = ctx.Queue()
    for idx_process in range(args.num_processes):
        cur_file_list = file_list[idx_process * num_file_per_process:min((idx_process + 1) *
                                                                         num_file_per_process, total_file)]
        p = ctx.Process(target=run_jobs, args=(idx_process, args, trans_q, cur_file_list, out_dir))
        p.start()
    t0 = time.time()
    t_begin = datetime.datetime.now()
    total = 0
    while True:
        if not trans_q.empty():
            results = trans_q.get()
            result, file_id, repeat_id = results
            total += 1
            print(f'finished {file_id} for repeat {repeat_id} ')
            t0 = time.time()

            if total >= total_file * args.repeat_num:
                exit(0)


if __name__ == '__main__':
    sys.exit(main())
