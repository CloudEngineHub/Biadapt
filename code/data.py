import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
# from pyquaternion import Quaternion
from camera import Camera
import random
import utils
from utils import get_6d_rot_loss
import copy
import ipdb


class SAPIENVisionDataset(data.Dataset):

    def __init__(self,
                 primact_types,
                 category_types,
                 data_features,
                 buffer_max_num,
                 img_size=224,
                 only_true_data=False,
                 succ_proportion=0.4,
                 fail_proportion=0.8,
                 coordinate_system='world',
                 exchange_ctpts=False,
                 find_task_for_invalid=True,
                 cat2freq=None,
                 no_true_false_equal=False,
                 is_original=True,
                 explore_proportion=0,
                 mode='all'):

        self.primact_types = primact_types
        self.category_types = category_types
        self.data_features = data_features
        self.buffer_max_num = buffer_max_num
        self.img_size = img_size
        self.only_true_data = only_true_data
        self.succ_proportion = succ_proportion
        self.fail_proportion = fail_proportion
        self.coordinate_system = coordinate_system
        self.exchange_ctpts = exchange_ctpts
        self.find_task_for_invalid = find_task_for_invalid
        self.is_original = is_original
        self.explore_proportion = explore_proportion
        self.mode = mode

        self.dataset = dict()
        for primact_type in primact_types:
            self.dataset[primact_type] = []

        self.shape_dataset = dict()

        # data buffer
        self.success_data = dict()
        self.fail_data = dict()
        self.invalid_data = dict()
        # for primact_type in primact_types:
        #     self.success_data[primact_type] = []
        #     self.fail_data[primact_type] = []
        #     self.invalid_data[primact_type] = []

        cat_list, shape_list, _, _ = utils.get_shape_list(all_categories=category_types,
                                                          mode=mode,
                                                          primact_type=self.primact_types[0])
        self.shape_list = shape_list
        self.category_dict = {'success': {}, 'fail': {}, 'invalid': {}}
        self.shape_dict = {'success': {}, 'fail': {}, 'invalid': {}}

        for cat in cat_list:
            self.category_dict['success'][cat] = 0
            self.category_dict['fail'][cat] = 0
            self.category_dict['invalid'][cat] = 0

        for shape in shape_list:
            self.shape_dict['success'][str(shape)] = 0
            self.shape_dict['fail'][str(shape)] = 0
            self.shape_dict['invalid'][str(shape)] = 0
            self.shape_dataset[str(shape)] = []

            self.success_data[str(shape)] = []
            self.fail_data[str(shape)] = []
            self.invalid_data[str(shape)] = []

        self.freq_dict = dict()

        if cat2freq:
            freqs = [int(x) for x in cat2freq.split(',')]
            for idx in range(len(cat_list)):
                self.freq_dict[cat_list[idx]] = freqs[idx]
        else:
            for idx in range(len(cat_list)):
                self.freq_dict[cat_list[idx]] = 1e4

        self.shape_freq_dict = dict()

        for idx in range(len(shape_list)):
            self.shape_freq_dict[str(shape_list[idx])] = int(1.3 * buffer_max_num / len(shape_list))

    def get_seq(self):
        length = len(self.dataset) // 32
        self.seq = np.arange(length)
        np.random.shuffle(self.seq)

    def load_data(self, data_list, flog=None):
        bar = ProgressBar()
        num_success, num_fail, num_invalid = 0, 0, 0
        num_explore, num_experience = 0, 0
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]
            print('cur_dir:', cur_dir)
            if not os.path.exists(cur_dir):
                print('No Exist PATH:', cur_dir)
                continue
            print('PATH Exist:',cur_dir)
            for root, dirs, files in os.walk(cur_dir):
                for file in sorted(files):
                    if 'json' not in file:
                        continue
                    mix_id = file.split('.')[0][7:]
                    trial_id = str(mix_id)

                    with open(os.path.join(cur_dir, 'result_%s.json' % mix_id), 'r') as fin:
                        try:
                            result_data = json.load(fin)
                        except Exception:
                            continue
                    if not 'shape_id' in result_data:
                        continue
                    shape_id = int(result_data['shape_id'])

                    category = result_data['category']
                    cur_primact_type = result_data['primact_type']
                    # gif_file_name = ''
                    if 'object_state' in result_data:
                        object_state = result_data['object_state']
                    elif 'state' in result_data:
                        object_state = result_data['state']
                    else:
                        object_state = 'closed'
                    # target_joint_id = int(result_data["target_object_part_joint_id"])
                    is_explore = False
                    if 'is_explore' in result_data:
                        if result_data['is_explore'] == 'True':
                            is_explore = True

                    img_dir = cur_dir.replace('files', 'img')
                    img_path = os.path.join(img_dir,
                                            str(mix_id) + '_' + str(category) + '_' + str(shape_id) + '_rgb.png')
                    gif_dir = cur_dir.replace('files', 'gif')
                    gif_path = os.path.join(gif_dir, str(mix_id) + '_' + str(category) + '_' + str(shape_id) + '.gif')
                    
                    
                    if not os.path.exists(gif_path):
                        continue
                    if cur_primact_type not in self.primact_types:
                        continue
                    if str(shape_id) not in self.shape_list:

                        continue
                    if category not in self.category_types:

                        continue
                    # print('------------------------here---------------------------')
                    valid = False if 'invalid' in cur_dir else True
                    success = True if 'succ' in cur_dir else False
                    # print('success:',success)
                    if self.only_true_data and (num_success > self.buffer_max_num):
                        print('Only Sucess Full')
                        break
                    if self.only_true_data and (not success):
                        continue

                    if (not self.only_true_data) and success and (num_success >
                                                                  self.buffer_max_num * self.succ_proportion):
                        print('Success Full')
                        continue

                    if valid and (not success) and (num_fail > (num_success+num_fail) * (1 - self.succ_proportion)):
                        print('Fail Full')
                        continue

                    if (not valid) and (num_invalid > self.buffer_max_num * (1 - self.fail_proportion)):
                        print('Invalid Full')
                        continue

                    if success and self.category_dict['success'][category] >= self.freq_dict[category]:
                        continue
                    elif success and self.shape_dict['success'][str(shape_id)] >= self.shape_freq_dict[str(shape_id)]:
                        continue

                    elif (not valid) and self.category_dict['invalid'][
                            category] >= self.category_dict['success'][category] * (1 - self.succ_proportion):
                        continue

                    # if 'gripper_direction_world1' not in result_data.keys():
                    #     continue

                    if self.explore_proportion != 0:
                        if is_explore and success and self.shape_dict['success'][str(shape_id)] > 0:
                            continue
                        elif (not is_explore) and (
                                not success) and valid and self.shape_dict['fail'][str(shape_id)] > 0:
                            continue
                        elif (not is_explore) and (not success) and (
                                not valid) and self.shape_dict['invalid'][str(shape_id)] > 0:
                            continue

                    contact_point_world1 = np.array(result_data['position_world1'], dtype=np.float32)
                    # gripper_up_world1 = np.array(result_data['gripper_direction_world1'], dtype=np.float32)
                    gripper_forward_world1 = np.array(result_data['gripper_forward_direction_world1'], dtype=np.float32)
                    start_dist1 = np.array(result_data['start_dist1'], dtype=np.float32)
                    displacement1 = np.array(result_data['displacement1'], dtype=np.float32)
                    maneuver_dist1 = np.array(result_data['maneuver_dist1'], dtype=np.float32)
                    pre_rotmat_world1 = np.array(result_data['pre_rotmat_world1'], dtype=np.float32)
                    final_rotmat_world1 = np.array(result_data['final_rotmat_world1'], dtype=np.float32)
                    pre_pos1 = pre_rotmat_world1[:3, 3]
                    final_pos1 = final_rotmat_world1[:3, 3]

                    gripper_up_world1 = final_rotmat_world1[:3, 2]

                    contact_point_world2 = np.array(result_data['position_world2'], dtype=np.float32)
                    # gripper_up_world2 = np.array(result_data['gripper_direction_world2'], dtype=np.float32)
                    gripper_forward_world2 = np.array(result_data['gripper_forward_direction_world2'], dtype=np.float32)
                    start_dist2 = np.array(result_data['start_dist2'], dtype=np.float32)
                    displacement2 = np.array(result_data['displacement2'], dtype=np.float32)
                    maneuver_dist2 = np.array(result_data['maneuver_dist2'], dtype=np.float32)
                    pre_rotmat_world2 = np.array(result_data['pre_rotmat_world2'], dtype=np.float32)
                    final_rotmat_world2 = np.array(result_data['final_rotmat_world2'], dtype=np.float32)
                    pre_pos2 = pre_rotmat_world2[:3, 3]
                    final_pos2 = final_rotmat_world2[:3, 3]

                    gripper_up_world2 = final_rotmat_world2[:3, 2]

                    camera_metadata = result_data['camera_metadata']
                    mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
                    cam2cambase = np.array(camera_metadata['cam2cambase'], dtype=np.float32)

                    pixel_locs_1 = np.array(result_data['pixel_locs_1'], dtype=np.int32)
                    pixel_locs_2 = np.array(result_data['pixel_locs_2'], dtype=np.int32)

                    task = 0
                    gt_motion = 0

                    with h5py.File(os.path.join(cur_dir, 'cam_XYZA_%s.h5' % mix_id), 'r') as fin:
                        cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                        cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                        cam_XYZA_pts = fin['pc'][:].astype(np.float32)

                    _, pc_center = utils.get_part_pc(cam_XYZA_id1,
                                                     cam_XYZA_id2,
                                                     cam_XYZA_pts,
                                                     coordinate_system='cambase',
                                                     mat44=mat44,
                                                     cam2cambase=cam2cambase)

                    if self.coordinate_system == 'world':
                        contact_point1, gripper_up1, gripper_forward1, contact_point2, gripper_up2, gripper_forward2 = \
                            contact_point_world1, gripper_up_world1, gripper_forward_world1, contact_point_world2, gripper_up_world2, gripper_forward_world2
                    elif self.coordinate_system == 'cambase':
                        world_batch = [
                            contact_point_world1.copy(),
                            gripper_up_world1.copy(),
                            gripper_forward_world1.copy(),
                            contact_point_world2.copy(),
                            gripper_up_world2.copy(),
                            gripper_forward_world2.copy()
                        ]
                        is_pc = [True, False, False, True, False, False]
                        camera_batch = utils.batch_coordinate_transform(world_batch,
                                                                        is_pc,
                                                                        transform_type='world2cam',
                                                                        mat44=mat44)
                        cambase_batch = utils.batch_coordinate_transform(camera_batch,
                                                                         is_pc,
                                                                         transform_type='cam2cambase',
                                                                         cam2cambase=cam2cambase,
                                                                         pc_center=pc_center)
                        contact_point1, gripper_up1, gripper_forward1, contact_point2, gripper_up2, gripper_forward2 = cambase_batch

                    cur_data = (
                        cur_dir,
                        category,
                        shape_id,
                        trial_id,
                        mix_id,
                        img_path,
                        object_state,  # basic
                        contact_point1,
                        gripper_up1,
                        gripper_forward1,
                        pixel_locs_1,
                        start_dist1,
                        displacement1,
                        maneuver_dist1,
                        pre_pos1,
                        final_pos1,
                        contact_point2,
                        gripper_up2,
                        gripper_forward2,
                        pixel_locs_2,
                        start_dist2,
                        displacement2,
                        maneuver_dist2,
                        pre_pos2,
                        final_pos2,
                        camera_metadata,
                        mat44,
                        cam2cambase,
                        pc_center,  #camera
                        task,
                        gt_motion,
                        valid,
                        success,
                        self.is_original,
                        is_explore)  #external
                    rec_data = (
                        cur_dir,
                        category,
                        shape_id,
                        trial_id,
                        mix_id,
                        img_path,
                        object_state,
                        contact_point2,
                        gripper_up2,
                        gripper_forward2,
                        pixel_locs_2,
                        start_dist2,
                        displacement2,
                        maneuver_dist2,
                        pre_pos2,
                        final_pos2,
                        contact_point1,
                        gripper_up1,
                        gripper_forward1,
                        pixel_locs_1,
                        start_dist1,
                        displacement1,
                        maneuver_dist1,
                        pre_pos1,
                        final_pos1,
                        camera_metadata,
                        mat44,
                        cam2cambase,
                        pc_center,  #camera
                        task,
                        gt_motion,
                        valid,
                        success,
                        self.is_original,
                        is_explore)

                    if success:
                        self.category_dict['success'][category] += 1
                        self.shape_dict['success'][str(shape_id)] += 1

                        self.dataset[cur_primact_type].append(cur_data)
                        self.shape_dataset[str(shape_id)].append(cur_data)
                        self.success_data[str(shape_id)].append(cur_data)
                        num_success += 1
                        if is_explore:
                            num_explore += 1
                        else:
                            num_experience += 1
                        if self.exchange_ctpts:
                            self.category_dict['success'][category] += 1
                            self.shape_dict['success'][str(shape_id)] += 1
                            self.dataset[cur_primact_type].append(rec_data)
                            self.success_data[str(shape_id)].append(rec_data)
                            num_success += 1
                            if is_explore:
                                num_explore += 1
                            else:
                                num_experience += 1

                    elif valid and (not success):
                        if not self.only_true_data:

                            # if num_neg2 < self.buffer_max_num * (self.fail_proportion - self.succ_proportion):
                            self.category_dict['fail'][category] += 1

                            self.shape_dict['fail'][str(shape_id)] += 1
                            self.dataset[cur_primact_type].append(cur_data)
                            self.fail_data[str(shape_id)].append(cur_data)
                            num_fail += 1
                            if is_explore:
                                num_explore += 1
                            else:
                                num_experience += 1
                            if self.exchange_ctpts:
                                self.category_dict['fail'][category] += 1
                                self.shape_dict['fail'][str(shape_id)] += 1
                                self.dataset[cur_primact_type].append(rec_data)
                                self.fail_data[str(shape_id)].append(rec_data)
                                num_fail += 1
                                if is_explore:
                                    num_explore += 1
                                else:
                                    num_experience += 1

                    else:
                        if not self.only_true_data:
                            self.category_dict['invalid'][category] += 1
                            self.shape_dict['invalid'][str(shape_id)] += 1
                            self.dataset[cur_primact_type].append(cur_data)
                            self.invalid_data[str(shape_id)].append(cur_data)
                            num_invalid += 1
                            if is_explore:
                                num_explore += 1
                            else:
                                num_experience += 1
                            if self.exchange_ctpts:
                                self.category_dict['invalid'][category] += 1
                                self.shape_dict['invalid'][str(shape_id)] += 1
                                self.dataset[cur_primact_type].append(rec_data)
                                self.invalid_data[str(shape_id)].append(rec_data)
                                num_invalid += 1
                                if is_explore:
                                    num_explore += 1
                                else:
                                    num_experience += 1

        utils.printout(flog,
                       f'success data: %d; fail data: %d; invalid data: %d' % (num_success, num_fail, num_invalid))

        cur_succ_rate = num_success/(num_success + num_fail)

        utils.printout(flog,
                       f'success data: %d; fail data: %d; invalid data: %d' % (num_success, num_fail, num_invalid))
        utils.printout(flog, f'explore data: %d; experience data: %d' % (num_explore, num_experience))

        print('category distribution: \nsuccess:', self.category_dict['success'], '\nfail:', self.category_dict['fail'],
              '\ninvalid:', self.category_dict['invalid'])
        print('shape distribution: \nsuccess:', self.shape_dict['success'], '\nfail:', self.shape_dict['fail'],
              '\ninvalid:', self.shape_dict['invalid'])

    def __str__(self):
        strout = '[SAPIENVisionDataset %d] primact_types: %s, img_size: %d\n' % \
                (len(self), ','.join(self.primact_types), self.img_size)
        for primact_type in self.primact_types:
            # strout += '\t<%s> True: %d False: %d\n' % (primact_type, len(self.true_data[primact_type]), len(self.false_data[primact_type]))
            strout += '\t<%s>\n' % primact_type
        return strout

    def __len__(self):
        max_data = 0
        for primact_type in self.primact_types:
            max_data = max(max_data, len(self.dataset[primact_type]))
        return max_data * len(self.primact_types)

    def __getitem__(self, index):

        primact_id = index % len(self.primact_types)
        primact_type = self.primact_types[primact_id]
        index = index // len(self.primact_types)

        cur_dir,category, shape_id, trial_id, mix_id,img_path,object_state, \
        contact_point1,gripper_up1,gripper_forward1,pixel_locs_1,start_dist1,displacement1,maneuver_dist1,pre_pos1,final_pos1, \
        contact_point2,gripper_up2,gripper_forward2,pixel_locs_2,start_dist2,displacement2,maneuver_dist2,pre_pos2,final_pos2,\
        camera_metadata,mat44,cam2cambase,pc_center, \
        task, gt_motion, valid,success,is_original,is_explore = self.dataset[primact_type][index]

        # print(result, is_original)
        data_feats = ()
        for feat in self.data_features:

            if feat == 'img':
                with Image.open(img_path) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_pc':
                with h5py.File(os.path.join(cur_dir, 'cam_XYZA_%s.h5' % mix_id), 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
                mask = (out[:, :, 3] > 0.5)
                pc = out[mask, :3]
                idx = np.arange(pc.shape[0])
                np.random.shuffle(idx)
                while len(idx) < 30000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:30000 - 1]
                pc = pc[idx, :]
                pc[:, 0] -= 5

                pc_world = (mat44[:3, :3] @ pc.T).T
                if self.coordinate_system == 'world':  # cam2world
                    pc = (mat44[:3, :3] @ pc.T).T
                elif self.coordinate_system == 'cambase':  # cam2cambase
                    pc = pc @ np.transpose(cam2cambase, (1, 0))
                    pc_centers = (pc.max(axis=0, keepdims=True) + pc.min(axis=0, keepdims=True)) / 2
                    pc_centers = pc_centers[0]
                    pc -= pc_centers
                out = torch.from_numpy(pc).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'cur_dir':
                data_feats = data_feats + (cur_dir,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'primact_type':
                data_feats = data_feats + (primact_type,)

            elif feat == 'category':
                data_feats = data_feats + (category,)
            elif feat == 'object_state':
                data_feats = data_feats + (object_state,)

            elif feat == 'is_original':
                data_feats = data_feats + (is_original,)
            elif feat == 'is_explore':
                data_feats = data_feats + (is_explore,)
            elif feat == 'task':
                data_feats = data_feats + (task,)
            elif feat == 'gt_motion':
                data_feats = data_feats + (gt_motion,)
            elif feat == 'ctpt1':
                ctpt1 = contact_point1
                data_feats = data_feats + (ctpt1,)

            elif feat == 'ctpt2':
                ctpt2 = contact_point2
                data_feats = data_feats + (ctpt2,)

            elif feat == 'dir1':
                dir1 = np.concatenate([gripper_up1, gripper_forward1])
                data_feats = data_feats + (dir1,)

            elif feat == 'dir2':
                dir2 = np.concatenate([gripper_up2, gripper_forward2])
                data_feats = data_feats + (dir2,)

            elif feat == 'dist1':
                dist1 = np.array([start_dist1, displacement1, maneuver_dist1])
                data_feats = data_feats + (dist1,)

            elif feat == 'dist2':
                dist2 = np.array([start_dist2, displacement2, maneuver_dist2])
                data_feats = data_feats + (dist2,)
            elif feat == 'pre_pos1':
                data_feats = data_feats + (pre_pos1,)
            elif feat == 'pre_pos2':
                data_feats = data_feats + (pre_pos2,)
            elif feat == 'final_pos1':
                data_feats = data_feats + (final_pos1,)
            elif feat == 'final_pos2':
                data_feats = data_feats + (final_pos2,)
            elif feat == 'pc_centers':
                data_feats = data_feats + (pc_center,)

            elif feat == 'valid':
                data_feats = data_feats + (valid,)

            elif feat == 'success':
                data_feats = data_feats + (success,)

            elif feat == 'trial_id':  #
                data_feats = data_feats + (trial_id,)
            elif feat == 'mix_id':  #
                data_feats = data_feats + (mix_id,)

            elif feat == 'camera_metadata':
                data_feats = data_feats + (camera_metadata,)

            elif feat == 'pixel_locs_1':
                data_feats = data_feats + (pixel_locs_1,)

            elif feat == 'pixel_locs_2':
                data_feats = data_feats + (pixel_locs_2,)

            elif feat == 'cam2cambase':
                data_feats = data_feats + (cam2cambase,)

            elif feat == 'mat44':
                data_feats = data_feats + (mat44,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats
