"""
    an RGB-D camera
"""
import numpy as np
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageColor
import os


class Camera(object):

    def __init__(self, env, near=0.1, far=100.0, image_size1=448, image_size2=448, dist=5.0, \
                 phi=np.pi / 10, theta=np.pi, fov=35, random_position=False, fixed_position=False, restrict_dir=False,
                 real_data=False, have_table = False, table_height=1.0, pos_rand_ratio=0.2, object_centered=False, mat44=None, cam2cambase=None):
        # builder = env.scene.create_actor_builder()
        # camera_mount_actor = builder.build_kinematic()
        self.env = env
        self.camera = env.scene.add_camera(name="camera",
                                           width=image_size1,
                                           height=image_size2,
                                           fovy=np.deg2rad(fov),
                                           near=near,
                                           far=far)
        self.have_table = have_table
        self.table_height = table_height
        self.object_centered = object_centered
        # set camera extrinsics
        if random_position:
            if restrict_dir:
                theta = (0.25 + np.random.random() * 0.5) * np.pi * 2  # [0.5π, 1.5π]
                phi = (np.random.random() + 1) * np.pi / 6
            else:
                theta = np.random.random() * np.pi * 2
                phi = (np.random.random() + 1) * np.pi / 6
        if fixed_position:
            theta = theta
            phi = phi  ##
        if random_position and real_data:
            theta = np.random.random() * np.pi * 2
            phi = np.random.random() * np.pi * 2

        pos = np.array([dist * np.cos(phi) * np.cos(theta), \
                        dist * np.cos(phi) * np.sin(theta), \
                        dist * np.sin(phi)])
        forward = -pos / np.linalg.norm(pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.vstack([forward, left, up]).T
        mat44[:3, 3] = pos  # mat44 is cam2world
        if object_centered:
            mat44[0, 3] += env.object_position_offset[0]
            mat44[1, 3] += env.object_position_offset[1]
        if have_table:
            mat44[2, 3] += env.object_position_offset[2] + table_height

        self.mat44 = mat44

        self.camera.entity.set_pose(Pose(mat44))

        # compute camera-base frame (camera-center, world-up-z, camera-front-x)
        cb_up = np.array([0, 0, 1], dtype=np.float32)
        cb_left = np.cross(cb_up, forward)
        cb_left /= np.linalg.norm(cb_left)
        cb_forward = np.cross(cb_left, cb_up)
        cb_forward /= np.linalg.norm(cb_forward)
        base_mat44 = np.eye(4)
        base_mat44[:3, :3] = np.vstack([cb_forward, cb_left, cb_up]).T
        base_mat44[:3, 3] = pos  # cambase2world
        self.base_mat44 = base_mat44
        self.cam2cambase = np.linalg.inv(base_mat44) @ mat44  # cam2cambase
        self.cam2cambase = self.cam2cambase[:3, :3]

        # log parameters
        self.near = near
        self.far = far
        self.dist = dist
        self.theta = theta
        self.phi = phi
        self.pos = pos

        # self.camera_mount_actor = camera_mount_actor

    def change_pose(self, phi=np.pi / 5, theta=np.pi, dist=5.0, random_position=False, restrict_dir=False):
        # set camera extrinsics
        if random_position:
            if restrict_dir:
                theta = (0.25 + np.random.random() * 0.5) * np.pi * 2
                phi = (np.random.random() + 1) * np.pi / 6
            else:
                theta = np.random.random() * np.pi * 2
                phi = (np.random.random() + 1) * np.pi / 6

        pos = np.array([dist * np.cos(phi) * np.cos(theta), \
                        dist * np.cos(phi) * np.sin(theta), \
                        dist * np.sin(phi)])
        forward = -pos / np.linalg.norm(pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.vstack([forward, left, up]).T
        mat44[:3, 3] = pos  # mat44 is cam2world
        if self.object_centered:
            mat44[0, 3] += self.env.object_position_offset[0]
            mat44[1, 3] += self.env.object_position_offset[1]
        if self.have_table:
            mat44[2, 3] += self.env.object_position_offset[2] + self.table_height

        self.mat44 = mat44
        self.camera.entity.set_pose(Pose(mat44))

        # compute camera-base frame (camera-center, world-up-z, camera-front-x)
        cb_up = np.array([0, 0, 1], dtype=np.float32)
        cb_left = np.cross(cb_up, forward)
        cb_left /= np.linalg.norm(cb_left)
        cb_forward = np.cross(cb_left, cb_up)
        cb_forward /= np.linalg.norm(cb_forward)
        base_mat44 = np.eye(4)
        base_mat44[:3, :3] = np.vstack([cb_forward, cb_left, cb_up]).T
        base_mat44[:3, 3] = pos  # cambase2world
        self.base_mat44 = base_mat44
        self.cam2cambase = np.linalg.inv(base_mat44) @ mat44  # cam2cambase
        self.cam2cambase = self.cam2cambase[:3, :3]

        self.dist = dist
        self.theta = theta
        self.phi = phi
        self.pos = pos

    def get_observation(self):
        self.camera.take_picture()
        # RGBA
        rgba = self.camera.get_picture("Color")
        rgba = (rgba * 255).clip(0, 255).astype(np.float32) / 255
        white = np.ones((rgba.shape[0], rgba.shape[1], 3), dtype=np.float32)
        mask = np.tile(rgba[:, :, 3:4], [1, 1, 3])
        rgb = rgba[:, :, :3] * mask + white * (1 - mask)

        #depth
        position = self.camera.get_picture("Position")
        z = -position[..., 2]
        render_depth = position[..., 3]

        return rgb, render_depth

    def compute_camera_XYZA(self, render_depth):
        position = self.camera.get_picture("Position")
        camera_matrix = self.camera.get_intrinsic_matrix()[:3, :3]
        y, x = np.where(render_depth < 1)
        # z = self.near * self.far / (self.far + render_depth * (self.near - self.far))

        z = -position[..., 2]
        permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        points = (permutation @ np.dot(np.linalg.inv(camera_matrix), np.stack([x, y, np.ones_like(x)] * z[y, x], 0))).T
        return y, x, points

    def compute_points(self):
        position = self.camera.get_picture("Position")
        render_depth = position[..., 3]
        points_opengl = position[..., :3][render_depth < 1]
        model_matrix = self.camera.get_model_matrix()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        return points_opengl, points_world

    @staticmethod
    def compute_XYZA_matrix(id1, id2, pts, size1, size2):
        out = np.zeros((size1, size2, 4), dtype=np.float32)
        out[id1, id2, :3] = pts
        out[id1, id2, 3] = 1
        return out

    def get_normal_map(self):  #OK!!!
        self.camera.take_picture()
        nor = self.camera.get_picture('Normal')
        nor = nor[:, :, :3]
        # convert from PartNet-space (x-right, y-up, z-backward) to SAPIEN-space (x-front, y-left, z-up)
        new_nor = np.array(nor, dtype=np.float32)
        new_nor[:, :, 0] = -nor[:, :, 2]
        new_nor[:, :, 1] = -nor[:, :, 0]
        new_nor[:, :, 2] = nor[:, :, 1]

        return new_nor

    def get_movable_link_mask(self, link_ids):

        seg_labels = self.camera.get_picture("Segmentation")  # [H, W, 4]
        visual_seg = seg_labels[..., 0]
        link_seg = seg_labels[..., 1]
        link_mask = np.zeros((link_seg.shape[0], link_seg.shape[1])).astype(np.uint8)

        for idx, lid in enumerate(link_ids):
            cur_link_pixels = int(np.sum(link_seg == lid))
            if cur_link_pixels > 0:
                link_mask[link_seg == lid] = idx + 1
        return link_mask

    def get_limit_mask(self, x_l, x_r, y_l, y_r):
        img_size = (448, 448)
        x = np.arange(img_size[1])  # x 方向 [0, 447]
        y = np.arange(img_size[0])  # y 方向 [0, 447]
        xx, yy = np.meshgrid(x, y)
        limit_condition = (xx >= x_l) & (xx <= x_r) & (yy >= y_l) & (yy <= y_r)
        ys, xs = np.where(limit_condition > 0)
        limit_mask = np.zeros((448, 448)).astype(np.uint8)
        limit_mask[np.where(limit_condition > 0)] = 1
        return limit_mask

    def get_conbine_mask(self, limit_mask, condition_mask):

        conbine_mask = np.zeros((448, 448)).astype(np.uint8)
        # conbine_mask[np.where(condition > 0 and gt_movable_link_mask > 0)] = 1
        conbine_mask[np.where((limit_mask > 0) & (condition_mask > 0))] = 1
        return conbine_mask

    def get_target_seg_mask(self, keyword='handle'):
        # read part seg partid2renderids
        partid2renderids = dict()  # {leg1:[id1,id2], leg2:[id_x, id_y]}
        for k in self.env.scene.render_id_to_visual_name:
            print(k, self.env.scene.render_id_to_visual_name[k])
            if self.env.scene.render_id_to_visual_name[k].split('-')[0] == keyword:  # e.g. leg-1 / base_body-3
                part_id = int(self.env.scene.render_id_to_visual_name[k].split('-')[-1])
                if part_id not in partid2renderids:
                    partid2renderids[part_id] = []
                partid2renderids[part_id].append(k)
        # generate 0/1 target mask
        part_seg = self.camera.get_obj_segmentation()
        # print('part_seg: ', part_seg, part_seg.shape)
        target_mask = np.zeros((part_seg.shape[0], part_seg.shape[1])).astype(np.uint8)
        for partid in partid2renderids:
            cur_part_mask = np.isin(part_seg, partid2renderids[partid])
            cur_part_mask_pixels = int(np.sum(cur_part_mask))
            if cur_part_mask_pixels > 0:
                target_mask[cur_part_mask] = partid
        return target_mask

    def get_object_mask(self):
        rgba = self.camera.get_albedo_rgba()
        return rgba[:, :, 3] > 0.5

    # return camera parameters
    def get_metadata(self):
        return {
            'pose': self.camera.get_pose(),
            'near': self.camera.get_near(),
            'far': self.camera.get_far(),
            'width': self.camera.get_width(),
            'height': self.camera.get_height(),
            'fov': self.camera.fovy,
            'camera_matrix': self.camera.get_intrinsic_matrix(),
            'projection_matrix': self.camera.get_projection_matrix(),
            'model_matrix': self.camera.get_model_matrix(),
            'mat44': self.mat44,
            'cam2cambase': self.cam2cambase,
            'basemat44': self.base_mat44,
        }

    # return camera parameters
    def get_metadata_json(self):
        return {
            'dist': self.dist,
            'theta': self.theta,
            'phi': self.phi,
            # 'pose': self.camera.get_pose(),
            'p': self.camera.get_pose().p.tolist(),
            'q': self.camera.get_pose().q.tolist(),
            'rpy': self.camera.get_pose().rpy.tolist(),
            'near': self.camera.get_near(),
            'far': self.camera.get_far(),
            'width': self.camera.get_width(),
            'height': self.camera.get_height(),
            'fov': self.camera.fovy,
            'camera_matrix': self.camera.get_intrinsic_matrix().tolist(),
            'projection_matrix': self.camera.get_projection_matrix().tolist(),
            'model_matrix': self.camera.get_model_matrix().tolist(),
            'mat44': self.mat44.tolist(),
            'cam2cambase': self.cam2cambase.tolist(),
            'basemat44': self.base_mat44.tolist(),
        }

    def get_picture_items(self):
        self.camera.take_picture()
        view1 = self.camera.get_picture('SegmentationView1')
        print('view1:', view1.shape)
        np.savetxt("../expdata/Try/Kettle/view10.txt", view1[..., 0], fmt='%d')
        np.savetxt("../expdata/Try/Kettle/view11.txt", view1[..., 1], fmt='%d')
        np.savetxt("../expdata/Try/Kettle/view12.txt", view1[..., 2], fmt='%d')
        np.savetxt("../expdata/Try/Kettle/view13.txt", view1[..., 3], fmt='%d')
        view0 = self.camera.get_picture('SegmentationView0')
        np.savetxt("../expdata/Try/Kettle/view00.txt", view0[..., 0], fmt='%d')
        np.savetxt("../expdata/Try/Kettle/view01.txt", view0[..., 1], fmt='%d')
        np.savetxt("../expdata/Try/Kettle/view02.txt", view0[..., 2], fmt='%d')
        np.savetxt("../expdata/Try/Kettle/view03.txt", view0[..., 3], fmt='%d')

        print('view0:', view0.shape)
        return

    def get_visual_label(self, outdir, shape_id):
        out_path = os.path.join(outdir, str(shape_id))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.camera.take_picture()
        seg_labels = self.camera.get_picture("Segmentation")  # [H, W, 4]
        visual_seg = seg_labels[..., 0]
        # link_seg = seg_labels[..., 1]
        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
        visual_image = visual_seg.astype(np.uint8)  # mesh-level
        visual_pil = Image.fromarray(color_palette[visual_image])
        visual_pil.save(os.path.join(out_path, "visual_seg.png"))
        np.savetxt(os.path.join(out_path, "visual_seg.txt"), visual_seg, fmt='%d')
        return

    def get_visual_label_mask(self, target_visual_id=1):
        seg_labels = self.camera.get_picture("Segmentation")  # [H, W, 4]
        visual_seg = seg_labels[..., 0]
        return visual_seg
        # visual_mask = np.zeros((visual_seg.shape[0], visual_seg.shape[1])).astype(np.uint8)
