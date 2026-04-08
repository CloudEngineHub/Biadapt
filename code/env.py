"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import os
import sys
import sapien.core as sapien
from sapien.core import Pose, SceneConfig
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
import copy
import json
import math
import ipdb
from utils import process_angle_limit, get_random_number, printout
import time


class ContactError(Exception):
    pass


class SVDError(Exception):
    pass


class Env(object):

    def __init__(self,
                 flog=None,
                 show_gui=True,
                 render_rate=20,
                 timestep=1 / 500,
                 object_position_offset=[0, 0, 0],
                 succ_ratio=0.1,
                 set_ground=False,
                 static_friction=0.30,
                 dynamic_friction=0.30):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset

        # scene
        scene_config = sapien.physx.PhysxSceneConfig()
        scene_config.gravity = np.array([0, 0, -9.81])
        scene_config.enable_pcm = False
        sapien.physx.set_scene_config(scene_config)

        sapien.physx.set_default_material(static_friction=static_friction,
                                          dynamic_friction=dynamic_friction,
                                          restitution=0.0)

        body_config = sapien.physx.PhysxBodyConfig()
        body_config.sleep_threshold = 0.0
        body_config.solver_position_iterations = 20
        sapien.physx.set_body_config(body_config)

        self.scene = sapien.Scene([
            sapien.physx.PhysxCpuSystem(),
            sapien.render.RenderSystem(),
        ])

        self.scene.set_timestep(timestep)

        if set_ground:
            self.scene.add_ground(altitude=0.0, render=False)
            # ground_material = sapien.render.RenderMaterial()
            # ground_material.base_color = np.array([202, 164, 114, 256]) / 256
            # ground_material.specular = 0.5
            # self.scene.add_ground(0, render_material=ground_material)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        self.scene.add_point_light(
            [1 + object_position_offset[0], 2 + object_position_offset[1], 2 + object_position_offset[2]], [1, 1, 1])
        self.scene.add_point_light(
            [1 + object_position_offset[0], -2 + object_position_offset[1], 2 + object_position_offset[2]], [1, 1, 1])
        self.scene.add_point_light(
            [-1 + object_position_offset[0], 0 + object_position_offset[1], 1 + object_position_offset[2]], [1, 1, 1])

        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(64)
        sapien.render.set_ray_tracing_denoiser("optix")
        sapien.render.RenderLightComponent.enable_shadow = True
        sapien.render.RenderLightComponent.set_shadow_map_size = 8192

        # GUI

        if show_gui:
            print('Viewer created')
            self.viewer = self.scene.create_viewer()
            self.viewer.set_camera_xyz(x=-3.0 + object_position_offset[0], y=1.0, z=3.0)
            self.viewer.set_camera_rpy(y=-0.4, p=-0.8, r=0)
            # self.viewer.window = False
            # self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False
        self.check_contact_strict = False
        self.contact_error = False

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.viewer.set_camera_xyz(x, y, z)
        self.viewer.set_camera_rpy(r=0, y=yaw, p=pitch)
        self.viewer.render()

    def load_object(self,
                    urdf,
                    material,
                    state='closed',
                    target_part_id=-1,
                    scale=1.0,
                    density=1.0,
                    stiffness=0,
                    damping=10,
                    lieDown=False,
                    given_joint_angles=None,
                    given_pose=None,
                    srdf=None,
                    target_joint_name=None):

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.scale = scale
        loader.set_material(static_friction=material[0], dynamic_friction=material[1], restitution=material[2])
        loader.set_density(density)

        self.object = loader.load(urdf, srdf)

        if given_pose:
            pose = given_pose
        else:
            if not lieDown:
                pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
            else:
                # pose = Pose([self.object_position_offset, 0, 0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0])
                pose = Pose([self.object_position_offset, 0, 0], [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0])

        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_entity().get_per_scene_id() for l in self.object.get_links()]
        self.all_link_names = [l.get_entity().get_name() for l in self.object.get_links()]

        self.all_joint_types = [j.type for j in self.object.get_joints()]
        self.movable_link_ids = []
        self.holdable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                self.movable_link_ids.append(j.get_child_link().get_entity().get_per_scene_id())
                self.holdable_link_ids.append(j.get_parent_link().get_entity().get_per_scene_id())

        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=stiffness, damping=damping)

        joint_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        target_part_joint_idx = -1
        for j in self.object.get_joints():
            if j.get_dof() == 1:

                if j.get_child_link().get_entity().get_per_scene_id() == target_part_id:
                    target_part_joint_idx = len(joint_angles)
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])

                self.joint_angles_upper.append(float(r))
                if target_joint_name is not None:
                    if j.get_name() != target_joint_name:
                        joint_angles.append(float(l))  #not target joint ,l_angle
                        continue

                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'middle':
                    joint_angles.append(float((l + r) / 2))
                elif state == 'random-middle':
                    joint_angles.append(float(get_random_number(l, r)))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        joint_angles.append(float(l))
                elif state == 'random-left':  #TODO
                    joint_angles.append(l + 1 / 4 * (r - l))
                elif state == 'low-left':  #TODO
                    joint_angles.append(l + 1 / 8 * (r - l))
                elif state == 'random-right':
                    joint_angles.append(r - 1 / 4 * (r - l))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)

        if given_joint_angles:
            joint_angles = given_joint_angles

        self.object.set_qpos(joint_angles)
        if target_part_id >= 0:
            return joint_angles, target_part_joint_idx

        return joint_angles

    def create_table(self,
                     pose: sapien.Pose,
                     size_x,
                     size_y,
                     height,
                     thickness=0.1,
                     color=(0.8, 0.6, 0.4),
                     friction=1.0,
                     restitution=0.01,
                     density=1000,
                     name='table') -> sapien.Entity:
        """Create a table (a collection of collision and visual shapes)."""
        builder = self.scene.create_actor_builder()

        # Tabletop
        tabletop_pose = sapien.Pose([0., 0., -thickness / 2 + height])  # Make the top surface's z equal to 0
        tabletop_half_size = [size_x / 2, size_y / 2, thickness / 2]
        tabletop_collision_size = [size_x / 10, size_y / 10, thickness / 10]
        table_material = self.scene.create_physical_material(friction, friction, restitution)
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=color)
        builder.add_box_collision(pose=tabletop_pose,
                                  half_size=tabletop_half_size,
                                  material=table_material,
                                  density=density)

        # Table legs (x4)
        for i in [-1, 1]:
            for j in [-1, 1]:
                x = i * 0.9 * (size_x - thickness) / 2
                y = j * 0.9 * (size_y - thickness) / 2
                table_leg_pose = sapien.Pose([x, y, -thickness / 2 + height / 2])
                table_leg_half_size = [thickness / 2, thickness / 2, (height - thickness) / 2]
                table_leg_collision_size = [thickness / 10, thickness / 10, (height - thickness) / 10]
                builder.add_box_collision(pose=table_leg_pose,
                                          half_size=table_leg_half_size,
                                          density=density,
                                          material=table_material)
                builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

        table = builder.build_static(name=name)
        table.set_pose(pose)
        self.have_table = True
        return table

    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def get_target_part_axes(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_entity().get_per_scene_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[1, 0]), float(mat[2, 0]), float(-mat[0, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')
        return joint_axes

    def get_target_part_axes_new(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_entity().get_per_scene_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[0, 0]), float(-mat[1, 0]), float(mat[2, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')

        return joint_axes

    def set_target_object_part_actor_ids(self, actor_id1, actor_id2):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor1 ID : %d \n' % actor_id1)
            self.flog.write('Set Target Object Part Actor2 ID : %d \n' % actor_id2)

        self.target_object_part_actor_id_1 = actor_id1  # not movable
        self.target_object_part_actor_id_2 = actor_id2  # not movable

        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id1]) - set([actor_id2]))

        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                child_link = j.get_child_link()
                parent_link = j.get_parent_link()
                if child_link.get_entity().get_per_scene_id() == actor_id1 and parent_link.get_entity(
                ).get_per_scene_id() == actor_id2:
                    self.target_object_part_actor_link_1 = child_link
                    self.target_object_part_actor_link_2 = parent_link
                elif child_link.get_entity().get_per_scene_id() == actor_id2 and parent_link.get_entity(
                ).get_per_scene_id() == actor_id1:
                    self.target_object_part_actor_link_1 = parent_link
                    self.target_object_part_actor_link_2 = child_link

        # monitor the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                child_link = j.get_child_link()
                parent_link = j.get_parent_link()
                if (j.get_child_link().get_entity().get_per_scene_id() == actor_id1 and
                        parent_link.get_entity().get_per_scene_id()
                        == actor_id2) or (j.get_child_link().get_entity().get_per_scene_id() == actor_id2 and
                                          parent_link.get_entity().get_per_scene_id() == actor_id1):
                    self.target_object_part_joint_id = idx
                    self.target_object_part_joint = j
                    # j.set_drive_property(damping=5)
                    self.target_object_part_joint_type = j.type
                    if self.flog is not None:
                        self.flog.write('Set Target Joint ID : %d \n' % idx)
                        self.flog.write('Set Target Joint Type : %s \n' % str(j.type))
                idx += 1

    def set_target_object_part_actor(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor1 ID : %d \n' % actor_id)
            self.flog.write('Set Target Object Part Actor2 ID : %d \n' % actor_id)

        self.target_object_part_actor_id_1 = actor_id  # not movable
        self.target_object_part_actor_id_2 = actor_id  # not movable

        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        for l in self.object.get_links():
            if l.entity.get_per_scene_id() == actor_id:
                self.target_object_part_actor_link_1 = l
                self.target_object_part_actor_link_2 = l
                break
        return

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_object_root_pose(self):
        return self.object.get_root_pose()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        # ipdb.set_trace()
        return float(qpos[self.target_object_part_joint_id])

    def get_target_part1_pose(self):
        return self.target_object_part_actor_link_1.get_pose()

    def get_target_part2_pose(self):
        return self.target_object_part_actor_link_2.get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.contact_error = False

    def dual_start_checking_contact(self, robot1_hand_actor_id, robot1_gripper_actor_ids, robot2_hand_actor_id,
                                    robot2_gripper_actor_ids, strict):

        self.check_dual_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot1_hand_actor_id = robot1_hand_actor_id
        self.robot2_hand_actor_id = robot2_hand_actor_id
        self.robot1_gripper_actor_ids = robot1_gripper_actor_ids
        self.robot2_gripper_actor_ids = robot2_gripper_actor_ids
        self.contact_error = False

    def end_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = False
        self.check_contact_strict = strict
        self.first_timestep_check_contact = False
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def dual_end_checking_contact(self, robot1_hand_actor_id, robot1_gripper_actor_ids, robot2_hand_actor_id,
                                  robot2_gripper_actor_ids, strict):
        self.check_contact = False
        self.check_contact_strict = strict
        self.first_timestep_check_contact = False
        self.robot1_hand_actor_id = robot1_hand_actor_id
        self.robot2_hand_actor_id = robot2_hand_actor_id
        self.robot1_gripper_actor_ids = robot1_gripper_actor_ids
        self.robot2_gripper_actor_ids = robot2_gripper_actor_ids

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.scene.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):

        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.viewer.render()

    def step(self):
        self.current_step += 1
        self.scene.step()
        # print('Check Contact:',self.check_contact)
        if self.check_contact:
            if not self.dual_check_contact_is_valid():
                raise ContactError()

    # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False
        valid = False
        for c in self.contacts:
            actor1 = c.bodies[0]
            actor2 = c.bodies[1]
            # print(actor1.get_name(),actor2.get_name())
            aid1 = actor1.get_entity().get_per_scene_id()
            aid2 = actor2.get_entity().get_per_scene_id()
            has_impulse = False

            for p in c.points:
                force = abs(p.impulse @ p.impulse)
                # print('Force: ', force)
                if force > 1e-5:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id_2) or \
                (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id_2):
                    contact, valid = True, True
                    print('Valid Contact ')
                    print(actor1.get_name(), actor2.get_name())
                elif (aid1 in self.robot_gripper_actor_ids and aid2 in self.robot_gripper_actor_ids):
                    contact, valid = True, False
                    print("No Init Contact")
                    return False

        self.first_timestep_check_contact = False
        if contact and valid:
            self.check_contact = False

        return True

    def half_check_contact(self):
        print('check!')
        self.contacts = self.scene.get_contacts()
        contact1 = False
        valid1 = False
        contact2 = False
        valid2 = False
        for c in self.contacts:
            actor1 = c.bodies[0]
            actor2 = c.bodies[1]
            # print(actor1.get_name(),actor2.get_name())
            aid1 = actor1.get_entity().get_per_scene_id()
            aid2 = actor2.get_entity().get_per_scene_id()
            has_impulse = False

            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot1_gripper_actor_ids and aid2 == self.target_object_part_actor_id_1) or \
                   (aid2 in self.robot1_gripper_actor_ids and aid1 == self.target_object_part_actor_id_1):
                    contact1, valid1 = True, True
                    print('Valid Contact1 ')
                elif (aid1 in self.robot2_gripper_actor_ids and aid2 == self.target_object_part_actor_id_2) or \
                   (aid2 in self.robot2_gripper_actor_ids and aid1 == self.target_object_part_actor_id_2):
                    contact2, valid2 = True, True
                    # print('Valid Contact2 ')

        # self.first_timestep_check_contact = False
        # if (contact1 and valid1) and (contact2 and valid2):
        #     self.check_contact = False
        return (contact1 and valid1) or (contact2 and valid2)

    def dual_check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False
        valid = False
        for c in self.contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            has_impulse = False
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot1_gripper_actor_ids and aid2 == self.target_object_part_actor_id_1) or \
                        (aid2 in self.robot1_gripper_actor_ids and aid1 == self.target_object_part_actor_id_1):
                    contact, valid = True, True
                if (aid1 in self.robot2_gripper_actor_ids and aid2 == self.target_object_part_actor_id_2) or \
                        (aid2 in self.robot2_gripper_actor_ids and aid1 == self.target_object_part_actor_id_2):
                    contact, valid = True, True

                if (aid1 in self.robot1_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                        (aid2 in self.robot1_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                    if self.check_contact_strict:
                        # print("## Gripper1-Object collision", c.actor1.name, c.actor2.name, aid1, aid2)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                if (aid1 in self.robot2_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                        (aid2 in self.robot2_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                    if self.check_contact_strict:
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True

                if aid1 == self.robot1_hand_actor_id or aid2 == self.robot1_hand_actor_id:
                    if self.check_contact_strict:
                        # print("## Hand1 collision", c.actor1.name, c.actor2.name, aid1, aid2)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                if aid1 == self.robot2_hand_actor_id or aid2 == self.robot2_hand_actor_id:
                    if self.check_contact_strict:
                        # print("## Hand2 collision", c.actor1.name, c.actor2.name, aid1, aid2)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True

                # starting pose should have no collision at all
                if (aid1 in self.robot1_gripper_actor_ids or aid1 == self.robot1_hand_actor_id or
                        aid2 in self.robot1_gripper_actor_ids or
                        aid2 == self.robot1_hand_actor_id) and self.first_timestep_check_contact:
                    # print("## Start Hand1 collision", c.actor1.name, c.actor2.name, aid1, aid2)
                    self.contact_error = True
                    return False
                if (aid1 in self.robot2_gripper_actor_ids or aid1 == self.robot2_hand_actor_id or
                        aid2 in self.robot2_gripper_actor_ids or
                        aid2 == self.robot2_hand_actor_id) and self.first_timestep_check_contact:
                    # print("## Start Hand2 collision", c.actor1.name, c.actor2.name, aid1, aid2)
                    self.contact_error = True
                    return False

        self.first_timestep_check_contact = False
        if contact and valid:
            self.check_contact = False
        return True

    def close_render(self):
        if not self.viewer.closed:
            self.viewer.close()
        self.viewer = None

    def wait_to_start(self):
        print('press q to start\n')
        self.scene.update_render()  # Update the world to the renderer
        if self.show_gui:
            self.viewer.render()

    def close(self):
        if self.show_gui:
            self.viewer.close()
        self.scene = None

    def visual_test(self):
        if self.show_gui:
            while not self.viewer.closed:  # Press key q to quit
                self.scene.step()  # Simulate the world
                self.scene.update_render()  # Update the world to the renderer
                self.viewer.render()
