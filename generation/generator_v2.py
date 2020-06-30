import pickle
import random
import time
import os
import csv
import copy
import xml.etree.ElementTree as ET

import cv2
import h5py
import pyro
import pyro.distributions as dist
import torch
import numpy as np
from tqdm import tqdm
import transforms3d as tf3d
from mujoco_py import functions

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder

from SyntheticArticulatedData.generation.mujocoCabinetParts import build_cabinet, sample_cabinet
from SyntheticArticulatedData.generation.mujocoDrawerParts import build_drawer, sample_drawers
from SyntheticArticulatedData.generation.mujocoMicrowaveParts import build_microwave, sample_microwave
from SyntheticArticulatedData.generation.mujocoToasterOvenParts import build_toaster, sample_toaster
from SyntheticArticulatedData.generation.mujocoDoubleCabinetParts import build_cabinet2, sample_cabinet2, \
    set_two_door_control
from SyntheticArticulatedData.generation.mujocoRefrigeratorParts import build_refrigerator, sample_refrigerator
from SyntheticArticulatedData.generation.utils import *
import SyntheticArticulatedData.generation.calibrations as calibrations


def white_bg(img):
    mask = 1 - (img > 0)
    img_cp = copy.deepcopy(img)
    img_cp[mask.all(axis=2)] = [255, 255, 255]
    return img_cp


def buffer_to_real(z, zfar, znear):
    return 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * z - 1))


def vertical_flip(img):
    return np.flip(img, axis=0)


def should_use_image_hacky(img, bnds=2):
    mask = img > 0
    img_w_bnds = cv2.rectangle(np.zeros((img.shape[0], img.shape[1]), np.uint8), (0, 0),
                               (img.shape[1], img.shape[0]), 255, thickness=bnds) > 0
    if np.logical_and(mask, img_w_bnds).sum() > 20:
        return False
    else:
        return True


def should_use_image(img, bigger_image):
    n_obj = (img > 0).sum()
    n_obj_big = (bigger_image > 0).sum()
    if n_obj < 50 or (n_obj / n_obj_big) < 0.40:  # Fraction of pixels within smaller image is small
        return False
    else:
        return True


class SceneGenerator():
    def __init__(self, root_dir='bull/test_cabinets/solo', masked=False, debug_flag=False):
        '''
        Class for generating simulated articulated object dataset.
        params:
            - root_dir: save in this directory
            - start_idx: index of first image saved - useful in threading context
            - depth_data: np array of depth images
            - masked: should the background of depth images be 0s or 1s?
        '''
        self.scenes = []
        self.savedir = root_dir
        self.masked = masked
        self.img_idx = 0
        self.depth_data = []
        self.debugging = debug_flag
        print(root_dir)

    def write_urdf(self, filename, xml):
        with open(filename, "w") as text_file:
            text_file.write(xml)

    def sample_obj(self, obj_type, mean_flag, left_only, cute_flag=False):
        if obj_type == 'microwave':
            l, w, h, t, left, mass = sample_microwave(mean_flag)
            if mean_flag:
                obj = build_microwave(l, w, h, t, left,
                                      set_pose=[1.0, 0.0, -0.15],
                                      set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_microwave(l, w, h, t, left,
                                      set_pose=[1.0, 0.0, -0.15],
                                      set_rot=base_quat)
            else:
                obj = build_microwave(l, w, h, t, left)

        elif obj_type == 'drawer':
            l, w, h, t, left, mass = sample_drawers(mean_flag)
            if mean_flag:
                obj = build_drawer(l, w, h, t, left,
                                   set_pose=[1.5, 0.0, -0.4],
                                   set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_drawer(l, w, h, t, left,
                                   set_pose=[1.2, 0.0, -0.15],
                                   set_rot=base_quat)
            else:
                obj = build_drawer(l, w, h, t, left)

        elif obj_type == 'toaster':
            l, w, h, t, left, mass = sample_toaster(mean_flag)
            if mean_flag:
                obj = build_toaster(l, w, h, t, left,
                                    set_pose=[1.5, 0.0, -0.3],
                                    set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_toaster(l, w, h, t, left,
                                    set_pose=[1.0, 0.0, -0.15],
                                    set_rot=base_quat)
            else:
                obj = build_toaster(l, w, h, t, left)

        elif obj_type == 'cabinet':
            l, w, h, t, left, mass = sample_cabinet(mean_flag)
            if mean_flag:
                if left_only:
                    left = True
                else:
                    left = False
                obj = build_cabinet(l, w, h, t, left,
                                    set_pose=[1.5, 0.0, -0.3],
                                    set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_cabinet(l, w, h, t, left,
                                    set_pose=[1.5, 0.0, -0.15],
                                    set_rot=base_quat)
            else:
                left = np.random.choice([True, False])
                obj = build_cabinet(l, w, h, t, left)

        elif obj_type == 'cabinet2':
            l, w, h, t, left, mass = sample_cabinet2(mean_flag)
            if mean_flag:
                obj = build_cabinet2(l, w, h, t, left,
                                     set_pose=[1.5, 0.0, -0.3],
                                     set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_cabinet2(l, w, h, t, left,
                                     set_pose=[1.5, 0.0, -0.15],
                                     set_rot=base_quat)
            else:
                obj = build_cabinet2(l, w, h, t, left)

        elif obj_type == 'refrigerator':
            l, w, h, t, left, mass = sample_refrigerator(mean_flag)
            if mean_flag:

                obj = build_refrigerator(l, w, h, t, left,
                                         set_pose=[1.5, 0.0, -0.3],
                                         set_rot=[0.0, 0.0, 0.0, 1.0])
            elif cute_flag:
                base_xyz, base_angle = sample_pose()
                base_quat = angle_to_quat(base_angle)
                obj = build_refrigerator(l, w, h, t, left,
                                         set_pose=[2.5, 0.0, -0.75],
                                         set_rot=base_quat)

            else:
                obj = build_refrigerator(l, w, h, t, left)

        else:
            raise Exception('uh oh, object not implemented!')
        return obj

    def sample_object_pose(self, obj, objtype):
        if objtype == 'drawer':
            base_xyz, base_angle_x, base_angle_y, base_angle_z = sample_pose_drawer_2()
        else:
            base_xyz, base_angle_x, base_angle_y, base_angle_z = sample_pose_2()

        base_quat = tf3d.euler.euler2quat(base_angle_x, base_angle_y, base_angle_z, axes='sxyz')
        obj.pose = base_xyz
        obj.rotation = base_quat

        base_name = 'cabinet_bottom'
        if objtype == 'refrigerator':
            base_name = 'fridgeinet_bottom'

        root = ET.fromstring(obj.xml)
        for body in root.find('worldbody').findall('body'):
            if body.attrib['name'] == base_name:
                base = body
        base.set('pos', '{} {} {}'.format(base_xyz[0], base_xyz[1], base_xyz[2]))
        base.set('quat', '{} {} {} {}'.format(base_quat[0], base_quat[1], base_quat[2], base_quat[3]))  # wxyz
        obj.xml = ET.tostring(root, encoding="unicode")
        return obj

    def generate_scenes(self, N, objtype, n_uni_obj=None, write_csv=True, save_imgs=True, mean_flag=False, left_only=False,
                        cute_flag=False):
        if n_uni_obj is None or n_uni_obj == 0:
            n_uni_obj = N
        obj_instances = [self.sample_obj(objtype, mean_flag, left_only, cute_flag=cute_flag) for i in range(n_uni_obj)]

        fname = os.path.join(self.savedir, 'params.csv')
        h5fname = os.path.join(self.savedir, 'complete_data.hdf5')
        self.img_idx = 0
        i = 0
        with h5py.File(h5fname, 'a') as h5File:
            pbar = tqdm(total=N)
            while i < N:
                # obj = self.sample_obj(objtype, mean_flag, left_only, cute_flag=cute_flag)
                obj = random.choice(obj_instances)
                obj = self.sample_object_pose(obj, objtype)
                xml = obj.xml
                fname = os.path.join(self.savedir, 'scene' + str(i).zfill(6) + '.xml')
                grp = h5File.create_group("obj_" + str(i).zfill(6))
                self.write_urdf(fname, xml)
                res = self.take_images(fname, obj, grp, use_force=False)
                if not res:
                    del h5File["obj_" + str(i).zfill(6)]
                else:
                    i += 1
                    pbar.update(1)
                    self.scenes.append(fname)
                    grp.create_dataset('mujoco_scene_xml', shape=(1,), dtype="S10", data=np.string_(xml))
        return

    def take_images(self, filename, obj, h5group, use_force=False):
        model = load_model_from_path(filename)
        sim = MjSim(model)
        modder = TextureModder(sim)
        # viewer=MjViewer(sim) # this fucking line has caused me (Ben) so much pain.

        embedding = np.append(obj.type, obj.geom.reshape(-1))
        if obj.type == 4 or obj.type == 5:
            # MULTI CABINET: get double the params.
            axis1, door1 = transform_param(obj.params[0][0], obj.params[0][1], obj)
            axis2, door2 = transform_param(obj.params[1][0], obj.params[1][1], obj)
            axes = np.append(axis1, axis2)
            doors = np.append(door1, door2)
            params = np.append(axes, doors)
            set_two_door_control(sim, 'cabinet2' if obj.type == 4 else 'refrigerator')
            n_qpos_variables = 2

        else:
            handle_name = 'handle_link'
            joint_body_name = 'cabinet_left_hinge'
            n_qpos_variables = 1
            if obj.type == 1:
                sim.data.ctrl[0] = 0.05
                handle_name = 'knob'
                joint_body_name = 'drawer'
            elif obj.geom[3] == 1:
                sim.data.ctrl[0] = -0.2
            else:
                sim.data.ctrl[0] = 0.2

            params = get_cam_relative_params2(obj)  # if 1DoF, params is length 10. If 2DoF, params is length 20.

        embedding_and_params = np.concatenate((embedding, params, obj.pose, obj.rotation))
        # object_reference_frame_in_world = np.concatenate((obj.pose, obj.rotation))

        # print('nqpos', n_qpos_variables)
        # print(self.img_idx, obj.pose)
        # print(embedding_and_params.shape)
        t = 0

        #########################
        IMG_WIDTH = calibrations.sim_width
        IMG_HEIGHT = calibrations.sim_height
        #########################

        force = np.array([0., 0., 0.])
        if use_force:
            # Generating Data by applying random Cartesian forces
            sim.data.ctrl[0] = 0.
            force = np.array([-1., 0., 0.])
            torque = np.array([0., 0., 0.])
            pt = sim.data.get_body_xpos(handle_name)
            bodyid = sim.model.body_name2id(handle_name)

        q_vals = []
        qdot_vals = []
        qddot_vals = []
        torque_vals = []
        applied_forces = []
        moving_frame_xpos_world = []
        moving_frame_xpos_ref_frame = []
        depth_imgs = torch.Tensor()

        img_counter = 0

        while t < 4000:
            if use_force:
                sim.data.qfrc_applied.fill(0.)  # Have to clear previous data
                functions.mj_applyFT(model, sim.data, force, torque, pt, bodyid, sim.data.qfrc_applied)
            sim.forward()
            sim.step()

            """ Recording data for linear regression at a different frequency than images """
            # if t % 10 == 0:
            if t % 250 == 0:
                img, depth = sim.render(IMG_WIDTH, IMG_HEIGHT, camera_name='external_camera_0', depth=True)
                depth = vertical_flip(depth)
                real_depth = buffer_to_real(depth, 12.0, 0.1)
                norm_depth = real_depth / 12.0

                # Checking if sampled object is within the image frame or not
                bigger_img = sim.render(int(1.5*IMG_WIDTH), int(1.5*IMG_HEIGHT),
                                        camera_name='external_camera_0', depth=False)
                if not should_use_image(img, bigger_img):
                    self.img_idx -= img_counter
                    return False

                if self.masked:
                    # remove background
                    mask = norm_depth > 0.99
                    norm_depth = (1 - mask) * norm_depth

                if self.debugging:
                    # save image to disk for visualization
                    # img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))

                    img = vertical_flip(img)
                    img = white_bg(img)
                    integer_depth = norm_depth * 255

                    imgfname = os.path.join(self.savedir, 'img' + str(self.img_idx).zfill(6) + '.png')
                    depth_imgfname = os.path.join(self.savedir, 'depth_img' + str(self.img_idx).zfill(6) + '.png')
                    cv2.imwrite(imgfname, img)
                    cv2.imwrite(depth_imgfname, integer_depth)

                # if IMG_WIDTH != 192 or IMG_HEIGHT != 108:
                #     depth = cv2.resize(norm_depth, (192,108))

                    depthfname = os.path.join(self.savedir, 'depth' + str(self.img_idx).zfill(6) + '.pt')
                    torch.save(torch.tensor(norm_depth.copy()), depthfname)

                depth_imgs = torch.cat((depth_imgs, torch.tensor(norm_depth.copy()).float().unsqueeze_(dim=0)))

                q_vals.append(copy.copy(sim.data.qpos[:n_qpos_variables]))
                qdot_vals.append(copy.copy(sim.data.qvel[:n_qpos_variables]))
                qddot_vals.append(copy.copy(sim.data.qacc[:n_qpos_variables]))
                torque_vals.append(copy.copy(sim.data.qfrc_applied[:n_qpos_variables]))
                applied_forces.append(copy.copy(force))
                x_pos = np.append(sim.data.get_body_xpos(handle_name), sim.data.get_body_xquat(handle_name))
                moving_frame_xpos_world.append(copy.copy(x_pos))  # quat comes in wxyz form
                joint_frame_in_world = np.append(sim.data.get_body_xpos(joint_body_name), obj.rotation)
                # moving_frame_xpos_ref_frame.append(copy.copy(
                #     change_frames(frame_B_wrt_A=joint_frame_in_world, pose_wrt_A=x_pos)))

                self.img_idx += 1
                img_counter += 1

            t += 1

        h5group.create_dataset('embedding_and_params', data=embedding_and_params)
        h5group.create_dataset('joint_frame_in_world', data=joint_frame_in_world)
        h5group.create_dataset('moving_frame_in_world', data=np.array(moving_frame_xpos_world))
        # h5group.create_dataset('moving_frame_in_ref_frame', data=np.array(moving_frame_xpos_ref_frame))
        h5group.create_dataset('depth_imgs', data=depth_imgs)

        h5group.create_dataset('q', data=np.array(q_vals))
        h5group.create_dataset('qdot', data=np.array(qdot_vals))
        h5group.create_dataset('qddot', data=np.array(qddot_vals))
        h5group.create_dataset('torques', data=np.array(torque_vals))
        h5group.create_dataset('forces', data=np.array(applied_forces))

        return True


# shapes and stuff
# if 1DoF, params is length 10. If 2DoF, params is length 20.
# embedding is length 5: type, l, w, h, left
# pose is length 3
# rotation is length 4
# finally, q is length 1 or 2.
# thus, for generating shape data:
# 1DoF: q is position 21
# 2DoF: q is position 31

# Object IDs
# 0 - microwave
# 1 - drawer
# 2 - cabinet
# 3 - toaster
# 4 - double cabinet
# 5 - refrigerator
