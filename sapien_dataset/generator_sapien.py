import copy
import os
import xml.etree.ElementTree as ET

import cv2
import h5py
import numpy as np
import torch
import transforms3d as tf3d
from SyntheticArticulatedData.generation import calibrations
from SyntheticArticulatedData.generation.utils import sample_pose_sapien
from mujoco_py import load_model_from_path, MjSim
from mujoco_py.modder import TextureModder
from tqdm import tqdm


# Load points for the object. Subsample if needed
# Sample pose of the object. Transform point cloud to that pose
# Use projection matrix to project to camera plane. Check the distance from the image center

def white_bg(img):
    mask = 1 - (img > 0)
    img_cp = copy.deepcopy(img)
    img_cp[mask.all(axis=2)] = [255, 255, 255]
    return img_cp


def buffer_to_real(z, zfar, znear):
    return 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * z - 1))


def vertical_flip(img):
    return np.flip(img, axis=0)


def should_use_image_hack(img, bigger_image):
    n_obj = (img > 0).sum()
    n_obj_big = (bigger_image > 0).sum()
    if n_obj < 50 or (n_obj / n_obj_big) < 0.25:  # Fraction of pixels within smaller image is small
        # return False
        return True
    else:
        return True


class SceneGeneratorSapien():
    def __init__(self, obj_idxs, xml_dir, root_dir='./data/', masked=False, debug_flag=False):
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

        self.obj_idxs = obj_idxs
        self.xml_dir = xml_dir
        print(root_dir)

    def save_scene_file(self, xml_tree_object, xml_path, fname):
        mesh_file_path = xml_path + '/textured_objs/'
        root = xml_tree_object.getroot()
        root.find('compiler').set('meshdir', mesh_file_path)
        xml_tree_object.write(fname, xml_declaration=True)

    def generate_scenes(self, N, obj_type):
        h5fname = os.path.join(self.savedir, 'complete_data.hdf5')
        self.img_idx = 0
        i = 0
        with h5py.File(h5fname, 'a') as h5File:
            pbar = tqdm(total=N)
            str_type = h5py.string_dtype()
            while i < N:
                o_id = self.obj_idxs[int(i % len(self.obj_idxs))]
                xml_path = os.path.join(self.xml_dir, o_id)
                obj_tree = copy.copy(ET.parse(xml_path + '/mobility_mujoco.xml'))
                root = obj_tree.getroot()

                # Sample object pose
                base_xyz, base_angle_x, base_angle_y, base_angle_z = sample_pose_sapien()
                base_quat = tf3d.euler.euler2quat(base_angle_x, base_angle_y, base_angle_z, axes='sxyz')  # wxyz
                # print("Sampled base pose:{}  {}".format(base_xyz, base_quat))
                # Update object pose
                for body in root.find('worldbody').findall('body'):
                    if body.attrib['name'] == 'base':
                        base = body
                base.set('pos', '{} {} {}'.format(base_xyz[0], base_xyz[1], base_xyz[2]))
                base.set('quat', '{} {} {} {}'.format(base_quat[0], base_quat[1], base_quat[2], base_quat[3]))  # wxyz
                fname = os.path.join(self.savedir, 'scene' + str(i).zfill(6) + '.xml')
                self.save_scene_file(obj_tree, xml_path, fname)

                # take images
                grp = h5File.create_group("obj_" + str(i).zfill(6))
                res = self.take_images(fname, o_id, grp, use_force=False)
                if not res:
                    del h5File["obj_" + str(i).zfill(6)]
                else:
                    i += 1
                    pbar.update(1)
                    self.scenes.append(fname)
                    ds = grp.create_dataset('mujoco_scene_xml', shape=(1,), dtype=str_type)
                    ds[:] = ET.tostring(root)
        return

    def take_images(self, filename, obj_idx, h5group, use_force=False):
        model = load_model_from_path(filename)
        sim = MjSim(model)
        modder = TextureModder(sim)

        # embedding = np.append(obj.type, obj.geom.reshape(-1))
        if obj_idx in ['7119', '7167', '7263', '7310']:
            handle_name = 'handle'
        elif obj_idx in ['7265', '7349', '7128']:
            handle_name = 'glass'
        else:
            handle_name = 'door'

        act_idx = 0
        if obj_idx in ['7349', '7366']:
            act_idx = 1

        n_qpos_variables = 1
        sim.data.ctrl[act_idx] = 0.1  # + 0.5 * np.random.randn()   # Random variation

        # obj_type = 0
        # embedding = np.append(obj_type, obj.geom.reshape(-1))
        # params = get_cam_relative_params2(obj)  # if 1DoF, params is length 10. If 2DoF, params is length 20.

        # embedding_and_params = np.concatenate((embedding, params, obj.pose, obj.rotation))
        # object_reference_frame_in_world = np.concatenate((obj.pose, obj.rotation))

        #########################
        IMG_WIDTH = calibrations.sim_width
        IMG_HEIGHT = calibrations.sim_height
        #########################

        q_vals = []
        qdot_vals = []
        qddot_vals = []
        torque_vals = []
        applied_forces = []
        moving_frame_xpos_world = []
        moving_frame_xpos_ref_frame = []
        depth_imgs = torch.Tensor()

        t = 0
        img_counter = 0

        while t < 4000:
            sim.forward()
            sim.step()

            if t % 250 == 0:
                img, depth = sim.render(IMG_WIDTH, IMG_HEIGHT, camera_name='external_camera_0', depth=True)
                depth = vertical_flip(depth)
                real_depth = buffer_to_real(depth, 12.0, 0.1)
                norm_depth = real_depth / 12.0

                # Checking if sampled object is within the image frame or not
                bigger_img = sim.render(int(1.5 * IMG_WIDTH), int(1.5 * IMG_HEIGHT),
                                        camera_name='external_camera_0', depth=False)
                if not should_use_image_hack(img, bigger_img):
                    self.img_idx -= img_counter
                    return False

                if self.masked:
                    # remove background
                    mask = norm_depth > 0.99
                    norm_depth = (1 - mask) * norm_depth

                if self.debugging:
                    img = vertical_flip(img)
                    img = white_bg(img)
                    integer_depth = norm_depth * 255

                    imgfname = os.path.join(self.savedir, 'img' + str(self.img_idx).zfill(6) + '.png')
                    depth_imgfname = os.path.join(self.savedir, 'depth_img' + str(self.img_idx).zfill(6) + '.png')
                    cv2.imwrite(imgfname, img)
                    cv2.imwrite(depth_imgfname, integer_depth)
                    depthfname = os.path.join(self.savedir, 'depth' + str(self.img_idx).zfill(6) + '.pt')
                    torch.save(torch.tensor(norm_depth.copy()), depthfname)

                depth_imgs = torch.cat((depth_imgs, torch.tensor(norm_depth.copy()).float().unsqueeze_(dim=0)))

                q_vals.append(copy.copy(sim.data.qpos[:n_qpos_variables]))
                qdot_vals.append(copy.copy(sim.data.qvel[:n_qpos_variables]))
                qddot_vals.append(copy.copy(sim.data.qacc[:n_qpos_variables]))
                torque_vals.append(copy.copy(sim.data.qfrc_applied[:n_qpos_variables]))
                x_pos = np.append(sim.data.get_geom_xpos(handle_name),
                                  tf3d.quaternions.mat2quat(sim.data.get_geom_xmat(handle_name)))
                moving_frame_xpos_world.append(copy.copy(x_pos))  # quat comes in wxyz form
                # joint_frame_in_world = np.append(sim.data.get_body_xpos(joint_body_name), obj.rotation)
                # moving_frame_xpos_ref_frame.append(copy.copy(
                #     change_frames(frame_B_wrt_A=joint_frame_in_world, pose_wrt_A=x_pos)))

                self.img_idx += 1
                img_counter += 1

            t += 1

        # h5group.create_dataset('embedding_and_params', data=embedding_and_params)
        # h5group.create_dataset('joint_frame_in_world', data=joint_frame_in_world)
        h5group.create_dataset('moving_frame_in_world', data=np.array(moving_frame_xpos_world))
        # h5group.create_dataset('moving_frame_in_ref_frame', data=np.array(moving_frame_xpos_ref_frame))
        h5group.create_dataset('depth_imgs', data=depth_imgs)

        h5group.create_dataset('q', data=np.array(q_vals))
        h5group.create_dataset('qdot', data=np.array(qdot_vals))
        h5group.create_dataset('qddot', data=np.array(qddot_vals))
        h5group.create_dataset('torques', data=np.array(torque_vals))
        h5group.create_dataset('forces', data=np.array(applied_forces))

        return True
