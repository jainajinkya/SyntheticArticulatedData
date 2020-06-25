import SyntheticArticulatedData.generation.calibrations as calibrations
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import transforms3d as tf3d


def write_urdf(filename, xml):
    header = '''<?xml version="1.0"?>
<robot name="cabinet">'''

    footer = '''
</robot>'''

    with open(filename, "w") as text_file:
        text_file.write(header + xml + footer)


def get_cam_params(cam='Kinect'):
    if cam == 'Kinect':
        # znear= pyro.sample("znear",dist.Uniform(0.09, 0.11)).item()
        # zfar = pyro.sample("zfar", dist.Uniform(11,13)).item()
        # fovy = pyro.sample("fovy", dist.Uniform(66, 74)).item()
        znear = calibrations.znear
        zfar = calibrations.zfar
        fovy = calibrations.color_fov_y

    elif cam == 'RealSense':
        znear = pyro.sample("znear", dist.Uniform(0.18, 0.22)).item()
        zfar = pyro.sample("zfar", dist.Uniform(9, 11)).item()
        fovy = pyro.sample("fovy", dist.Uniform(84, 89)).item()
    else:
        raise Exception('check your camera name')
    return znear, zfar, fovy


def make_single_string(param):
    return '" %f "' % param


def make_string(param_tuple):
    return ' " %f %f %f " ' % param_tuple


def angle_to_quat(angle, axis=None):
    if axis is None:
        axis = [0, 0, 1]
    qx = axis[0] * np.sin(angle / 2)
    qy = axis[1] * np.sin(angle / 2)
    qz = axis[2] * np.sin(angle / 2)
    qw = np.cos(angle / 2)
    return np.array([qw, qx, qy, qz])


def shuffle_quat(quat):
    # convert quaternion from wxyz to zxyw
    return [quat[3], quat[1], quat[2], quat[0]]


def make_quat_string(quat):
    # quat=shuffle_quat(quat)
    return ' " %f %f %f %f" ' % tuple(quat)


def get_cam_relative_params(obj):
    obj_position = obj.pose
    obj_angle = 3.1415926 - obj.rotation
    obj_axis = [0, 0, 1]

    obj_rot_matrix = tf3d.axangles.axangle2mat(obj_axis, obj_angle)
    obj_transform = tf3d.affines.compose(obj_position, obj_rot_matrix, np.ones(3))

    axis = obj.params[0]
    door = obj.params[1]
    axis_in_obj_frame = [axis[0], axis[1], axis[2], 1.0]
    axis_in_world_frame = (np.matmul(obj_transform, axis_in_obj_frame))[:3]
    ax_quat = tf3d.quaternions.axangle2quat(obj_axis, obj.rotation)  # [w, qx, qy, qz]

    axis_and_quat = np.append(axis_in_world_frame, ax_quat)
    axis_and_door = np.append(axis_and_quat, door)

    # assert len(axis_and_door) == 10
    return axis_and_door


def get_cam_relative_params2(obj):
    '''
        Converts an object's axis coordinates from object frame to camera frame
    '''
    obj_position = obj.pose

    obj_rot_matrix = tf3d.quaternions.quat2mat(obj.rotation)
    obj_transform = tf3d.affines.compose(obj_position, obj_rot_matrix, np.ones(3))

    axis = obj.params[0]
    door = obj.params[1]

    axis_in_obj_frame = [axis[0], axis[1], axis[2], 1.0]
    axis_in_world_frame = (np.matmul(obj_transform, axis_in_obj_frame))[:3]
    ax_quat = obj.rotation  # [w, qx, qy, qz]

    axis_and_quat = np.append(axis_in_world_frame, ax_quat)
    axis_and_door = np.append(axis_and_quat, door)

    # assert len(axis_and_door) == 10
    return axis_and_door


def transform_param(axis, door, obj):
    obj_position = obj.pose

    obj_rot_matrix = tf3d.quaternions.quat2mat(obj.rotation)
    obj_transform = tf3d.affines.compose(obj_position, obj_rot_matrix, np.ones(3))

    axis_in_obj_frame = [axis[0], axis[1], axis[2], 1.0]
    axis_in_world_frame = (np.matmul(obj_transform, axis_in_obj_frame))[:3]
    ax_quat = obj.rotation  # [w, qx, qy, qz]

    if obj.type == 3:
        # drawer - axis of translation is in x
        ax_quat = obj.rotation * tf3d.quaternions.axangle2quat([1, 0, 0], -1.57)

    axis_and_quat = np.append(axis_in_world_frame, ax_quat)
    return axis_and_quat, door


def sample_quat():
    rpy = pyro.sample('euler',
                      dist.Uniform(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([2 * 3.14, 2 * 3.14, 2 * 3.14]))).numpy()
    quat = tf3d.euler.euler2quat(rpy[0], rpy[1], rpy[2])
    return quat


def sample_pose():
    xyz = pyro.sample('origin', dist.Uniform(torch.tensor([1.0, -0.5, -0.7]), torch.tensor([2.0, 0.5, 0.3]))).numpy()
    angle = pyro.sample('angle', dist.Uniform(3 / 4 * 3.14, 5 / 4 * 3.14)).item()  # Sampling only wrt z-axes
    # angle_y = pyro.sample('angle_y', dist.Beta(0.5, 0.5)).item() * np.pi/2  # Setting range to [0, pi/2]
    return tuple(xyz), angle


def sample_pose_2():
    xyz = pyro.sample('origin', dist.Uniform(torch.tensor([1.0, -0.5, -0.7]), torch.tensor([2.0, 0.5, 0.3]))).numpy()
    angle_x = pyro.sample('angle_x', dist.Uniform(-np.pi / 4, np.pi / 4)).item()
    # angle_y = pyro.sample('angle_y', dist.Beta(0.5, 0.5)).item() * np.pi/2  # Setting range to [0, pi/2]
    angle_y = pyro.sample('angle_y', dist.Uniform(0., np.pi / 2)).item()  # Setting range to [0, pi/2]
    angle_z = pyro.sample('angle_z', dist.Uniform(3 / 4 * 3.14, 5 / 4 * 3.14)).item()
    return tuple(xyz), angle_x, angle_y, angle_z


def sample_pose_sapien():
    xyz = pyro.sample('origin', dist.Uniform(torch.tensor([1.75, -0.5, -0.7]), torch.tensor([2.75, 0.5, 0.3]))).numpy()
    angle_x = pyro.sample('angle_x', dist.Uniform(-np.pi/4, np.pi/4)).item()
    # angle_y = pyro.sample('angle_y', dist.Beta(0.5, 0.5)).item() * np.pi/2  # Setting range to [0, pi/2]
    angle_y = pyro.sample('angle_y', dist.Uniform(-np.pi/4, np.pi/4)).item()  # Setting range to [0, pi/2]
    angle_z = pyro.sample('angle_z', dist.Uniform(-1 / 4 * 3.14, 1 / 4 * 3.14)).item()
    return tuple(xyz), angle_x, angle_y, angle_z


def sample_pose_fridge(l, w):
    y = pyro.sample('y', dist.Uniform(torch.tensor([-1.0]),
                                      torch.tensor([1.0]))).item()
    x_min = (y + w) / np.arctan(27. * 3.14 / 180.) + l
    x = pyro.sample('x', dist.Uniform(torch.tensor([x_min]),
                                      torch.tensor([3.5]))).item()
    z = pyro.sample('z', dist.Uniform(torch.tensor([-1.5]),
                                      torch.tensor([-0.7]))).item()

    x_min = max(x_min, 1.5)

    xyz = np.array([x, y, z])
    # print(xyz)
    # xyz=pyro.sample('origin', dist.Uniform(torch.tensor([x_min,-1.0,-2.0]),torch.tensor([3.3, 1.0 , -0.7]))).numpy()
    angle = pyro.sample('angle', dist.Uniform(3 / 4 * 3.14, 5 / 4 * 3.14)).item()
    return tuple(xyz), angle


def sample_pose_drawer():
    xyz = pyro.sample('origin', dist.Uniform(torch.tensor([1.0, -0.5, -0.8]), torch.tensor([2.0, 0.5, 0.0]))).numpy()
    angle_z = pyro.sample('angle', dist.Uniform(3 / 4 * 3.14, 5 / 4 * 3.14)).item()
    angle_y = pyro.sample('angle_y', dist.Beta(0.5, 0.5)).item() * (-np.pi / 2)  # Setting range to [-pi/2, 0]
    return tuple(xyz), angle_z, angle_y


def sample_pose_drawer_2():
    xyz = pyro.sample('origin', dist.Uniform(torch.tensor([1.0, -0.5, -0.7]), torch.tensor([2.0, 0.5, 0.0]))).numpy()
    angle_x = pyro.sample('angle_x', dist.Uniform(-np.pi / 6, np.pi / 6)).item()
    # angle_y = pyro.sample('angle_y', dist.Beta(0.5, 0.5)).item() * (-np.pi/2)  # Setting range to [-pi/2, 0]
    angle_y = pyro.sample('angle_y', dist.Uniform(-np.pi / 2, 0)).item()
    angle_z = pyro.sample('angle_z', dist.Uniform(3 / 4 * 3.14, 5 / 4 * 3.14)).item()
    return tuple(xyz), angle_x, angle_y, angle_z


def change_frames(frame_B_wrt_A, pose_wrt_A):
    A_T_pose = tf3d.affines.compose(T=pose_wrt_A[:3],
                                    R=tf3d.quaternions.quat2mat(pose_wrt_A[3:]),  # quat in  wxyz
                                    Z=np.ones(3))
    A_rot_mat_B = tf3d.quaternions.quat2mat(frame_B_wrt_A[3:])

    # Following as described in Craig
    B_T_A = tf3d.affines.compose(T=-A_rot_mat_B.T.dot(frame_B_wrt_A[:3]),
                                 R=A_rot_mat_B.T,
                                 Z=np.ones(3))

    B_T_pose = B_T_A.dot(A_T_pose)
    trans, rot, scale, _ = tf3d.affines.decompose44(B_T_pose)
    quat = tf3d.quaternions.mat2quat(rot)
    return np.concatenate((trans, quat))  # return quat in wxyz
