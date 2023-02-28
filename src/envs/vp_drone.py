import os
import sys
import time

import airsim as sim
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

from src.utils.sim_utils import get_FullImage, get_RGBImage, cpose2msg
from src.utils.env_utils import setup_dir
from src.reconstruction.reconstruction import pcd_from_rgbd_mask

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)


class CameraOnBoard:
    def __init__(self, c, obj_name, h=576, w=1024, fov=np.deg2rad(45)):
        print("Create Camera Onboard ROS node. --------------------")
        self.c = c
        self.obj_name = obj_name
        success = self.c.simSetSegmentationObjectID("[\w]*", 0, True)
        for name in self.obj_name:
            success = self.c.simSetSegmentationObjectID(name, 1)
        get_FullImage(self.c, ROOT_DIR + "/archive/airsim/" + 'test', '0', self.obj_name[0])

        self.h, self.w = h, w
        f = w / 2 / np.tan(fov / 2)
        cx, cy = w / 2, h / 2
        self.o3d_K = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, cx, cy)

    def read_rgb(self, fnames):
        get_RGBImage(self.c, fnames)

    def capture_rgbd(self, fnames, idx_drone, obj_name):
        img_rgb, img_seg, img_masked, img_depth = get_FullImage(self.c, fnames)
        return img_rgb, img_seg, img_masked, img_depth

    def read_pcd(self, fnames, idx_drone, obj_name, vis_pcd=False):
        img_rgb, img_seg, img_masked, img_depth = get_FullImage(self.c, fnames, idx_drone, obj_name)
        # xx, yy = np.meshgrid(np.linspace(0, self.w - 1, self.w), np.linspace(0, self.h - 1, self.h))
        # uv_h = np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, self.h * self.w))
        # uv_3d = np.linalg.inv(self.o3d_K.intrinsic_matrix) @ uv_h
        # rays_d = uv_3d / uv_3d[-1, :]
        # pcd_c = rays_d.T * np.tile(img_depth.reshape(-1, 1), (1, 3))
        pcd_c_np_xyzrgb = pcd_from_rgbd_mask(self.o3d_K, img_rgb, img_depth, img_seg)
        return pcd_c_np_xyzrgb


def get_ori_drone(pos_drone, pos_target, if_return_np=False):
    pos_view = (torch.from_numpy(pos_target).reshape(-1, 3) - pos_drone.reshape(-1, 3)).flatten()
    theta = torch.atan2(pos_view[1], pos_view[0]).numpy().tolist()

    pitch = torch.atan2(pos_view[2], torch.norm(pos_view[:-1]))
    roll = np.pi
    yaw = theta
    pos_2go_c_w_msg = pos_drone.flatten().tolist()
    camera_pose = sim.Pose(sim.Vector3r(pos_2go_c_w_msg[0], pos_2go_c_w_msg[1], pos_2go_c_w_msg[2]),
                           sim.to_quaternion(-pitch, roll, yaw))
    if if_return_np:
        return np.hstack((camera_pose.position.to_numpy_array(), camera_pose.orientation.to_numpy_array()))
    else:
        return camera_pose


class VPDroneSimCV:
    """docstring for View Planning Robot."""

    def __init__(self, c, obj_names, n_drones):
        print("Create View Planning Drone ------------------------")

        # initialize the drones and cameras.
        self.c = c
        self.obj_names = obj_names
        self.n_drones = n_drones

        self.camera_ob = CameraOnBoard(c, obj_names)
        self.rot_const = R.from_euler('x', 180, degrees=True).as_matrix()

    def set_fname(self, fname):
        self.fname = fname

    def take_off(self):
        self.pos_drones_sim = [torch.tensor([3 * (int((self.n_drones - 1)/2) - i),
                                             3 * (int((self.n_drones - 1)/2) - i),
                                             8]) for i in range(self.n_drones)]
        self.pos_drones_sim = torch.vstack(self.pos_drones_sim).float()
        self.ori_drones_sim = torch.tile(torch.tensor([[0, 0, 0, 1]]), (self.n_drones, 1)).float()

        for k in range(self.n_drones):
            cpose_msg = cpose2msg(self.pos_drones_sim[k].numpy(), self.ori_drones_sim[k].numpy())
            self.move_to(cpose_msg, 0.1, k)
        return self.pos_drones_sim, self.ori_drones_sim

    def feedback_pose_abs(self):
        # for k in range(self.n_drones):
        #     pose_drone_sim_d = self.c.simGetCameraInfo(str(k)).pose
        #     pos_actor_sim_d = pose_drone_sim_d.position.to_numpy_array()
        #     ori_actor_sim_d_quat = pose_drone_sim_d.orientation.to_numpy_array()
        #     ori_actor_sim_d_matrix = R.from_quat(ori_actor_sim_d_quat).as_matrix()
        #
        #     ori_actor_sim_w_quat = R.from_matrix(self.rot_const @ ori_actor_sim_d_matrix).as_quat()
        #     pos_actor_sim_w = (self.rot_const @ pos_actor_sim_d.reshape(3, -1)).reshape(-1, 3)
        #
        #     self.pos_drones_sim[k] = torch.from_numpy(pos_actor_sim_w).float()
        #     self.ori_drones_sim[k] = torch.from_numpy(ori_actor_sim_w_quat).float()
        print("Feedback drones position\n", self.pos_drones_sim)
        print("Feedback drones orientation\n", self.ori_drones_sim)
        return self.pos_drones_sim, self.ori_drones_sim

    def locate_actors(self):
        return

    def move_to(self, camera_pose, time_sleep, idx_drone):
        """
        A wrapper function over the arm controller.
        The manipulator controller is blocked when point-to-point trajectory is sent so sometime the
        end effector couldn't reach the target position within the given time
        """
        self.pos_drones_sim[idx_drone] = torch.from_numpy(camera_pose.position.to_numpy_array().astype(np.float16)).float()
        self.ori_drones_sim[idx_drone] = torch.from_numpy(camera_pose.orientation.to_numpy_array().astype(np.float16)).float()

        print("--------------------------------")
        print("Moving to pos. \t", self.pos_drones_sim[idx_drone])
        print("Moving to ori. \t", self.ori_drones_sim[idx_drone])

        pos_drone_sim_w, ori_actor_sim_w_quat = self.pos_drones_sim[idx_drone].numpy(), self.ori_drones_sim[idx_drone].numpy()
        ori_drone_sim_w_matrix = R.from_quat(ori_actor_sim_w_quat).as_matrix()

        pos_drone_sim_d = (self.rot_const @ pos_drone_sim_w.reshape(3, -1)).reshape(-1, 3)
        ori_drone_sim_d_quat = R.from_matrix(self.rot_const @ ori_drone_sim_w_matrix).as_quat()

        cpose_msg = cpose2msg(pos_drone_sim_d, ori_drone_sim_d_quat)
        self.c.simSetVehiclePose(cpose_msg, True)
        time.sleep(time_sleep)

    def capture_rgbd(self, fname, idx_drone):
        img_rgb, img_seg, img_masked, img_depth = self.camera_ob.capture_rgbd(self.fname + fname, idx_drone, self.obj_names)
        cpose = np.concatenate((self.pos_drones_sim[idx_drone], self.ori_drones_sim[idx_drone])).astype(np.float16)
        print("Capture rgbd from pos.\t", cpose[:3])
        print("Capture rgbd from ori.\t", cpose[3:])
        return img_masked, cpose

    def capture_pcd(self, fname, idx_drone):
        pcd_c = self.camera_ob.read_pcd(self.fname + fname, idx_drone, self.obj_names)
        cpose = np.concatenate((self.pos_drones_sim[idx_drone], self.ori_drones_sim[idx_drone])).astype(np.float16)
        print("Capture pcd from pos.\t", cpose[:3])
        print("Capture pcd from ori.\t", cpose[3:])
        return pcd_c, cpose

    def get_ori_drone(self, pos_drone, pos_target):
        return get_ori_drone(pos_drone, pos_target)
