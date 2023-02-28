import time
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.envs.vp_drone import VPDroneSimCV
from src.utils.env_utils import canonical2w
from src.utils.sim_utils import get_FullImage


class AirsimEnv:

    def __init__(self, args, client):
        self.args = args
        self.c = client
        self.obj_name = args.obj_name
        self.n_drones = args.n_drones
        self.full_ws = args.len_workspace
        self.safe_rad = args.safe_rad
        self.step_size_t = args.step_size_t

        self.pos_actors_cub = torch.tensor([[0.25, 0, 1],
                                            [0., -0.5, 1],
                                            [0., 0.5, 1],
                                            [0., 0, 2],
                                            [-0.25, 0, 1]])
        self.ori_actors_cub = torch.tensor([[1, 0, 0],
                                            [0, -1, 0],
                                            [0, 1, 0],
                                            [0, 0, 1],
                                            [-1, 0, 0]]).float()

        self._sim_update_states_actor(self.obj_name)
        # self.pos_drones_sim = []
        # for k in range(self.n_drones):
        #     pose_di_sim = self.c.getMultirotorState(vehicle_name='drone_' + str(k + 1))
        #     pos_di_sim = pose_di_sim.kinematics_estimated.position.to_numpy_array()
        #     self.pos_drones_sim.append(pos_di_sim)
        # self.pos_drones_sim = torch.from_numpy(np.vstack(self.pos_drones_sim))

        self.vp_robot = VPDroneSimCV(client, self.obj_name, self.n_drones)

    def _sim_update_states_actor(self, name_list):
        self.num_actors = len(name_list)
        self.pose_actors_sim = []
        self.pos_actors_sim = torch.zeros((self.num_actors, 3))
        self.ori_actors_sim_quat = torch.zeros((self.num_actors, 4))
        self.Trans_can2w_list = torch.zeros((len(name_list), 4, 4))
        self.pos_patches_sim = torch.zeros((self.num_actors, 5, 3))
        self.ori_patches_sim = torch.zeros((self.num_actors, 5, 3))

        for i in range(len(name_list)):
            name = name_list[i]
            # -------------------- Obtain pose in world frame --------------------
            pose_actor_sim = self.c.simGetObjectPose(name)
            pos_actor_sim = pose_actor_sim.position.to_numpy_array()
            ori_actor_sim_quat = pose_actor_sim.orientation.to_numpy_array()

            r = R.from_quat(ori_actor_sim_quat)
            ori_actor_sim_matrix = r.as_matrix()

            rot_const = R.from_euler('x', 0, degrees=True).as_matrix()
            Trans = torch.hstack((torch.from_numpy(rot_const @ ori_actor_sim_matrix),
                                  torch.from_numpy(rot_const @ pos_actor_sim.reshape(3, -1))))
            Trans_can2w = torch.vstack((Trans, torch.tensor([0, 0, 0, 1])))

            self.pose_actors_sim.append(pose_actor_sim)
            self.pos_actors_sim[i] = torch.from_numpy(pos_actor_sim)
            self.ori_actors_sim_quat[i] = torch.from_numpy(ori_actor_sim_quat)
            self.Trans_can2w_list[i] = Trans_can2w

            self.pos_patches_sim[i] = canonical2w(self.pos_actors_cub, Trans_can2w)
            self.ori_patches_sim[i] = (
                        torch.from_numpy(rot_const @ ori_actor_sim_matrix).float() @ self.ori_actors_cub.T).T

        self.n_patches = self.num_actors * 5
        self.pos_patches_flat = self.pos_patches_sim.reshape(-1, 3)
        self.ori_patches_flat = self.ori_patches_sim.reshape(-1, 3)

    def _sim_update_states_drone(self):
        self.pos_drones_sim, self.ori_drones_sim = self.vp_robot.feedback_pose_abs()
        return self.pos_drones_sim

    def env_setup_drone(self):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        self.vp_robot.take_off()

    def env_setup_seg(self, save_path):
        success = self.c.simSetSegmentationObjectID("[\w]*", 0, True)
        for name in self.obj_name:
            success = self.c.simSetSegmentationObjectID(name, 1)
        get_FullImage(self.c, save_path + '{}_' + '{:03d}_{:03d}'.format(0, 0), '0', self.obj_name[0])

    def execute_path_multiactors(self, path_length, path_full, vis_center_full, save_path, i):

        for j in range(np.max(path_length)):

            # Synchronize cameras by pausing the simulation.
            self.c.simPause(True)

            for k in range(self.n_drones):
                # Obtain the view point to go for k-th drone
                pos_2go_c_w_i = path_full[k, j].reshape(-1, 3).float().flatten().numpy()
                pos_vis_w_i = vis_center_full[k, j].reshape(-1, 3).float().flatten().numpy()
                print("================================================================================")
                print("Iterate %i, path step %i, drone idx %i" % (i, j, k))
                print("Planning viewing point ", pos_2go_c_w_i)
                print("Viewing center", pos_vis_w_i)

                # Automatically calculate orientation of the camera.
                camera_pose = self.vp_robot.get_ori_drone(pos_2go_c_w_i, pos_vis_w_i)
                self.vp_robot.move_to(camera_pose, self.args.time_sleep, k)

                # obtain images from camera.
                print("Save point cloud with path index %i, img index %i, drone index %i" % (i, j, k))
                img_masked, cpose = self.vp_robot.capture_rgbd('/{}_' + '{:03d}_{:03d}_{:03d}'.format(i, j, k), k)
                # np.save(save_path + '/pcd_' + '{:03d}_{:03d}_{:03d}.npy'.format(i, j, k), pcd_np_xyzrgb)

                # Save camera pose.
                # cpose = self.vp_robot.kortex.feedback_pose_abs()
                # camera_pose_save = np.concatenate((cpose[:3], R.from_euler('xyz', cpose[3:]).as_quat()))
                camera_pose_save = cpose.flatten()
                np.save(save_path + '{}_'.format('/pose/pose') + '{:03d}_{:03d}_{:03d}.npy'.format(i, j, k),
                        camera_pose_save)

            self.c.simPause(False)
            time.sleep(self.args.time_sleep)
