# Reconstruction Error from chamfer loss.

import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.envs.env_airsim import AirsimEnv
from src.envs.vp_drone import get_ori_drone
from src.utils.env_utils import pose2mat, canonical2w, setup_dir
from src.utils.pcd_utils import draw_frame, np2pcd_xyzrgb, pcd2np, pcd2np_xyzrgb
from src.vp_local.vp_planner import VPLocalPlanner
from src.vp_local.vp_sample import sample_views
from src.reconstruction.reconstruction import inside_triangle_binary


class ExpVPLocal(AirsimEnv):

    def __init__(self, args, client, fname):
        super(ExpVPLocal, self).__init__(args, client)
        self.save_path = ROOT_DIR + "/archive/airsim/" + fname
        setup_dir(self.save_path)

        gt_filename_stl = self.save_path + '/gts/michelle_0.obj'
        rot_mat_gt = Rotation.from_euler('xyz', [90, 0, 0], degrees=True)
        T_mat_gt = np.eye(4)
        T_mat_gt[:3, :3] = rot_mat_gt.as_matrix()

        self.n_pcd = 20000
        self.mesh_gt = o3d.io.read_triangle_mesh(gt_filename_stl).transform(T_mat_gt)
        self.pcd_gt = self.mesh_gt.sample_points_uniformly(number_of_points=self.n_pcd)

        self.safe_rad, self.step_rad = 8.0, 2.0
        self.full_ws = 2 * self.safe_rad + 1.0 * self.step_rad

        self.alpha = 0.03

        self.pos_sample = sample_views(torch.eye(4).float(), r=self.safe_rad, step_size_t=self.step_rad / 2.0, full_ws=self.full_ws)
        self.cpose_sample = [get_ori_drone(pos, np.zeros(3), if_return_np=True) for pos in self.pos_sample]
        self.cpose_sample = np.asarray(self.cpose_sample)
        print("Length of sample poses: ", len(self.cpose_sample))

        self.camera_mat = np.array([[0, 0, 1, 0],
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])

        np.random.seed(0)

    def vp_capture_pcd(self, pos_cam, fname):
        """
        Capture point cloud from position candidate.
        Args:
            pos_cam: camera position in canonical view.
            fname: file name (type to save).
        """
        cpos_w = canonical2w(torch.from_numpy(pos_cam).reshape(-1, 3), self.Trans_can2w_list[0])
        camera_pose = self.vp_robot.get_ori_drone(cpos_w, self.pos_actors_sim[0].numpy())
        self.vp_robot.move_to(camera_pose, self.args.time_sleep, idx_drone=0)
        pcd_c_np_xyzrgb, cpose_w = self.vp_robot.capture_pcd('/{}_' + fname, idx_drone=0)
        pcd_c = np2pcd_xyzrgb(pcd_c_np_xyzrgb)
        return pcd_c, cpose_w

    def exp_vp_local(self):
        self.c.simPause(False)

        self.env_setup_drone()

        self.vp_robot.set_fname(self.save_path)

        self._sim_update_states_drone()

        # Pause from the beginning to obtain static mesh.
        # self.c.simPause(True)

        self._sim_update_states_actor(self.obj_name)
        np.save(self.save_path + '/T_vp.npy', self.Trans_can2w_list)
        np.save(self.save_path + '/apose_vp.npy', np.hstack((self.pos_actors_sim, self.ori_actors_sim_quat)))

        # Prepare segmentation.
        self.env_setup_seg(self.save_path)

        vp_planner = VPLocalPlanner([0, 0, 0], safe_rad=self.safe_rad, step_rad=self.step_rad)

        for i, pos_cam in enumerate(self.pos_sample):
            pos_cam = pos_cam.numpy()
            cpose = get_ori_drone(pos_cam, np.zeros(3), if_return_np=True)
            print("Camera pose: ", pos_cam)

            pcd_c, cpose_save = self.vp_capture_pcd(pos_cam, '{:03d}_stay'.format(i))

            # Capture initial point cloud from the camera.
            mesh_c = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_c, self.alpha)
            mesh_c.compute_vertex_normals()

            verts = np.asarray(mesh_c.vertices)
            tris = np.asarray(mesh_c.triangles)
            normals = np.asarray(mesh_c.triangle_normals)
            pos_patches = np.mean(verts[tris], axis=1).reshape(-1, 3)

            # Transformation to the canonical view.
            pos_patches_can = canonical2w(torch.from_numpy(pos_patches), torch.from_numpy(pose2mat(cpose)))
            ori_patches_can = (pose2mat(cpose)[:3, :3] @ normals.T).T

            # 1. View planning with cuboid.
            cpos_vp, ppas, path_pos_cam_c_vp = vp_planner.vp_local_mesh(self.pos_actors_cub.numpy(),
                                                                        self.ori_actors_cub.numpy(), cpose)
            cpose_vp_path = [get_ori_drone(pos, np.zeros(3), if_return_np=True) for pos in path_pos_cam_c_vp]
            # cpose_vp_frames = [draw_frame(pos_cam[:3], pos_cam[3:], scale=0.5) for pos_cam in cpose_vp_path]
            # o3d.visualization.draw_geometries([pcd_gt, draw_frame()] + cpose_vp_frames)

            # 2. View planning with full mesh.
            cpos_mesh, ppas, path_pos_cam_w_mesh = vp_planner.vp_local_mesh(pos_patches_can.numpy(), ori_patches_can,
                                                                            cpose)
            cpose_mesh_path = [get_ori_drone(pos, np.zeros(3), if_return_np=True) for pos in path_pos_cam_w_mesh]
            # cpose_mesh_frames = [draw_frame(pos_cam[:3], pos_cam[3:], scale=0.5) for pos_cam in cpose_mesh_path]
            # o3d.visualization.draw_geometries([draw_frame()] + cpose_mesh_frames)

            # 3. View planning with greedy method.
            pos_greedy = pos_cam - pos_cam / np.linalg.norm(pos_cam) * self.step_rad
            cpos_greedy = pos_greedy if np.linalg.norm(pos_greedy) > self.safe_rad \
                else pos_greedy / np.linalg.norm(pos_greedy) * self.safe_rad
            cpose_greedy_path = [get_ori_drone(cpos_greedy, np.zeros(3), if_return_np=True)]
            # cpose_greedy_frames = [draw_frame(pos_cam[:3], pos_cam[3:], scale=0.5) for pos_cam in cpose_greedy_path]
            # o3d.visualization.draw_geometries([pcd_gt, draw_frame()] + cpose_greedy_frames)

            noise = np.random.normal(0, 0.6, 3)
            cpose_noise = cpose.copy()
            cpose_noise[:3] += noise

            # 4. View planning with cuboid noise.
            cpos_vp_noise, ppas, path_pos_cam_c_vp_noise = vp_planner.vp_local_mesh(self.pos_actors_cub.numpy(),
                                                                                    self.ori_actors_cub.numpy(),
                                                                                    cpose_noise)
            cpose_vp_noise_path = [get_ori_drone(pos, np.zeros(3), if_return_np=True) for pos in
                                   path_pos_cam_c_vp_noise]

            # 5. View planning with mesh noise.
            cpos_mesh_noise, ppas, path_pos_cam_w_mesh_noise = vp_planner.vp_local_mesh(pos_patches_can.numpy(),
                                                                                        ori_patches_can,
                                                                                        cpose_noise)
            cpose_mesh_noise_path = [get_ori_drone(pos, np.zeros(3), if_return_np=True) for pos in path_pos_cam_w_mesh_noise]

            # 6. View planning with greedy noise.
            pos_cam_noise = pos_cam + noise
            pos_greedy_noise = pos_cam_noise - pos_cam_noise / np.linalg.norm(pos_cam_noise) * self.step_rad
            cpos_greedy_noise = pos_greedy_noise if np.linalg.norm(pos_greedy_noise) > self.safe_rad \
                else pos_greedy_noise / np.linalg.norm(pos_greedy_noise) * self.safe_rad
            cpose_greedy_noise_path = [get_ori_drone(cpos_greedy_noise, np.zeros(3), if_return_np=True)]

            # Capture point cloud from camera pose in canonical view.
            pcd_stay_c, cpose_stay_w = pcd_c, cpose_save
            pcd_vp_c, cpose_vp_w = self.vp_capture_pcd(cpos_vp, '{:03d}_vp'.format(i))
            pcd_mesh_c, cpose_mesh_w = self.vp_capture_pcd(cpos_mesh, '{:03d}_mesh'.format(i))
            pcd_greedy_c, cpose_greedy_w = self.vp_capture_pcd(cpos_greedy, '{:03d}_greedy'.format(i))
            pcd_vp_noise_c, cpose_vp_noise_w = self.vp_capture_pcd(cpos_vp_noise, '{:03d}_vp_noise'.format(i))
            pcd_mesh_noise_c, cpose_mesh_noise_w = self.vp_capture_pcd(cpos_mesh_noise, '{:03d}_mesh_noise'.format(i))
            pcd_greedy_noise_c, cpose_greedy_noise_w = self.vp_capture_pcd(cpos_greedy_noise, '{:03d}_greedy_noise'.format(i))

            # Transform point cloud into the world coordinate.
            pcd_stay_w = copy.deepcopy(pcd_stay_c).transform(pose2mat(cpose_stay_w) @ self.camera_mat)
            pcd_vp_w = copy.deepcopy(pcd_vp_c).transform(pose2mat(cpose_vp_w) @ self.camera_mat)
            pcd_mesh_w = copy.deepcopy(pcd_mesh_c).transform(pose2mat(cpose_mesh_w) @ self.camera_mat)
            pcd_greedy_w = copy.deepcopy(pcd_greedy_c).transform(pose2mat(cpose_greedy_w) @ self.camera_mat)
            pcd_vp_noise_w = copy.deepcopy(pcd_vp_noise_c).transform(pose2mat(cpose_vp_noise_w) @ self.camera_mat)
            pcd_mesh_noise_w = copy.deepcopy(pcd_mesh_noise_c).transform(pose2mat(cpose_mesh_noise_w) @ self.camera_mat)
            pcd_greedy_noise_w = copy.deepcopy(pcd_greedy_noise_c).transform(pose2mat(cpose_greedy_noise_w) @ self.camera_mat)

            # Transform point cloud into the canonical coordinate.
            pcd_stay_can = copy.deepcopy(pcd_stay_w).transform(np.linalg.inv(self.Trans_can2w_list[0]))
            pcd_vp_can = copy.deepcopy(pcd_vp_w).transform(np.linalg.inv(self.Trans_can2w_list[0]))
            pcd_mesh_can = copy.deepcopy(pcd_mesh_w).transform(np.linalg.inv(self.Trans_can2w_list[0]))
            pcd_greedy_can = copy.deepcopy(pcd_greedy_w).transform(np.linalg.inv(self.Trans_can2w_list[0]))
            pcd_vp_noise_can = copy.deepcopy(pcd_vp_noise_w).transform(np.linalg.inv(self.Trans_can2w_list[0]))
            pcd_mesh_noise_can = copy.deepcopy(pcd_mesh_noise_w).transform(np.linalg.inv(self.Trans_can2w_list[0]))
            pcd_greedy_noise_can = copy.deepcopy(pcd_greedy_noise_w).transform(np.linalg.inv(self.Trans_can2w_list[0]))

            # Save the point cloud.
            pcd_path = self.save_path + '/pcd/pcd_{:03d}_'.format(i) + '{}' + '.pcd'
            print("Write point cloud to address.", pcd_path)
            o3d.io.write_point_cloud(pcd_path.format('stay'), pcd_stay_can)
            o3d.io.write_point_cloud(pcd_path.format('vp'), pcd_vp_can)
            o3d.io.write_point_cloud(pcd_path.format('mesh'), pcd_mesh_can)
            o3d.io.write_point_cloud(pcd_path.format('greedy'), pcd_greedy_can)
            o3d.io.write_point_cloud(pcd_path.format('vp_noise'), pcd_vp_noise_can)
            o3d.io.write_point_cloud(pcd_path.format('mesh_noise'), pcd_mesh_noise_can)
            o3d.io.write_point_cloud(pcd_path.format('greedy_noise'), pcd_greedy_noise_can)

            camera_frames = [draw_frame(cpose[:3], cpose[3:], scale=1.0),
                             draw_frame(cpose_vp_path[-1][:3], cpose_vp_path[-1][3:], scale=0.5),
                             draw_frame(cpose_mesh_path[-1][:3], cpose_mesh_path[-1][3:], scale=0.5),
                             draw_frame(cpose_greedy_path[-1][:3], cpose_greedy_path[-1][3:], scale=0.5),
                             draw_frame(cpose_vp_noise_path[-1][:3], cpose_vp_noise_path[-1][3:], scale=0.5),
                             draw_frame(cpose_mesh_noise_path[-1][:3], cpose_mesh_noise_path[-1][3:], scale=0.5),
                             draw_frame(cpose_greedy_noise_path[-1][:3], cpose_greedy_noise_path[-1][3:], scale=0.5)]
            # o3d.visualization.draw_geometries([self.pcd_gt, pcd_vp_can, draw_frame()])
            # o3d.visualization.draw_geometries([self.pcd_gt, pcd_greedy_can, draw_frame()])
            # o3d.visualization.draw_geometries([self.pcd_gt, pcd_stay_can, draw_frame()])

    def compute_chamfer_loss(self):

        chamfer_loss_f_list, chamfer_loss_b_list = [], []

        for i, cpose in tqdm(enumerate(self.cpose_sample)):
            pcd_path = self.save_path + '/pcd/pcd_{:03d}_'.format(i) + '{}' + '.pcd'
            pcd_vp_can = o3d.io.read_point_cloud(pcd_path.format('vp'))
            pcd_mesh_can = o3d.io.read_point_cloud(pcd_path.format('mesh'))
            pcd_greedy_can = o3d.io.read_point_cloud(pcd_path.format('greedy'))
            pcd_stay_can = o3d.io.read_point_cloud(pcd_path.format('stay'))
            pcd_vp_noise_can = o3d.io.read_point_cloud(pcd_path.format('vp_noise'))
            pcd_mesh_noise_can = o3d.io.read_point_cloud(pcd_path.format('mesh_noise'))
            pcd_greedy_noise_can = o3d.io.read_point_cloud(pcd_path.format('greedy_noise'))

            chamfer_loss_f_vp = np.mean(np.asarray(pcd_vp_can.compute_point_cloud_distance(self.pcd_gt)))
            chamfer_loss_f_mesh = np.mean(np.asarray(pcd_mesh_can.compute_point_cloud_distance(self.pcd_gt)))
            chamfer_loss_f_greedy = np.mean(np.asarray(pcd_greedy_can.compute_point_cloud_distance(self.pcd_gt)))
            chamfer_loss_f_stay = np.mean(np.asarray(pcd_stay_can.compute_point_cloud_distance(self.pcd_gt)))
            chamfer_loss_f_vp_noise = np.mean(np.asarray(pcd_vp_noise_can.compute_point_cloud_distance(self.pcd_gt)))
            chamfer_loss_f_mesh_noise = np.mean(np.asarray(pcd_mesh_noise_can.compute_point_cloud_distance(self.pcd_gt)))
            chamfer_loss_f_greedy_noise = np.mean(np.asarray(pcd_greedy_noise_can.compute_point_cloud_distance(self.pcd_gt)))

            chamfer_loss_b_vp = np.mean(np.asarray(self.pcd_gt.compute_point_cloud_distance(pcd_vp_can)))
            chamfer_loss_b_mesh = np.mean(np.asarray(self.pcd_gt.compute_point_cloud_distance(pcd_mesh_can)))
            chamfer_loss_b_greedy = np.mean(np.asarray(self.pcd_gt.compute_point_cloud_distance(pcd_greedy_can)))
            chamfer_loss_b_stay = np.mean(np.asarray(self.pcd_gt.compute_point_cloud_distance(pcd_stay_can)))
            chamfer_loss_b_vp_noise = np.mean(np.asarray(self.pcd_gt.compute_point_cloud_distance(pcd_vp_noise_can)))
            chamfer_loss_b_mesh_noise = np.mean(np.asarray(self.pcd_gt.compute_point_cloud_distance(pcd_mesh_noise_can)))
            chamfer_loss_b_greedy_noise = np.mean(np.asarray(self.pcd_gt.compute_point_cloud_distance(pcd_greedy_noise_can)))

            chamfer_loss_f_list.append(
                np.asarray([chamfer_loss_f_vp,
                            chamfer_loss_f_stay,
                            chamfer_loss_f_mesh,
                            chamfer_loss_f_greedy,
                            chamfer_loss_f_vp_noise,
                            chamfer_loss_f_mesh_noise,
                            chamfer_loss_f_greedy_noise,
                            ]) * 1000)
            chamfer_loss_b_list.append(
                np.asarray([chamfer_loss_b_vp,
                            chamfer_loss_b_stay,
                            chamfer_loss_b_mesh,
                            chamfer_loss_b_greedy,
                            chamfer_loss_b_vp_noise,
                            chamfer_loss_b_mesh_noise,
                            chamfer_loss_b_greedy_noise,
                            ]) * 1000)

            # o3d.visualization.draw_geometries([pcd_vp_can, draw_frame()])
            # o3d.visualization.draw_geometries([pcd_mesh_can, draw_frame()])
            # o3d.visualization.draw_geometries([pcd_greedy_can, draw_frame()])
            # o3d.visualization.draw_geometries([pcd_stay_can, draw_frame()])

        chamfer_loss_f_list = np.vstack(chamfer_loss_f_list)
        chamfer_loss_b_list = np.vstack(chamfer_loss_b_list)
        np.save(self.save_path + '/chamfer_loss_f_list.npy', chamfer_loss_f_list)
        np.save(self.save_path + '/chamfer_loss_b_list.npy', chamfer_loss_b_list)

        print("chamfer loss forward: vp, stay, mesh, greedy, vp_noise, mesh_noise, greedy_noise",
              np.mean(chamfer_loss_f_list, axis=0))
        print("chamfer loss backward: vp, stay, mesh, greedy, vp_noise, mesh_noise, greedy_noise",
              np.mean(chamfer_loss_b_list, axis=0))
        print(chamfer_loss_f_list.mean(axis=0) + chamfer_loss_b_list.mean(axis=0))

    def compute_coverage(self):

        verts = np.asarray(self.mesh_gt.vertices)
        tris = np.asarray(self.mesh_gt.triangles)

        coverage_list = []

        for i, cpose in tqdm(enumerate(self.cpose_sample)):
            pcd_path = self.save_path + '/pcd/pcd_{:03d}_'.format(i) + '{}' + '.pcd'
            pcd_vp_can = o3d.io.read_point_cloud(pcd_path.format('vp'))
            pcd_mesh_can = o3d.io.read_point_cloud(pcd_path.format('mesh'))
            pcd_greedy_can = o3d.io.read_point_cloud(pcd_path.format('greedy'))
            pcd_stay_can = o3d.io.read_point_cloud(pcd_path.format('stay'))
            pcd_vp_noise_can = o3d.io.read_point_cloud(pcd_path.format('vp_noise'))
            pcd_mesh_noise_can = o3d.io.read_point_cloud(pcd_path.format('mesh_noise'))
            pcd_greedy_noise_can = o3d.io.read_point_cloud(pcd_path.format('greedy_noise'))

            seen_tris = [inside_triangle_binary(verts[tris[ii]], pcd2np(pcd_vp_can)) for ii in range(tris.shape[0])]
            seen_tris = np.array(seen_tris)
            rate_vp = seen_tris.sum() / seen_tris.shape[0]

            seen_tris = [inside_triangle_binary(verts[tris[ii]], pcd2np(pcd_mesh_can)) for ii in range(tris.shape[0])]
            seen_tris = np.array(seen_tris)
            rate_mesh = seen_tris.sum() / seen_tris.shape[0]

            seen_tris = [inside_triangle_binary(verts[tris[ii]], pcd2np(pcd_greedy_can)) for ii in range(tris.shape[0])]
            seen_tris = np.array(seen_tris)
            rate_greedy = seen_tris.sum() / seen_tris.shape[0]

            seen_tris = [inside_triangle_binary(verts[tris[ii]], pcd2np(pcd_stay_can)) for ii in range(tris.shape[0])]
            seen_tris = np.array(seen_tris)
            rate_stay = seen_tris.sum() / seen_tris.shape[0]

            seen_tris = [inside_triangle_binary(verts[tris[ii]], pcd2np(pcd_vp_noise_can)) for ii in range(tris.shape[0])]
            seen_tris = np.array(seen_tris)
            rate_vp_noise = seen_tris.sum() / seen_tris.shape[0]

            seen_tris = [inside_triangle_binary(verts[tris[ii]], pcd2np(pcd_mesh_noise_can)) for ii in range(tris.shape[0])]
            seen_tris = np.array(seen_tris)
            rate_mesh_noise = seen_tris.sum() / seen_tris.shape[0]

            seen_tris = [inside_triangle_binary(verts[tris[ii]], pcd2np(pcd_greedy_noise_can)) for ii in range(tris.shape[0])]
            seen_tris = np.array(seen_tris)
            rate_greedy_noise = seen_tris.sum() / seen_tris.shape[0]

            coverage_list.append(np.asarray([rate_vp, rate_stay, rate_mesh, rate_greedy, rate_vp_noise, rate_mesh_noise, rate_greedy_noise]))

        coverage_list = np.vstack(coverage_list)
        np.save(self.save_path + '/coverage_list.npy', coverage_list)

        print("coverage_list", coverage_list.shape)
        print("coverage_list", coverage_list.mean(axis=0))


def vis_vp_local_chamfer_loss(fname='exp_vp_local'):
    save_path = ROOT_DIR + "/archive/airsim/" + fname
    chamfer_loss_f_list = np.load(save_path + '/chamfer_loss_f_list.npy')
    chamfer_loss_b_list = np.load(save_path + '/chamfer_loss_b_list.npy')

    chamfer_loss_f_vp = chamfer_loss_f_list[:, 0]
    chamfer_loss_f_stay = chamfer_loss_f_list[:, 1]
    chamfer_loss_f_mesh = chamfer_loss_f_list[:, 2]
    chamfer_loss_f_greedy = chamfer_loss_f_list[:, 3]

    chamfer_loss_b_vp = chamfer_loss_b_list[:, 0]
    chamfer_loss_b_stay = chamfer_loss_b_list[:, 1]
    chamfer_loss_b_mesh = chamfer_loss_b_list[:, 2]
    chamfer_loss_b_greedy = chamfer_loss_b_list[:, 3]

    chamfer_loss_avg_vp = (chamfer_loss_f_vp + chamfer_loss_b_vp) / 2
    chamfer_loss_avg_stay = (chamfer_loss_f_stay + chamfer_loss_b_stay) / 2
    chamfer_loss_avg_mesh = (chamfer_loss_f_mesh + chamfer_loss_b_mesh) / 2
    chamfer_loss_avg_greedy = (chamfer_loss_f_greedy + chamfer_loss_b_greedy) / 2

    fig, ax = plt.subplots()
    ax.set_xlabel('Chamfer loss (lower bound)')
    ax.set_ylabel('Chamfer loss for view planning methods')
    ax.set_title('View planning for reconstruction quality')

    ax.scatter(chamfer_loss_f_stay, chamfer_loss_f_stay, c='b', marker='o', label='lower bound')
    ax.scatter(chamfer_loss_f_stay, chamfer_loss_f_vp, c='r', marker='^', label='vp')
    ax.scatter(chamfer_loss_f_stay, chamfer_loss_f_greedy, c='g', marker='s', label='greedy')
    ax.scatter(chamfer_loss_f_stay, chamfer_loss_f_mesh, c='y', marker='v', label='upper bound')

    ax.legend()
    ax.grid(True)
    fig.savefig(save_path + '/vp_chamfer_f.png', dpi=100, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_xlabel('Chamfer loss (lower bound)')
    ax.set_ylabel('Chamfer loss for view planning methods')
    ax.set_title('View planning for reconstruction quality')

    ax.scatter(chamfer_loss_b_stay, chamfer_loss_b_stay, c='b', marker='o', label='lower bound')
    ax.scatter(chamfer_loss_b_stay, chamfer_loss_b_vp, c='r', marker='^', label='vp')
    ax.scatter(chamfer_loss_b_stay, chamfer_loss_b_greedy, c='g', marker='s', label='greedy')
    ax.scatter(chamfer_loss_b_stay, chamfer_loss_b_mesh, c='y', marker='v', label='upper bound')

    ax.legend()
    ax.grid(True)
    fig.savefig(save_path + '/vp_chamfer_b.png', dpi=100, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_xlabel('Chamfer loss (stay)')
    ax.set_ylabel('Chamfer loss for view planning methods')
    ax.set_title('View planning for reconstruction quality')

    ax.scatter(chamfer_loss_avg_stay, chamfer_loss_avg_stay, c='b', marker='o', label='stay')
    ax.scatter(chamfer_loss_avg_stay, chamfer_loss_avg_vp, c='r', marker='^', label='vp')
    ax.scatter(chamfer_loss_avg_stay, chamfer_loss_avg_greedy, c='g', marker='s', label='greedy')
    ax.scatter(chamfer_loss_avg_stay, chamfer_loss_avg_mesh, c='y', marker='v', label='mesh')

    ax.legend()
    ax.grid(True)
    fig.savefig(save_path + '/vp_chamfer_avg.png', dpi=100, bbox_inches='tight')


if __name__ == "__main__":
    import airsim as sim
    from src.args import VPArgs

    ROOT_DIR = os.path.abspath("../../")
    sys.path.append(ROOT_DIR)

    opts = VPArgs()
    args = opts.get_args()

    args.n_actors = 1
    args.n_drones = 1

    args.obj_name = ['person_actor_1']

    c = sim.MultirotorClient()
    c.confirmConnection()

    exp = ExpVPLocal(args, c, 'exp_vp_local')
    # exp.exp_vp_local()

    exp.compute_chamfer_loss()
    exp.compute_coverage()
    # vis_vp_local_chamfer_loss('exp_vp_local')
