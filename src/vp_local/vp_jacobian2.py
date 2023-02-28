# Reconstruction Error from chamfer loss.

import os
import sys

from tqdm import tqdm
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from src.utils.pcd_utils import draw_frame, pcd2np, np2pcd_xyz
from src.utils.env_utils import pose2mat
from src.utils.tsp_utils import safe_cons
from src.reconstruction.reconstruction import reconstruction_single_view, get_pixel_per_patch
from src.envs.vp_drone import get_ori_drone
from src.visual.visualize import vis_ppa_patch


def compute_ppas(pos_drone, pos_actor, ori_actor):
    vec = pos_actor - pos_drone
    dist = np.linalg.norm(vec, axis=1)

    cos_theta_q12norm = np.sum(vec * ori_actor, axis=1) / (dist * np.linalg.norm(ori_actor, axis=1))
    ppas = cos_theta_q12norm / dist
    ppas_avg = np.mean(np.abs(ppas))
    return ppas_avg


def local_cons(x, param_pos_drone, param_t):
    # num_actors, num_dim = param_pos_drone.shape
    # x = x.reshape(-1, num_dim)
    x = x.reshape(-1, 3)
    return param_t - np.linalg.norm(param_pos_drone - x, axis=1)


def ppa_jacob(pos_drone, pos_actor, ori_actor):
    n_patch, n_dim = pos_actor.shape
    pos_drone = np.tile(pos_drone.reshape(-1, 3), (n_patch, 1))
    ori_actor = (ori_actor.T / np.linalg.norm(ori_actor.reshape(-1, 3), axis=1)).T
    vec_a2d = pos_drone - pos_actor
    dist_a2d = np.linalg.norm(vec_a2d, axis=1).reshape(-1, 1)

    vec_jac = (ori_actor * (dist_a2d ** 2) - np.sum(vec_a2d * ori_actor, axis=1).reshape(-1,
                                                                                         1) * 2 * vec_a2d) / dist_a2d ** 4
    return np.sum(vec_jac, axis=0)


class CameraOnBoard:
    def __init__(self, h=576, w=1024, fov=45, f=None, K=None):
        self.h, self.w = h, w
        self.fov = np.deg2rad(fov)
        self.f = self.w / 2 / np.tan(self.fov / 2) if f is None else f
        cx, cy = self.w / 2, self.h / 2
        self.o3d_k = o3d.camera.PinholeCameraIntrinsic(self.w, self.h, self.f, self.f, cx, cy) if K is None else o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[0, 0], K[0, 2], K[1, 2])

        self.camera_mat = np.array([[0, 0, 1, 0],
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])

    def capture_rgbd_from_pcd(self, pcd_gt, cpose):
        trans_can2w = pose2mat(cpose)

        pcd_xyz = pcd2np(pcd_gt)
        pcd_xyz_h = np.hstack((pcd_xyz, np.ones_like(pcd_xyz[:, 0]).reshape(-1, 1)))
        pcd_can = (np.linalg.inv(trans_can2w @ self.camera_mat) @ pcd_xyz_h.T).T

        # Visualize pcd in canonical view for testing purpose.
        # o3d.visualization.draw_geometries([np2pcd_xyz(pcd_can[:, :3]), draw_frame()])

        depth_can = (self.o3d_k.intrinsic_matrix @ pcd_can[:, :3].T).T
        depth_can[:, :2] = np.floor((depth_can[:, :2].T / depth_can[:, -1].T).T)

        depth_img = np.ones((self.h, self.w)) * np.max(depth_can[:, -1])
        mask = np.zeros((self.h, self.w, 3)).astype(np.int)
        for i, point in enumerate(depth_can):
            x, y = point[:2].astype(np.int)
            depth = point[2]
            if x < 0 or x > self.w - 1 or y < 0 or y > self.h - 1:
                pass
            else:
                depth_img[y, x] = min(depth_img[y, x], depth)
                mask[y, x] = np.ones(3)
        color = mask.copy()
        return color, depth_img, mask


class VPLocalPlanner:
    def __init__(self, pos_inits, safe_rad, step_rad, camera_ob=CameraOnBoard(), pcd_gt=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_inits = pos_inits
        self.cpose_list = []
        self.ppa_list = []
        self.ppa_hist = -np.inf
        self.safe_rad = np.asarray(safe_rad)
        self.step_rad = np.asarray(step_rad)
        self.alpha = 0.05

        self.camera_ob = camera_ob
        self.pcd_gt = pcd_gt

    def vp_optimizer(self, mesh, pos_inits, optim_size=0.05):
        """
        TSP-N planner.
        Args:
            mesh: R^(n * 3). Position of patches.
            pos_inits:
            optim_size:

        Returns:
            sol_dict: Dictionary for solutions.
        """
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        pos_actors = np.mean(verts[tris], axis=1)

        vec_jacob = ppa_jacob(pos_inits, pos_actors, normals)
        vec_jacob = vec_jacob / np.linalg.norm(vec_jacob)

        print("Jacobian vector:\t", vec_jacob)
        print("Safety constraints unsatisfied", np.sum(safe_cons(pos_inits, pos_actors, self.safe_rad) < 0))

        step_size = self.step_rad
        pos_test = pos_inits + vec_jacob * step_size
        direction = (compute_ppas(pos_test, pos_actors, normals) - compute_ppas(pos_inits, pos_actors, normals)) > 0

        pos_update = pos_inits - (-1) ** direction * vec_jacob * optim_size
        n_steps = int(self.step_rad // optim_size)
        for i in range(n_steps):
            if (safe_cons(pos_update, pos_actors, self.safe_rad) < 0).any():
                break
            vec_jacob = ppa_jacob(pos_update, pos_actors, normals)
            vec_jacob = vec_jacob / np.linalg.norm(vec_jacob)
            pos_update = pos_update - (-1) ** direction * vec_jacob * optim_size

        print("Updated pos.\t", pos_update)
        # Visualization to the solution.
        pos_output = pos_update.reshape(-1, 3)
        ppas = compute_ppas(pos_update, pos_actors, normals)
        return pos_output, ppas

    def vp_local(self, cpose_inits, n_iters=3):
        cpose = cpose_inits
        print("Initialized pose\t", cpose_inits[:3])
        self.cpose_list = [cpose_inits]
        self.ppa_list = []
        for i in range(n_iters):
            self.ppa_hist = -np.inf
            color, depth, mask = self.camera_ob.capture_rgbd_from_pcd(self.pcd_gt, cpose)
            pcd_w = reconstruction_single_view(cpose, depth, color, mask, np.array([0, 0, 0, 0, 0, 0, 1]), if_can=False)
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_w, self.alpha)
            self.mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

            cpose[:3], ppas = self.vp_optimizer(self.mesh, cpose[:3])
            cpose = get_ori_drone(cpose[:3], np.mean(pcd2np(pcd_w), axis=0), if_return_np=True)

            self.cpose_list.append(cpose)
            self.ppa_list.append(ppas)
        print("PPA values: ", self.ppa_list)
        self.cpose_list, self.ppa_list = np.vstack(self.cpose_list), np.asarray(self.ppa_list)
        return cpose

    def f_ppas(self, pos_drone, pos_actors, ori_actors):
        ppas = compute_ppas(pos_drone, pos_actors, ori_actors)
        self.ppa_hist = ppas
        self.optim_iters = self.optim_iters + 1
        print("Optim. {:02d} with pos.\t".format(self.optim_iters), pos_drone)
        return -ppas


def vp_local(mesh_gt, cpose, num_pcd=10000, n_iters=5, safe_rad=6.0, step_rad=1.0):
    pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=num_pcd)
    # cpose = get_ori_drone(cpose[:3], np.zeros(3), if_return_np=True)

    planner = VPLocalPlanner(pcd_gt, cpose, safe_rad=safe_rad, step_rad=step_rad)
    cpose_optim = planner.vp_local(pcd_gt, cpose, n_iters=n_iters)
    cpose_list = planner.cpose_list

    camera_frames = [draw_frame(cpose_list[i, :3], cpose_list[i, 3:], scale=1.0) for i in range(len(cpose_list))]
    o3d.visualization.draw_geometries([draw_frame(), planner.mesh] + camera_frames, mesh_show_back_face=True)

    # np.save(save_path + '/cpose_list.npy', np.asarray(cpose_list))
    # np.save(save_path + '/ppa_list.npy', np.asarray(planner.ppa_list))
    return cpose_optim, cpose_list, planner.ppa_list


def extract_pixel_per_patch(mesh_gt, cpose_list, num_pcd=10000):
    pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=num_pcd)
    ppa_list = []

    # for cpose in cpose_list:
    for cpose in tqdm(cpose_list):
        print(f"Get pixel per patch for camera pose \n", cpose.tolist())
        camera_ob = CameraOnBoard()
        color, depth, mask = camera_ob.capture_rgbd_from_pcd(pcd_gt, cpose)
        pcd_w = reconstruction_single_view(cpose, depth, color, mask, np.array([0, 0, 0, 0, 0, 0, 1]), if_can=False)

        # o3d.visualization.draw_geometries([mesh_gt, pcd_w, draw_frame(cpose[:3], cpose[3:])])
        # o3d.visualization.draw_geometries([mesh_gt, pcd_w])

        ppas, _, _ = get_pixel_per_patch(pcd_w, mesh_gt)
        ppa_list.append(ppas)

    # np.save(save_path + '/tris_count_list.npy', np.asarray(ppa_list))
    return ppa_list


if __name__ == "__main__":
    fname = 'exp_dynamic_corl_v1'
    gt_idx = 0
    n_pcd = 20000

    # Read reconstruction ground truth.
    save_path = ROOT_DIR + "/archive/airsim/" + fname
    gt_filename_stl = save_path + '/gts/gt_{:03d}.stl'.format(gt_idx)
    mesh_gt = o3d.io.read_triangle_mesh(gt_filename_stl)

    filename = save_path + '{}_' + '{:03d}_{:03d}_{:03d}'.format(0, 42, 2)
    # cpose = np.load(filename.format('/pose/pose') + '.npy').flatten()
    cpose = np.array([3, -8, 3, 0, 0, 0, 1])
    cpose = get_ori_drone(cpose[:3], np.zeros(3), if_return_np=True)

    cpose_optim, cpose_list, ppa_values = vp_local(mesh_gt, cpose, num_pcd=n_pcd, n_iters=20, step_rad=0.2)
    tris_count_list = extract_pixel_per_patch(mesh_gt, cpose_list, num_pcd=n_pcd)

    # cpose_list = np.load(save_path + '/cpose_list.npy')
    # tris_count_list = np.load(save_path + '/tris_count_list.npy')
    # ppa_values = np.load(save_path + '/ppa_list.npy')

    # print(np.asarray(cpose_list)[:, :3])
    # print(np.mean(np.stack(tris_count_list), axis=1))

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    ax.plot(ppa_values, color="red", marker="o")
    ax.set_xlabel("Iters.", fontsize=14)
    ax.set_ylabel("PPA values", color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(np.mean(np.stack(tris_count_list), axis=1), color="blue", marker="o")
    ax2.set_ylabel("Avg. Pixel Per Patch", color="blue", fontsize=14)
    fig.savefig(save_path + '/vis_local/ppas.png',
                dpi=100,
                bbox_inches='tight')

    # camera_frames = [draw_frame(cpose_list[i, :3], cpose_list[i, 3:], scale=1.0) for i in range(len(cpose_list))]
    # o3d.visualization.draw_geometries([pcd_gt] + camera_frames, mesh_show_back_face=True)

    # Rotate the ground truth for visualization.
    rot_mat_gt = Rotation.from_euler('xyz', [270, 0, 0], degrees=True)
    pcd_gt_np = np.asarray(mesh_gt.vertices)
    pcd_gt_np_align = rot_mat_gt.apply(pcd_gt_np)
    mesh_gt.vertices = o3d.utility.Vector3dVector(pcd_gt_np_align)

    for i, tris_count in enumerate(tris_count_list):
        vis_ppa_patch(mesh_gt, tris_count, save_path + f'/vis_local/ppas_{i:03d}.png')
