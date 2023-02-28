import os
import sys

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import open3d as o3d
import torch

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from src.envs.vp_drone import get_ori_drone
from src.vp_local.vp_jacobian2 import compute_ppas
from src.utils.env_utils import spherical_to_cartesian, canonical2w
from src.vp_local.vp_jacobian2 import extract_pixel_per_patch


def sample_views(Trans_can2w, r=6, step_size_r=30, step_size_t=2, full_ws=20):
    theta_range = np.linspace(-180 + step_size_r, 180, int(360 / step_size_r))
    phi_range = np.linspace(step_size_r, 90 - step_size_r, int(180 / (2 * step_size_r)) - 1)
    r_range = np.linspace(r, full_ws / 2, int((full_ws / 2 - r) / step_size_t) + 1)
    theta_mesh, phi_mesh, r_mesh = np.meshgrid(theta_range, phi_range, r_range)
    grid_sph = torch.from_numpy(np.vstack((theta_mesh.reshape(-1), phi_mesh.reshape(-1), r_mesh.reshape(-1))).T)

    grid_cart = spherical_to_cartesian(grid_sph).float()
    pos_w = canonical2w(grid_cart, Trans_can2w)
    return pos_w


def find_correlations(fname='exp_vp_local', from_save=False):
    gt_idx = 0
    n_pcd = 20000

    # Read reconstruction ground truth.
    save_path = ROOT_DIR + "/archive/airsim/" + fname
    gt_filename_stl = save_path + '/gts/gt_{:03d}.stl'.format(gt_idx)
    mesh_gt = o3d.io.read_triangle_mesh(gt_filename_stl)
    mesh_gt.compute_vertex_normals()

    pos_sample = sample_views(torch.eye(4).float())
    cpose_sample = [get_ori_drone(pos, np.zeros(3), if_return_np=True) for pos in pos_sample]
    cpose_sample = np.asarray(cpose_sample)

    if not from_save:
        verts = np.asarray(mesh_gt.vertices)
        tris = np.asarray(mesh_gt.triangles)
        normals = np.asarray(mesh_gt.triangle_normals)
        pos_actors = np.mean(verts[tris], axis=1).reshape(-1, 3)

        ppa_values = np.asarray([compute_ppas(cpose[:3].reshape(-1, 3), pos_actors, normals) for cpose in cpose_sample])
        tris_count_list = extract_pixel_per_patch(mesh_gt, cpose_sample, num_pcd=n_pcd)

        np.save(save_path + '/ppa_values.npy', ppa_values)
        np.save(save_path + '/tris_count_list.npy', np.asarray(tris_count_list))
    else:
        ppa_values = np.load(save_path + '/ppa_values.npy')
        tris_count_list = np.load(save_path + '/tris_count_list.npy')

    tris_counts = np.mean(np.stack(tris_count_list), axis=1)
    tris_coverage = np.sum(np.stack(tris_count_list) > 0, axis=1) / tris_count_list[0].shape

    pos_actors_cub = np.array([[0.25, 0, 1],
                               [0., -0.5, 1],
                               [0., 0.5, 1],
                               [0., 0, 2],
                               [-0.25, 0, 1]])
    ori_actors_cub = np.array([[1, 0, 0],
                               [0, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1],
                               [-1, 0, 0]])

    ppa_values_cub = np.asarray(
        [compute_ppas(cpose[:3].reshape(-1, 3), pos_actors_cub, ori_actors_cub) for cpose in cpose_sample])

    fig, (ax1, ax3) = plt.subplots(2, sharex=True)
    fig.set_size_inches(6, 8)

    ax1.scatter(ppa_values, tris_counts, c='b', marker='o')
    # axs[0].set_xlabel('PPA values')
    ax1.set_ylabel('Avg. Pixels Per Patches', color='blue')
    ax1.set_ylim(0.1, 0.6)
    ax1.set_title('Correlations between PPA and Reconstruction Quality')

    ax2 = ax1.twinx()
    ax2.scatter(ppa_values, tris_coverage, color="r", marker="^")
    ax2.set_ylabel("Coverage of triangles", color="red")
    ax2.set_ylim(0.05, 0.3)

    lines12 = [[(ppa_values[i], tris_counts[i]), (ppa_values[i], (tris_coverage[i] - 0.05) * 2 + 0.1)] for i in
               range(len(ppa_values))]
    lc = mc.LineCollection(lines12, colors='r', linewidths=0.5, linestyle='dashed')
    ax1.add_collection(lc)

    ax3.scatter(ppa_values, ppa_values, c='b', marker='o')
    ax3.set_xlabel('PPA values')
    ax3.set_title('Pre-modeling Actor as a Cuboid')

    ax3.scatter(ppa_values, ppa_values_cub, color="g", marker="s")
    ax3.set_ylabel("PPA values for cuboid", color="green")
    ax3.set_ylim(0.02, 0.12)

    lines13 = [[(ppa_values[i], ppa_values[i]), (ppa_values[i], ppa_values_cub[i])] for i in range(len(ppa_values))]
    lc = mc.LineCollection(lines13, colors='g', linewidths=0.5, linestyle='dashed')
    ax3.add_collection(lc)

    fig.savefig(save_path + '/sample_ppa_ppp_integ.png', dpi=100, bbox_inches='tight')
    # fig.savefig(save_path + '/sample_ppa_ppa_cub.png', dpi=100, bbox_inches='tight')


if __name__ == "__main__":
    find_correlations(from_save=True)
