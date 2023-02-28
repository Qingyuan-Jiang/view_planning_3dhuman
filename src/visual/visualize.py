import open3d as o3d
import numpy as np
import copy

import matplotlib.pyplot as plt
import os
import sys

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)


def vis_colored_mesh(mesh_gt, vis_index, save_path, mode='coverage'):
    color_mesh = copy.deepcopy(mesh_gt)
    colors = np.ones_like(np.asarray(mesh_gt.vertices)) * 0.8
    # for k in range(n_drones):
    #     colors[np.unique(np.asarray(mesh_gt.triangles)[vis_list[k]]), -1] = 0.8
    if mode == 'overlap':
        colors[np.unique(np.asarray(mesh_gt.triangles)[vis_index]), :-1] = 0
    elif mode == 'coverage':
        colors[np.unique(np.asarray(mesh_gt.triangles)[vis_index]), 1:] = 0
    elif mode == 'mix':
        colors[np.unique(np.asarray(mesh_gt.triangles)[vis_index]), 1:] = 0
        colors[np.unique(np.asarray(mesh_gt.triangles)[vis_index]), :-1] = 0
    else:
        raise Exception('Wrong mode')

    color_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([color_mesh, pcd_i_can])
    # o3d.visualization.draw_geometries([color_mesh])

    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(color_mesh)
    vis.update_geometry(color_mesh)
    # vis.add_geometry(pcd_i_can)
    # vis.update_geometry(pcd_i_can)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()


def vis_coverage(pcd_i, mesh_gt, seen_tris, save_fname):
    verts = np.asarray(mesh_gt.vertices)
    tris = np.asarray(mesh_gt.triangles)

    color_mesh = copy.deepcopy(mesh_gt)
    colors = np.ones_like(verts) * 0.8
    colors[np.unique(tris[seen_tris]), 1:] = 0.
    color_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([color_mesh, pcd_i_can])

    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(color_mesh)
    vis.update_geometry(color_mesh)
    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(save_path + '/vis/coverage_{:03d}_{:03d}.png'.format(i, j))
    vis.capture_screen_image(save_fname)
    vis.destroy_window()


def vis_ppa_patch(mesh_gt, tris_count, save_fname):
    verts = np.asarray(mesh_gt.vertices)
    tris = np.asarray(mesh_gt.triangles)

    color_mesh = copy.deepcopy(mesh_gt)

    colors = np.ones_like(verts) * 0.8
    for i in range(1, tris_count.max()):
        colors[np.unique(tris[tris_count == i]), 0] = 0.8 ** i
        colors[np.unique(tris[tris_count == i]), 1:] = 0.
    color_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([color_mesh])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(color_mesh)
    vis.update_geometry(color_mesh)
    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(save_path + '/vis/coverage_{:03d}_{:03d}.png'.format(i, j))
    vis.capture_screen_image(save_fname)
    vis.destroy_window()


def plot_rates(fname='circle005_overlap_30_michelle', rate_address='/vis/coverage_rate', nframes=180):
    save_path = ROOT_DIR + "/archive/airsim/" + fname

    ang_delta = int(360 / nframes)
    rate_list = np.load(save_path + rate_address + '.npy')
    ang_axis = np.arange(0, 360, ang_delta)
    plt.plot(ang_axis, np.asarray(rate_list))
    # plt.show()
    plt.savefig(save_path + rate_address + '.png')
