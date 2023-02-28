# Reconstruction Error from chamfer loss.

import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import copy

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
import matplotlib.pyplot as plt

from src.utils.pcd_utils import draw_frame, pcd2np, pcd_update_points, pcd_cat_list
from src.utils.env_utils import w2canonical, canonical2w
from src.utils.env_utils import pose2mat
from src.args import TestArgs
from src.utils.pcd_utils import np2pcd_xyzrgb


def pcd_from_rgbd_mask(o3d_K, rgb, depth, mask_img, h=576, w=1024):
    xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    uv_h = np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
    uv_3d = np.linalg.inv(o3d_K.intrinsic_matrix) @ uv_h
    rays_d = uv_3d / uv_3d[-1, :]
    pcd_c = rays_d.T * np.tile(depth.reshape(-1, 1), (1, 3))

    mask = (mask_img > 0).reshape(-1, 3)
    pcd_actor_w = pcd_c[mask[:, 0], :3]
    color_actor = rgb.reshape(-1, 3)[mask[:, 0]] / 255
    return np.hstack((pcd_actor_w, color_actor))


def reconstruction_from_frames_pcd(fname, niters, frames, post_name='reconstruct', ws=0.2):
    save_path = ROOT_DIR + "/archive/kortex/" + fname
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save Intrinsic matrix.
    # h, w = 576, 1024
    # fov = np.deg2rad(45)
    # f = w / 2 / np.tan(fov / 2)
    # cx, cy = w / 2, h / 2
    # o3d_K = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, cx, cy)

    for i in range(niters):
        pcd_sum = o3d.geometry.PointCloud()
        pcd_np_list, rgb_np_list = [], []
        frame_list = []
        frame_w = draw_frame()
        # w_frame = draw_frame(np.array([-40, 0, -8]) + np.array([1, 0, 0]), [0, 0, 0, 1])

        for j in range(frames):

            filename = save_path + '{}_' + '{:03d}_{:03d}_{:03d}'.format(i, j, 0)
            if not os.path.exists(filename.format('/pcd') + '.npy'):
                print("File not exists with address. ", filename.format('/pcd') + '.npy')
                break

            print("Start merging frame trajectory index %i, frame %i" % (i, j))
            cpose = np.load(filename.format('/pose') + '.npy')
            # depth = np.load(filename.format('/depth') + '.npy')
            # color = np.array(Image.open(filename.format('/rgb') + '.png'))
            # mask_img = np.array(Image.open(filename.format('/mask') + '.png'))
            pcd_c_np_xyzrgb = np.load(filename.format('/pcd') + '.npy')

            pcd_c_xyzrgb = np2pcd_xyzrgb(pcd_c_np_xyzrgb)
            # o3d.visualization.draw_geometries([pcd_c_xyzrgb, draw_frame()])

            pcd_c_xyz = pcd_c_np_xyzrgb[:, :3]

            # camera_mat = np.array([[0, 0, 1, 0.02750],
            #                        [1, 0, 0, 0.066],
            #                        [0, 1, 0, -0.00305],
            #                        [0, 0, 0, 1]])
            camera_mat = np.eye(4)

            Trans_mat = pose2mat(cpose)
            frame_list.append(draw_frame(cpose[:3], cpose[3:], scale=0.2))

            pcd_w_np_xyz = (Trans_mat @ camera_mat @ np.hstack(
                (pcd_c_xyz, np.ones_like(pcd_c_xyz[:, 0]).reshape(-1, 1))).T).T

            pcd_np_list.append(pcd_w_np_xyz[:, :3])
            rgb_np_list.append(pcd_c_np_xyzrgb[:, 3:])

        pcd_sum.points = o3d.utility.Vector3dVector(np.vstack(pcd_np_list))
        pcd_sum.colors = o3d.utility.Vector3dVector(np.vstack(rgb_np_list))

        # o3d.visualization.draw_geometries([frame_w, pcd_sum] + frame_list)

        # pcd_downsample = pcd_sum.voxel_down_sample(voxel_size=0.005)
        pcd_downsample = pcd_sum

        pose_actor = np.load(save_path + '/apose_{:03d}.npy'.format(i))
        pcd_list = []
        for i in range(len(pose_actor)):
            pos_actor_i = pose_actor[i, :3]
            x, y, z = pos_actor_i
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(x - ws / 2, y - ws / 2, z - ws / 2),
                                                       max_bound=(x + ws / 2, y + ws / 2, z + ws / 2))
            pcd_actor_i = copy.deepcopy(pcd_downsample).crop(bbox)
            pcd_list.append(pcd_actor_i)

        pcd_all_actors = pcd_cat_list(pcd_list)
        o3d.visualization.draw_geometries([frame_w, pcd_all_actors] + frame_list)

        o3d.io.write_point_cloud(save_path + '/pcd_{:03d}_'.format(i) + post_name + '.pcd', pcd_all_actors)
        o3d.io.write_point_cloud(save_path + '/pcd_{:03d}_'.format(i) + post_name + '.ply', pcd_all_actors)


def reconstruction_single_view(cpose, depth, color, mask_img, apose,
                               h=576, w=1024, fov=np.deg2rad(45), if_can=True):
    f = w / 2 / np.tan(fov / 2)
    cx, cy = w / 2, h / 2
    o3d_K = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, cx, cy)
    camera_mat = np.array([[0, 0, 1, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])

    # frame_list.append(draw_frame(cpose[:3], cpose[3:]))

    rot_mat = Rotation.from_quat(cpose[3:]).as_matrix()
    Trans = np.hstack((rot_mat, cpose[:3].reshape(3, -1)))
    Trans_mat = np.vstack((Trans, np.array([0, 0, 0, 1])))

    # Classical methods.
    xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    uv_h = np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
    uv_3d = np.linalg.inv(o3d_K.intrinsic_matrix) @ uv_h
    rays_d = uv_3d / uv_3d[-1, :]
    pcd_c = rays_d.T * np.tile(depth.reshape(-1, 1), (1, 3))

    pcd_w = (Trans_mat @ camera_mat @ np.hstack((pcd_c, np.ones_like(pcd_c[:, 0]).reshape(-1, 1))).T).T

    mask = (mask_img > 0).reshape(-1, 3)
    pcd_actor_w = pcd_w[mask[:, 0], :3]
    color_actor = color.reshape(-1, 3)[mask[:, 0]] / 255

    pcd_w = o3d.geometry.PointCloud()
    pcd_w.points = o3d.utility.Vector3dVector(pcd_actor_w)
    pcd_w.colors = o3d.utility.Vector3dVector(color_actor)
    # o3d.visualization.draw_geometries([pcd_i])

    if if_can:
        trans_actor = pose2mat(apose)
        pcd_can_torch = w2canonical(torch.from_numpy(pcd2np(pcd_w).reshape(-1, 3)),
                                    torch.from_numpy(trans_actor))
        pcd_can = pcd_update_points(pcd_w, pcd_can_torch.numpy())
        return pcd_can
    else:
        return pcd_w


def inside_triangle(tri_pts, pt, dist_tol=0.01, tri_tol=0.00001):
    p = pt
    u = tri_pts[1] - tri_pts[0]
    v = tri_pts[2] - tri_pts[0]
    n = torch.cross(u, v)
    w = p - tri_pts[0]
    gamma = (torch.cross(u.tile(len(w), 1), w) @ n) / (n @ n)
    beta = (torch.cross(w, v.tile(len(w), 1)) @ n) / (n @ n)
    alpha = 1 - gamma - beta

    xx = torch.stack([alpha, beta, gamma])
    res = torch.all((xx <= 1 + tri_tol) & (xx >= 0 - tri_tol), 0)

    n_ = n / torch.norm(n)
    dists = torch.abs(p @ n_ - tri_pts[0] @ n_)
    res = res * (dists <= dist_tol)
    return res


def inside_triangle_binary(tri_pts, pt, dist_tol=0.01, tri_tol=0.00001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tri_pts = torch.from_numpy(tri_pts).to(device)
    pt = torch.from_numpy(pt).to(device)
    return inside_triangle(tri_pts, pt, dist_tol=dist_tol, tri_tol=tri_tol).cpu().numpy().max()


def inside_triangle_count(tri_pts, pt, dist_tol=0.01, tri_tol=0.00001):

    return inside_triangle(tri_pts, pt, dist_tol=dist_tol, tri_tol=tri_tol).cpu().numpy().sum()


def get_coverage(pcd_i, mesh_gt,
                 rot_mat_i=Rotation.from_euler('xyz', [90, 0, 0], degrees=True),
                 rot_mat_gt=Rotation.from_euler('xyz', [270, 0, 0], degrees=True)):
    pcd_i_np = rot_mat_i.apply(pcd2np(pcd_i))
    pcd_i_can = pcd_update_points(pcd_i, pcd_i_np)

    # Align ground truth.
    pcd_gt_np = np.asarray(mesh_gt.vertices)
    pcd_gt_np_align = rot_mat_gt.apply(pcd_gt_np)
    mesh_gt.vertices = o3d.utility.Vector3dVector(pcd_gt_np_align)

    verts = np.asarray(mesh_gt.vertices)
    tris = np.asarray(mesh_gt.triangles)

    # frame_w = draw_frame()
    # o3d.visualization.draw_geometries([mesh_gt, pcd_i_can, frame_w])
    # o3d.visualization.draw_geometries([mesh_gt, pcd_i_can])

    yy = [inside_triangle_binary(verts[tris[ii]], pcd_i_np) for ii in range(tris.shape[0])]

    seen_tris = np.array(yy)
    # rate = seen_tris.sum() / seen_tris.shape[0]
    # print("Iteration: ", i, "frame index: ", j, "Coverage ", rate)
    # print("Iteration: ", i, "frame index: ", j, "Coverage ", rate)
    return seen_tris, pcd_i_can, mesh_gt


def get_pixel_per_patch(pcd_i, mesh_gt,
                        rot_mat_i=Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
                        rot_mat_gt=Rotation.from_euler('xyz', [0, 0, 0], degrees=True)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcd_i_np = rot_mat_i.apply(pcd2np(pcd_i))
    pcd_i_can = pcd_update_points(pcd_i, pcd_i_np)
    pcd_i_torch = torch.from_numpy(pcd_i_np).to(device)

    # Align ground truth.
    pcd_gt_np = np.asarray(mesh_gt.vertices)
    pcd_gt_np_align = rot_mat_gt.apply(pcd_gt_np)
    mesh_gt.vertices = o3d.utility.Vector3dVector(pcd_gt_np_align)

    verts = np.asarray(mesh_gt.vertices)
    tris = np.asarray(mesh_gt.triangles)

    verts = torch.from_numpy(verts).to(device)
    tris = torch.from_numpy(tris).long().to(device)

    # frame_w = draw_frame()
    # o3d.visualization.draw_geometries([mesh_gt, pcd_i_can, frame_w])
    # o3d.visualization.draw_geometries([mesh_gt, pcd_i_can])

    seen_tris = np.array([inside_triangle_count(verts[tris[ii]], pcd_i_torch) for ii in range(tris.shape[0])])
    return seen_tris, pcd_i_can, mesh_gt


def reconstruction_from_frames(fname='cv_imgs_01', niters=10, frames=300, post_name='tspn'):
    save_path = ROOT_DIR + "/archive/airsim/" + fname
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save Intrinsic matrix.
    h, w = 576, 1024
    fov = np.deg2rad(45)
    f = w / 2 / np.tan(fov / 2)
    cx, cy = w / 2, h / 2
    o3d_K = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, cx, cy)

    for i in range(niters):
        pcd_sum = o3d.geometry.PointCloud()
        pcd_list, rgb_list = [], []
        # frame_list = []
        # w_frame = draw_frame(np.array([-40, 0, -8]) + np.array([1, 0, 0]), [0, 0, 0, 1])

        for j in range(frames):
            filename = save_path + '{}_' + '{:03d}_{:03d}'.format(i, j)
            if not os.path.exists(filename.format('/pose') + '.npy'):
                break

            print("Start merging frame trajectory index %i, frame %i" % (i, j))
            cpose = np.load(filename.format('/pose') + '.npy')
            depth = np.load(filename.format('/depth') + '.npy')
            color = np.array(Image.open(filename.format('/rgb') + '.png'))
            mask_img = np.array(Image.open(filename.format('/mask') + '.png'))

            camera_mat = np.array([[0, 0, 1, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]])

            # frame_list.append(draw_frame(cpose[:3], cpose[3:]))

            rot_mat = Rotation.from_quat(cpose[3:]).as_matrix()
            Trans = np.hstack((rot_mat, cpose[:3].reshape(3, -1)))
            Trans_mat = np.vstack((Trans, np.array([0, 0, 0, 1])))

            # Classical methods.
            xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
            uv_h = np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
            uv_3d = np.linalg.inv(o3d_K.intrinsic_matrix) @ uv_h
            rays_d = uv_3d / uv_3d[-1, :]
            pcd_c = rays_d.T * np.tile(depth.reshape(-1, 1), (1, 3))

            pcd_w = (Trans_mat @ camera_mat @ np.hstack((pcd_c, np.ones_like(pcd_c[:, 0]).reshape(-1, 1))).T).T

            mask = (mask_img > 0).reshape(-1, 3)
            pcd_actor_w = pcd_w[mask[:, 0], :3]
            color_actor = color.reshape(-1, 3)[mask[:, 0]] / 255
            rgb_list.append(color_actor)
            pcd_list.append(pcd_actor_w)

        pcd_sum.points = o3d.utility.Vector3dVector(np.vstack(pcd_list))
        pcd_sum.colors = o3d.utility.Vector3dVector(np.vstack(rgb_list))

        frame_w = draw_frame()
        pcd_downsample = pcd_sum.voxel_down_sample(voxel_size=0.01)
        # o3d.visualization.draw_geometries([pcd_downsample, frame_w] + frame_list)

        pose_actor = np.load(save_path + '/apose_{:03d}.npy'.format(i))
        trans_actor = pose2mat(pose_actor)
        pcd_can_torch = w2canonical(torch.from_numpy(pcd2np(pcd_downsample).reshape(-1, 3)),
                                    torch.from_numpy(trans_actor))
        pcd_can = pcd_update_points(pcd_downsample, pcd_can_torch.numpy())
        # o3d.visualization.draw_geometries([pcd_can, frame_w])

        o3d.io.write_point_cloud(save_path + '/pcd_{:03d}_'.format(i) + post_name + '.pcd', pcd_can)


def reconstruction_from_multi_frames(fname='cv_imgs', niters=1, n_drones=1, frames=300, post_name='tspn',
                                     if_cast2can=True, if_save2mesh=False):
    save_path = ROOT_DIR + "/archive/airsim/" + fname
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save Intrinsic matrix.
    h, w = 576, 1024
    fov = np.deg2rad(45)
    f = w / 2 / np.tan(fov / 2)
    cx, cy = w / 2, h / 2
    o3d_K = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, cx, cy)

    for i in range(niters):
        pcd_sum = o3d.geometry.PointCloud()
        pcd_list, rgb_list = [], []
        # frame_list = []
        # w_frame = draw_frame(np.array([-40, 0, -8]) + np.array([1, 0, 0]), [0, 0, 0, 1])

        for j in range(frames):

            for k in range(n_drones):
                filename = save_path + '{}_' + '{:03d}_{:03d}_{:03d}'.format(i, j, k)
                if not os.path.exists(filename.format('/pose') + '.npy'):
                    break

                print("Start merging frame trajectory index %i, frame %i, drone %i" % (i, j, k))
                cpose = np.load(filename.format('/pose') + '.npy')
                depth = np.load(filename.format('/depth') + '.npy')
                color = np.array(Image.open(filename.format('/rgb') + '.png'))
                mask_img = np.array(Image.open(filename.format('/mask') + '.png'))

                camera_mat = np.array([[0, 0, 1, 0],
                                       [1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1]])

                # frame_list.append(draw_frame(cpose[:3], cpose[3:]))

                rot_mat = Rotation.from_quat(cpose[3:]).as_matrix()
                Trans = np.hstack((rot_mat, cpose[:3].reshape(3, -1)))
                Trans_mat = np.vstack((Trans, np.array([0, 0, 0, 1])))

                # Classical methods.
                xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
                uv_h = np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
                uv_3d = np.linalg.inv(o3d_K.intrinsic_matrix) @ uv_h
                rays_d = uv_3d / uv_3d[-1, :]
                pcd_c = rays_d.T * np.tile(depth.reshape(-1, 1), (1, 3))

                pcd_w = (Trans_mat @ camera_mat @ np.hstack((pcd_c, np.ones_like(pcd_c[:, 0]).reshape(-1, 1))).T).T

                mask = (mask_img > 0).reshape(-1, 3)
                pcd_actor_w = pcd_w[mask[:, 0], :3]
                color_actor = color.reshape(-1, 3)[mask[:, 0]] / 255
                rgb_list.append(color_actor)
                pcd_list.append(pcd_actor_w)

        pcd_sum.points = o3d.utility.Vector3dVector(np.vstack(pcd_list))
        pcd_sum.colors = o3d.utility.Vector3dVector(np.vstack(rgb_list))

        frame_w = draw_frame()
        pcd_downsample = pcd_sum.voxel_down_sample(voxel_size=0.01)
        # o3d.visualization.draw_geometries([pcd_downsample, frame_w] + frame_list)

        if if_cast2can:
            pose_actor = np.load(save_path + '/apose_{:03d}.npy'.format(i))
            trans_actor = pose2mat(pose_actor)
            pcd_can_torch = w2canonical(torch.from_numpy(pcd2np(pcd_downsample).reshape(-1, 3)),
                                        torch.from_numpy(trans_actor))
            pcd_can = pcd_update_points(pcd_downsample, pcd_can_torch.numpy())
        else:
            pcd_can = pcd_downsample

        # o3d.visualization.draw_geometries([pcd_can, frame_w])
        if not os.path.exists(save_path + '/pcd/'):
            os.mkdir(save_path + '/pcd/')

        pcd_path = save_path + '/pcd/pcd_{:03d}_'.format(i) + post_name + '.pcd'
        print("Write point cloud to address.", pcd_path)
        o3d.io.write_point_cloud(pcd_path, pcd_can)
        o3d.io.write_point_cloud(save_path + '/pcd/pcd_{:03d}_'.format(i) + post_name + '.ply', pcd_can)

        # if if_save2mesh:
        #     radii = [0.015, 0.02, 0.04]
        #     pcd_can.estimate_normals()
        #     rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_can, o3d.utility.DoubleVector(radii))
        #     o3d.visualization.draw_geometries([pcd_can, rec_mesh])
        #     o3d.io.write_triangle_mesh(save_path + '/mesh_{:03d}_'.format(i) + post_name + '.ply', rec_mesh)


if __name__ == '__main__':
    opts = TestArgs()
    args = opts.get_args()

    # reconstruction_from_frames('michelle_tspn', 10, 200, 'tspn')
    # reconstruction_from_multi_frames('michelle_multi_tspn', niters=10, n_drones=3, frames=10, post_name='tspn')
    # reconstruction_from_multi_frames('multiactor_test', niters=1, n_drones=3, frames=100, post_name='tspn',
    #                                  if_cast2can=False, if_save2mesh=True)
    # reconstruction_from_multi_frames('exp_static_v2', niters=1, n_drones=3, frames=200, post_name='tspn',
    #                                  if_cast2can=False, if_save2mesh=True)
    # reconstruction_from_multi_frames('orbit_v1', niters=1, n_drones=3, frames=200, post_name='orbit',
    #                                  if_cast2can=False, if_save2mesh=True)
    reconstruction_from_multi_frames(args.fname, niters=1, n_drones=3, frames=200, post_name=args.post_name,
                                     if_cast2can=False, if_save2mesh=True)
