import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from src.utils.env_utils import spherical_to_cartesian


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def set_axes_full(ax, range, x_off=0, y_off=0, z_off=0):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    ax.set_xlim3d([-range + x_off, range + x_off])
    ax.set_ylim3d([-range + y_off, range + y_off])
    ax.set_zlim3d([0 + z_off, 2 * range + z_off])


def draw_cuboid(ax, T=torch.eye(4), scale=1.0):
    # draw square window and lines connecting it to camera center
    draw_actors = torch.tensor([[0.25, -0.5, 2],
                                [0.25, -0.5, 0],
                                [0.25, 0.5, 0],
                                [0.25, 0.5, 2],
                                [-0.25, -0.5, 2],
                                [-0.25, -0.5, 0],
                                [-0.25, 0.5, 0],
                                [-0.25, 0.5, 2]]).T * scale
    draw_actors_h = torch.vstack((draw_actors, torch.ones_like(draw_actors[0, :])))
    draw_actors_w = (T @ draw_actors_h)[:3]

    # draw_actors = draw_actors + center
    draw_actors_front = draw_actors_w[:, :4]
    draw_actors_back = draw_actors_w[:, 4:]

    draw_actors_front = torch.hstack((draw_actors_front, draw_actors_front[:, :1])).numpy()
    ax.plot(draw_actors_front[0, :], draw_actors_front[1, :], draw_actors_front[2, :], 'k-')

    draw_actors_back = torch.hstack((draw_actors_back, draw_actors_back[:, :1])).numpy()
    ax.plot(draw_actors_back[0, :], draw_actors_back[1, :], draw_actors_back[2, :], 'k-')

    for i in range(4):
        ax.plot(draw_actors_w[0, [i, i + 4]].numpy(), draw_actors_w[1, [i, i + 4]].numpy(),
                draw_actors_w[2, [i, i + 4]].numpy(), 'k-')


def draw_rectangle(ax):
    """Draw rectangle as the actor."""
    draw_actors = torch.tensor([[-0.5, 0, 2],
                                [-0.5, 0, 0],
                                [0.5, 0, 0],
                                [0.5, 0, 2]]).T
    draw_actors_vec = torch.tensor([[0, 1, 0],
                                    [0, 1, 0],
                                    [0, 1, 0],
                                    [0, 1, 0]])
    # draw rectangle as the actor
    draw_actors = torch.cat((draw_actors, draw_actors[:, :1]), dim=1).numpy()
    ax.plot(draw_actors[0, :], draw_actors[1, :], draw_actors[2, :], 'k-')

    # ax.plot(draw_actors_vec[0, :], draw_actors_vec[1, :], draw_actors_vec[2, :], 'r-')


def draw_axis(ax, scale=1):
    ax.plot([0, scale], [0, 0], [0, 0], 'r-')
    ax.plot([0, 0], [0, scale], [0, 0], 'g-')
    ax.plot([0, 0], [0, 0], [0, scale], 'b-')


def draw_paths(ax, path, l_style='k-', marker='o', c='b'):
    ax.plot(path[:, 0], path[:, 1], path[:, 2], l_style)
    ax.scatter(xs=path[:, 0], ys=path[:, 1], zs=path[:, 2], marker=marker, c=c)


def draw_configs(configs, fname, actor_model='cuboid'):
    fig = plt.figure()

    # draw coordinate system of camera
    ax = fig.add_subplot(projection='3d')

    draw_axis(ax)

    if actor_model == 'cuboid':
        draw_cuboid(ax)
    elif actor_model == 'rectangle':
        draw_rectangle(ax)
    else:
        raise Exception('Wrong actor model')

    # Draw actor path.
    path1_s, path2_s = configs[:, :3].reshape(-1, 3), configs[:, 3:].reshape(-1, 3)
    path1_c, path2_c = spherical_to_cartesian(path1_s), spherical_to_cartesian(path2_s)
    draw_paths(ax, path1_c.numpy(), 'b--', marker='o', c='b')
    draw_paths(ax, path2_c.numpy(), 'r--', marker='^', c='r')

    pos_drone1, pos_drone2 = configs[0, :3].reshape(-1, 3), configs[0, 3:].reshape(-1, 3)
    pos_drone1, pos_drone2 = spherical_to_cartesian(pos_drone1), spherical_to_cartesian(pos_drone2)
    ax.scatter(xs=pos_drone1[:, 0], ys=pos_drone1[:, 1], zs=pos_drone1[:, 2], marker='o', c='y', s=100)
    ax.scatter(xs=pos_drone2[:, 0], ys=pos_drone2[:, 1], zs=pos_drone2[:, 2], marker='^', c='y', s=100)
    set_axes_equal(ax)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    fig.tight_layout()
    plt.grid()
    plt.savefig(fname)
    plt.show()


def draw_rewards(traj_rets, test_rets, save_path):
    fig, ax = plt.subplots()
    ax.set_title('Rewards w.r.t. the epoch. ')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rewards')
    ax.grid(True)

    x_epochs = np.linspace(0, len(traj_rets), len(test_rets))
    line_loss, = ax.plot(traj_rets, '-b', label='train', linewidth=2)
    line_test, = ax.plot(x_epochs, test_rets, '-r', label='test', linewidth=2)
    plt.legend([line_loss, line_test], ['train', 'test'])
    plt.savefig(save_path)
    return


def draw_values(r_dp, r_rl, save_path):
    fig, ax = plt.subplots()
    ax.set_title('Rewards w.r.t. each step in one episodes. ')
    ax.set_xlabel('step')
    ax.set_ylabel('Rewards')
    ax.grid(True)

    # x_epochs = np.linspace(0, len(r_dp), len(r_rl))
    line_rl, = ax.plot(r_rl, '-b', label='rl', linewidth=2)
    line_dp, = ax.plot(r_dp, '-r', label='dp', linewidth=2)
    plt.legend([line_rl, line_dp], ['rl', 'dp'])
    plt.savefig(save_path)
    plt.show()
    return


def draw_app_values(pos_mesh, pos_drone, app_mesh, fname=None, render=False):
    """
    Draw APP values assuming the 2nd drone is fixed.
    :param: pos_mesh: position mesh in configuration space with shape ((n_theta * n_gamma * n_r), 3)
            app_mesh: app value for each configs with shape ((n_theta * n_gamma * n_r), 1)
    """
    fig = plt.figure()

    # draw coordinate system of camera
    ax = plt.gca(projection='3d')

    draw_axis(ax)
    draw_cuboid(ax)

    # Draw actor path.
    pos_mesh_c = spherical_to_cartesian(pos_mesh.reshape(-1, 3))
    pos_drone_c = spherical_to_cartesian(pos_drone.reshape(-1, 3))
    ax.scatter(xs=pos_mesh_c[:, 0].numpy(), ys=pos_mesh_c[:, 1].numpy(), zs=pos_mesh_c[:, 2].numpy(),
               marker='o', c=10000 * app_mesh ** 2, s=10000 * app_mesh ** 2)
    ax.scatter(xs=pos_drone_c[:, 0].numpy(), ys=pos_drone_c[:, 1].numpy(), zs=pos_drone_c[:, 2].numpy(),
               marker='^', c='r', s=100)
    set_axes_equal(ax)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    fig.tight_layout()
    plt.grid()
    if fname is not None:
        plt.savefig(fname)
        plt.close(fig)
    if render:
        plt.show()


def vis_tspn_neighbors(pos_p_c, ori_p_c, safe_rad, ppa_thr, ax, step_size_r=10, step_size_t=1, full_ws=20, f=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    theta_range = np.linspace(-180 + step_size_r, 180, int(360 / step_size_r))
    phi_range = np.linspace(0, 90, int(180 / (2 * step_size_r)) + 1)
    r_range = np.linspace(safe_rad, full_ws / 2, int((full_ws / 2 - safe_rad) / step_size_t) + 1)
    theta_mesh, phi_mesh, r_mesh = np.meshgrid(theta_range, phi_range, r_range)
    grid_sph = torch.from_numpy(np.vstack((theta_mesh.reshape(-1), phi_mesh.reshape(-1), r_mesh.reshape(-1))).T)

    pos_q_c = (spherical_to_cartesian(grid_sph)).to(device).float()
    pos_p_c, ori_p_c = pos_p_c.to(device), ori_p_c.to(device).float()
    pos_q_c = pos_q_c + pos_p_c  # Transform to world frame

    vecp2q = pos_q_c - pos_p_c
    dp2q = torch.norm(vecp2q, dim=1)
    cos_theta_q12norm = torch.div((vecp2q @ ori_p_c.T), (dp2q * torch.norm(ori_p_c)))
    ppa_mesh = cos_theta_q12norm / dp2q

    valid_ppa_idx = (ppa_mesh > ppa_thr).to(device)
    valid_safe_idx = (dp2q > safe_rad).to(device)
    valid_idx = torch.logical_and(valid_ppa_idx, valid_safe_idx)

    valid_mesh_pos = pos_q_c[valid_idx].cpu().numpy()
    valid_mesh_ppa = ppa_mesh[valid_idx].cpu().numpy()
    ax.scatter(xs=valid_mesh_pos[:, 0], ys=valid_mesh_pos[:, 1], zs=valid_mesh_pos[:, 2],
               marker='o', c=10000 * valid_mesh_ppa ** 2, s=1000 * valid_mesh_ppa ** 2)
    return


def vis_safe_region(center, safe_rad, ax, step_size_r=10, step_size_t=1):
    theta_range = np.linspace(-180 + step_size_r, 180, int(360 / step_size_r))
    phi_range = np.linspace(0, 90, int(180 / (2 * step_size_r)) + 1)
    r_range = np.linspace(0, safe_rad, int(safe_rad / step_size_t) + 1)
    theta_mesh, phi_mesh, r_mesh = np.meshgrid(theta_range, phi_range, r_range)
    grid_sph = torch.from_numpy(np.vstack((theta_mesh.reshape(-1), phi_mesh.reshape(-1), r_mesh.reshape(-1))).T)

    pos_q_c = (spherical_to_cartesian(grid_sph)) + center
    ax.scatter(xs=pos_q_c[:, 0], ys=pos_q_c[:, 1], zs=pos_q_c[:, 2], marker='o',
               c='r', s=0.5 * np.ones_like(pos_q_c[:, 0]))
    return
