import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def spherical_to_cartesian(pos_sph):
    """
    Transform coordination representation from spherical to Cartesian.
    :param pos_sph: Position in spherical representation (Degree: from -170 to 180).
    :return: Position in cartesian representation. Shape 3 * 1.
    """
    theta, phi, r = pos_sph[:, 0], pos_sph[:, 1], pos_sph[:, 2]
    r_xy = r * torch.sin(phi * np.pi / 180.0)
    return torch.stack([torch.cos(theta * np.pi / 180.0) * r_xy,
                        torch.sin(theta * np.pi / 180.0) * r_xy,
                        torch.cos(phi * np.pi / 180.0) * r]).T


def cartesian_to_spherical_rad(pos_cart):
    """
    Transform coordination representation from cartesian to spherical.
    :param pos_cart: Position in cartesian representation.
    :return: Position in spherical representation (Rad: from -pi to pi).
    """
    x, y, z = pos_cart[:, 0], pos_cart[:, 1], pos_cart[:, 2]
    r_xy = torch.sqrt(x ** 2 + y ** 2)
    r = torch.sqrt(r_xy ** 2 + z ** 2)
    theta = torch.atan2(y, x)
    phi = torch.atan2(r_xy, z)
    return torch.stack([theta, phi, r]).T


def cartesian_to_spherical_deg(pos_cart):
    """
    Transform coordination representation from cartesian to spherical.
    :param pos_cart: position in cartesian coordinate with shape (-1, 3)
    :return: position in spherical coordinate with shape (-1, 3)
    """
    return config_rad2deg(cartesian_to_spherical_rad(pos_cart).reshape(-1, 3))


def oripos2T(ori, pos):
    Trans = torch.hstack((torch.from_numpy(ori),
                          torch.from_numpy(pos.reshape(3, -1))))
    Trans_can2w = torch.vstack((Trans, torch.tensor([0, 0, 0, 1])))
    return Trans_can2w


def w2canonical(w_pos, trans_can2w):
    """
    Coordinate transformation from world frame to canonical frame.
    Args:
        w_pos: position (R^3) in world frame.
        trans_can2w: Transformation matrix from canonical frame to world frame. (R^(4*4)).

    Returns:
        position (R^3) in canonical frame.
    """
    # w_pos_h = torch.cat((w_pos, torch.tensor([1.0])))
    w_pos_h = torch.hstack((w_pos, torch.ones_like(w_pos[:, 0]).reshape(-1, 1))).T
    can_pos_h = torch.inverse(trans_can2w).float() @ w_pos_h.float()
    return can_pos_h[:3].T


def canonical2w(can_pos, trans_can2w):
    """
    Coordinate transformation from canonical frame to world frame.
    Args:
        can_pos: position (R^3) in canonical frame.
        trans_can2w: Transformation matrix from canonical frame to world frame. (R^(4*4)).

    Returns:
        position (R^3) in world frame.
    """
    can_pos_h = torch.hstack((can_pos, torch.ones_like(can_pos[:, 0].reshape(-1, 1)))).T
    w_pos_h = trans_can2w.float() @ can_pos_h.float()
    return w_pos_h[:3].T


def pose2mat(pose):
    pos = pose[:3]
    quat = pose[3:]

    r = R.from_quat(quat)
    rot_mat = r.as_matrix()
    trans = np.hstack((rot_mat, pos.reshape(-1, 1)))
    return np.vstack((trans, np.array([0, 0, 0, 1])))


def mat2pose(mat):
    pos = mat[:3, 3]
    quat = R.from_matrix(mat[:3, :3]).as_quat()
    return np.concatenate((pos, quat))


def config_rad2deg(pos_sph_rad):
    """
    Transform spherical coordination representation from rad to degree.
    :param pos_sph_rad: Position in rad representation. Shape N * 3. [theta, phi, r]
    :return: Position in degree representation. Shape N * 3. [theta, phi, r]
    """
    theta_rad, phi_rad, r = pos_sph_rad[:, 0], pos_sph_rad[:, 1], pos_sph_rad[:, 2]
    theta_deg = theta_rad / np.pi * 180.0
    phi_deg = phi_rad / np.pi * 180.0
    return torch.stack((theta_deg, phi_deg, r), dim=1)


def config_deg2rad(pos_sph_deg):
    """
    Transform spherical coordination representation from rad to degree.
    :param pos_sph_deg: Position in rad representation. Shape N * 3. [theta, phi, r]
    :return: Position in degree representation. Shape N * 3. [theta, phi, r]
    """
    theta_deg, phi_deg, r = pos_sph_deg[:, 0], pos_sph_deg[:, 1], pos_sph_deg[:, 2]
    theta_rad = theta_deg / 180.0 * np.pi
    phi_rad = phi_deg / 180.0 * np.pi
    return torch.stack((theta_rad, phi_rad, r), dim=1)


def get_2d_ori_vector(ori_actor):
    ori_actor = ori_actor.reshape(-1, 1)
    vec = torch.hstack([torch.cos(ori_actor), torch.sin(ori_actor)])
    return vec


def _get_distance(p1, p2):
    assert (p1.shape == p2.shape), "p1 and p2 have different shape."
    return torch.norm(p1 - p2)


def split_configs_by_steps(configs, step_size):
    """
    Split series of configurations by a step size.
    Args:
        configs: N * (3 * n) configurations in spherical space, represented by degree format.
        step_size: Euclidean distance that the drones are constrained by each time step.

    Returns:
        configs_cont: continuous configurations that each movement is constrained within the step size.
    """
    n_configs, n_drone = configs.shape
    n_drone = int(n_drone / 3)

    configs_cont_list, configs_len_list = [], []
    for i_drone in range(n_drone):
        configs_i_s = configs[:, 3 * i_drone:3 * i_drone + 3]
        configs_i_c = spherical_to_cartesian(configs_i_s)
        configs_cont_i = configs_i_c[0].reshape(-1, 3)
        for j_config in range(1, n_configs):
            config_next = configs_i_c[j_config].reshape(-1, 3)
            config_init = configs_cont_i[-1].reshape(-1, 3)
            while torch.norm(config_init - config_next) > step_size:
                # delta = config_next - config_init
                config_init = (config_next - config_init) / torch.norm(config_next - config_init) * step_size + config_init
                configs_cont_i = torch.vstack((configs_cont_i, config_init.reshape(-1, 3)))
            configs_cont_i = torch.vstack((configs_cont_i, config_next))
        configs_cont_list.append(configs_cont_i)
        configs_len_list.append(len(configs_cont_i))

    max_config_len = np.max(configs_len_list)
    configs_cont = torch.zeros((max_config_len, 3 * n_drone))
    for i_drone in range(n_drone):
        configs_len = configs_len_list[i_drone]
        configs_cont[:configs_len, 3*i_drone:3*i_drone + 3] = configs_cont_list
        configs_cont[configs_len:, 3*i_drone:3*i_drone + 3] = torch.tile(configs_cont_list[-1].reshape(-1, 3),
                                                                         (max_config_len - configs_len, 1))
    return configs_cont


def get_app_value(pos_q1_c, pos_q2_c, pos_p_c, ori_p_c,
                 f=0.1, ang_thr=torch.tensor(np.deg2rad(90)), safe_off=4, safe_rad=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ang_thr = ang_thr.to(device)

    pos_q1_c, pos_q2_c = pos_q1_c.to(device), pos_q2_c.to(device)
    pos_p_c, ori_p_c = pos_p_c.to(device), ori_p_c.to(device)

    vecp2q1 = pos_q1_c - pos_p_c
    vecp2q2 = pos_q2_c - pos_p_c

    dp2q1, dp2q2 = torch.norm(vecp2q1, dim=1), torch.norm(vecp2q2, dim=1)
    cos_theta_q12norm = torch.div((vecp2q1 @ ori_p_c.T), (dp2q1 * torch.norm(ori_p_c)))
    cos_theta_q22norm = torch.div((vecp2q2 @ ori_p_c.T), (dp2q2 * torch.norm(ori_p_c)))

    app_q1 = cos_theta_q12norm / (dp2q1 + f) ** 2
    app_q2 = cos_theta_q22norm / (dp2q2 + f) ** 2
    app = torch.max(app_q1, app_q2)

    invisible_idx = torch.logical_and(cos_theta_q12norm < torch.cos(ang_thr),
                                      cos_theta_q22norm < torch.cos(ang_thr)).to(device)
    invalid_dist_idx = torch.logical_or(dp2q1 < safe_rad - safe_off, dp2q2 < safe_rad - safe_off).to(device)
    invalid_idx = torch.logical_or(invisible_idx, invalid_dist_idx)

    const = app.max()
    app[invalid_idx] = torch.zeros(invalid_idx.sum()).to(device) * const
    return app


def setup_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + '/depth'):
        os.makedirs(save_path + '/depth')
    if not os.path.exists(save_path + '/seg'):
        os.makedirs(save_path + '/seg')
    if not os.path.exists(save_path + '/rgb'):
        os.makedirs(save_path + '/rgb')
    if not os.path.exists(save_path + '/mask'):
        os.makedirs(save_path + '/mask')
    if not os.path.exists(save_path + '/pcd'):
        os.makedirs(save_path + '/pcd')
    if not os.path.exists(save_path + '/vis'):
        os.makedirs(save_path + '/vis')
    if not os.path.exists(save_path + '/gts'):
        os.makedirs(save_path + '/gts')
    if not os.path.exists(save_path + '/pose'):
        os.makedirs(save_path + '/pose')

def vec2hom(vec):
    return np.concatenate((vec, np.array([1])))


def hom2vec(vec):
    return vec[:-1]


def vec2hom_batch(vec):
    return np.hstack((vec, np.ones_like(vec[:, 0]).reshape(-1, 1)))


def hom2vec_batch(vec):
    return vec[:, :-1]
