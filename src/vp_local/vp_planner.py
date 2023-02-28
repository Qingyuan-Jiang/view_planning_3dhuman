import numpy as np
import open3d as o3d

from src.utils.pcd_utils import np2pcd_xyzrgb
from src.utils.env_utils import pose2mat, mat2pose


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


def safe_cons(x, param_pos_actors, param_r):
    """
    Safe Distance constraints. Drones have to maintain safe distance between the actor.
    Args:
        x: R^(n * 3). Visiting points for each patch.
        param_pos_actors: R^(n * 3). Position for each patch.
        param_r: R. Safe radius for each patch.

    Returns:

    """
    num_actors, num_dim = param_pos_actors.shape
    x = x.reshape(-1, num_dim)
    return np.linalg.norm(x - param_pos_actors, axis=1) - param_r


def ppa_jacob_pos(pos_drone, ori_drone, pos_actor, ori_actor):
    n_patch, n_dim = pos_actor.shape
    pos_drone = np.tile(pos_drone.reshape(-1, 3), (n_patch, 1))
    ori_drone = np.tile(ori_drone.reshape(-1, 3), (n_patch, 1))
    ori_actor = (ori_actor.T / np.linalg.norm(ori_actor.reshape(-1, 3), axis=1)).T
    vec_a2d = pos_drone - pos_actor
    dist_a2d = np.linalg.norm(vec_a2d, axis=1).reshape(-1, 1)

    # vec_jac = (ori_actor * (dist_a2d ** 2) - np.sum(vec_a2d * ori_actor, axis=1)
    # .reshape(-1, 1) * 2 * vec_a2d) / dist_a2d ** 4
    dot = np.sum(ori_actor * ori_drone, axis=1)
    vec_jac = - dot.T * vec_a2d.T / dist_a2d.T ** 3
    return np.sum(vec_jac.T, axis=0)


def vec_decompose(vec, u):
    u = u / np.linalg.norm(u)
    v = vec - vec @ u / (np.linalg.norm(vec) * np.linalg.norm(u)) * u
    return v / np.linalg.norm(v)


class VPLocalPlanner:
    def __init__(self, pos_inits, safe_rad, step_rad):
        self.pos_inits = pos_inits
        self.cpose_list = []
        self.ppa_list = []
        self.ppa_hist = -np.inf
        self.safe_rad = np.asarray(safe_rad)
        self.step_rad = np.asarray(step_rad)
        self.alpha = 0.05

    def vp_optimizer(self, pos_actors, normals, pos_inits, ori_inits, optim_size=0.02):
        """
        Optimize the visiting points for each patch.
        Args:
            mesh: R^(n * 3). Position of patches.
            pos_inits:
            optim_size:

        Returns:
            sol_dict: Dictionary for solutions.
        """
        path = []

        vec_jacob = ppa_jacob_pos(pos_inits, ori_inits, pos_actors, normals)
        vec_jacob = vec_jacob / np.linalg.norm(vec_jacob)

        step_size = self.step_rad
        pos_test = pos_inits + vec_jacob * step_size
        direction = (compute_ppas(pos_test, pos_actors, normals) - compute_ppas(pos_inits, pos_actors, normals)) > 0

        pos_update = pos_inits - (-1) ** direction * vec_jacob * optim_size
        path.append(pos_update)

        n_steps = int(self.step_rad // optim_size)
        for i in range(n_steps):
            if (safe_cons(pos_update, pos_actors, self.safe_rad) < 0).any():
                vec_jacob = ppa_jacob_pos(pos_update, ori_inits, pos_actors, normals)
                vec_jacob = vec_jacob / np.linalg.norm(vec_jacob)
                vec_jacob_comp = vec_decompose(vec_jacob, np.mean(pos_actors, axis=0) - pos_update)

                pos_test = pos_inits + vec_jacob * step_size
                direction = (compute_ppas(pos_test, pos_actors, normals) - compute_ppas(pos_inits, pos_actors,
                                                                                        normals)) > 0

                pos_update = pos_update - (-1) ** direction * vec_jacob_comp * optim_size
                path.append(pos_update)
            else:
                vec_jacob = ppa_jacob_pos(pos_update, ori_inits, pos_actors, normals)
                vec_jacob = vec_jacob / np.linalg.norm(vec_jacob)

                pos_test = pos_inits + vec_jacob * step_size
                direction = (compute_ppas(pos_test, pos_actors, normals) - compute_ppas(pos_inits, pos_actors, normals)) > 0

                pos_update = pos_update - (-1) ** direction * vec_jacob * optim_size
                path.append(pos_update)

        # for j in range(n_steps - 1 - i):
        #     vec_jacob = ppa_jacob_pos(pos_update, ori_inits, pos_actors, normals)
        #     vec_jacob = vec_jacob / np.linalg.norm(vec_jacob)
        #     vec_jacob_comp = vec_decompose(vec_jacob, np.mean(pos_actors, axis=0) - pos_update)
        #
        #     pos_test = pos_inits + vec_jacob * step_size
        #     direction = (compute_ppas(pos_test, pos_actors, normals) - compute_ppas(pos_inits, pos_actors, normals)) > 0
        #
        #     pos_update = pos_update - (-1) ** direction * vec_jacob_comp * optim_size
        #     path.append(pos_update)

        pos_output = pos_update.reshape(-1, 3)
        ppas = compute_ppas(pos_update, pos_actors, normals)
        return pos_output, ppas, path

    def vp_local_mesh(self, pos_patches, normals, cpose_inits):
        cpose = np.asarray(cpose_inits)
        pos_drone = cpose[:3]
        ori_drone = pose2mat(cpose) @ np.array([[0, 0, 1, 1]]).T
        ori_drone = ori_drone[:3]
        pos_drone, ppas, path = self.vp_optimizer(pos_patches, normals, pos_drone, ori_drone)
        return pos_drone, ppas, path
