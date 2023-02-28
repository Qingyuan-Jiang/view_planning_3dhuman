import numpy as np
import torch
from tsp_solver.greedy_numpy import solve_tsp
from scipy.spatial.transform import Rotation
from src.utils.env_utils import spherical_to_cartesian
from scipy import optimize


def get_tsp_path(pos_targets, pos_start=None):
    num_cand, num_dim = pos_targets.shape
    tsp_cand = torch.vstack((pos_start, pos_targets)) if pos_start is not None else pos_targets
    num_cand = len(tsp_cand)
    dist_matrix = np.zeros((num_cand, num_cand))
    for i in range(num_cand):
        for j in range(i):
            cand1, cand2 = tsp_cand[i], tsp_cand[j]
            dist_matrix[i, j] = torch.norm(cand1 - cand2)

    path = solve_tsp(dist_matrix, endpoints=(0, None))
    configs = tsp_cand[path].reshape(-1, num_dim)
    return path, tsp_cand, configs


def get_tsp_length(pos_traj, pos_start=None):
    """
    Get sum of Euclidean distance along the trajectory.
    Args:
        pos_traj: N*d array. N, the number of points on the trajectory. d, the Euclidean dimension.

    Returns:
        d_sum: summed Euclidean distance.
    """
    pos_traj_whole = torch.vstack((pos_start, pos_traj)) if pos_start is not None else pos_traj
    vec = pos_traj_whole[1:, :] - pos_traj_whole[:-1, :]
    d_sum = torch.sum(torch.norm(vec, dim=1), dim=0)
    return d_sum


def get_tsp_cons(pos_traj, pos_centers, r):
    """
    Obtain the TSP-disk constraints given visiting points and the center of disks.
    Args:
        pos_traj: The position of current visiting points for each disk. N * d array.
        pos_centers: The position of each disk's center. N * d array.
        r: radius of the disks. Constant.

    Returns:
        vec_gx: N * 1 array. each stands for the || x_j - r ||_2 - r
    """
    vec = pos_traj - pos_centers
    vec_gx = torch.norm(vec, dim=1) - r
    return vec_gx


def get_ppa(pos_drone, pos_actor, ori_actor,
            f=0.1, ang_thr=torch.tensor(np.deg2rad(90)), safe_off=4, safe_rad=5, app_thresh=0.12):
    """
    Calculate 2D APP values given position of actor and orientation of actor.
    Args:
        pos_drone: position of drones in Cartesian space.
        pos_actor: position of the actor in Cartesian space.
        ori_actor: orientation of the actor in Cartesian space.

    Returns:

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ang_thr = ang_thr.to(device)

    pos_drone = pos_drone.to(device)
    pos_actor, ori_actor = pos_actor.to(device), ori_actor.to(device)

    vecp2q1 = pos_drone - pos_actor

    dp2q1 = torch.norm(vecp2q1, dim=1).reshape(-1, 1)
    cos_theta_q12norm = torch.div((vecp2q1 @ ori_actor.T).reshape(-1, 1),
                                  (dp2q1 * torch.norm(ori_actor)).reshape(-1, 1))
    app = torch.div(cos_theta_q12norm, (dp2q1 + f))

    invisible_idx = (cos_theta_q12norm < torch.cos(ang_thr)).to(device)
    invalid_dist_idx = (dp2q1 < safe_rad - safe_off).to(device)
    invalid_idx = torch.logical_or(invisible_idx, invalid_dist_idx)

    app[invalid_idx] = torch.zeros(invalid_idx.sum()).to(device)
    # print(app.max(), app.max() * 0.75, app.max() * 0.8)
    valid_idx = (app > app_thresh).cpu().numpy()
    return app, valid_idx.flatten()


def dist_sphere(pos1_c, pos2_c, r_min=0.0, origin=torch.zeros(3)):
    """
    Calculate the distance on the sphere outside of safe space.
    Args:
        pos1_c: R^(n * 3). Cartesian coordinates for the 1st position.
        pos2_c: R^(n * 3). Cartesian coordinates for the 2nd position.
        r_min: R^1. Minimum radius of inner sphere.
        origin: R^3. Origin point of the actor

    Returns:

    """
    assert len(pos1_c) == len(pos2_c), "Should contain same number of points."
    # print("pos1", pos1_c)
    # print("pos2", pos2_c)

    # Calculate distance between crossing line and the actor.   ----------------
    # Vector from point 1 to point 2, from point 1 to origin.
    # pos1_c, pos2_c = pos1_c.float(), pos2_c.float()
    origin = origin.double()
    vec_12 = (pos2_c - pos1_c).reshape(-1, 3)
    vec_10 = (origin - pos1_c).reshape(-1, 3)
    vec_01 = (pos1_c - origin).reshape(-1, 3)
    vec_02 = (pos2_c - origin).reshape(-1, 3)

    # Distance using cross dot.
    vec_cross = torch.cross(vec_10, vec_12, dim=1)
    dist_l = torch.divide(torch.norm(vec_cross, dim=1), torch.norm(vec_12, dim=1))

    dist12 = torch.norm(vec_12, dim=1)
    dist10 = torch.norm(vec_10, dim=1)

    # Index that distance are greater than safe radius.
    # i.e. shortest path are the segment between them.
    # r_min = torch.tensor(r_min).float()

    cos_theta = torch.divide(torch.sum(vec_12 * vec_10, dim=1), dist12 * dist10)
    lamb = torch.divide(cos_theta * dist10, dist12)

    idx_dist = torch.logical_or(dist_l >= r_min, torch.logical_or(lamb > 1, lamb < 0))
    # print("Segment distance to the center.", dist_l)
    # print("Not intersecting with safe region.", idx_dist)

    # Otherwise, the shortest path are consists of two lines, and an arc.
    # Calculate distance for the two lines.
    dist1 = torch.sqrt(torch.norm(vec_01, dim=1) ** 2 - r_min ** 2)
    dist2 = torch.sqrt(torch.norm(vec_02, dim=1) ** 2 - r_min ** 2)

    # Angle of the middle arc.
    ang_12 = torch.acos(
        torch.divide(torch.sum(vec_01 * vec_02, dim=1), (torch.norm(vec_01, dim=1) * torch.norm(vec_02, dim=1))))
    ang1 = torch.atan2(dist1, r_min)
    ang2 = torch.atan2(dist2, r_min)
    ang_mid = ang_12 - ang1 - ang2

    # Distance for the arc.
    dist_mid = r_min * ang_mid

    # Sum up for those shortest distance are not segments.
    dist_sum = dist1 + dist2 + dist_mid

    dist_ret = dist_sum.detach()
    dist_ret[idx_dist] = dist12[idx_dist].detach()
    return dist_ret


def f_norm(x, param_pos_drone):
    """Objective function with Euclidean distance."""
    num_drones, num_dim = param_pos_drone.shape
    x = x.reshape(-1, num_dim)
    L = np.linalg.norm(x[0] - param_pos_drone)
    L = L + np.sum(np.linalg.norm(x[1:, :] - x[:-1, :], axis=1))
    return L


def f_sphere(x, param_pos_drone, r_min):
    """Objective function with spherical distance."""
    num_drones, num_dim = param_pos_drone.shape
    x = x.reshape(-1, num_dim)
    L = dist_sphere(torch.from_numpy(param_pos_drone.reshape(-1, 3)), torch.from_numpy(x[0].reshape(-1, 3)), r_min)
    L = L + torch.sum(
        dist_sphere(torch.from_numpy(x[:-1, :].reshape(-1, 3)), torch.from_numpy(x[1:, :].reshape(-1, 3)), r_min))
    # print("Optimizing objective function with length %f" % L)
    return L


def app_cons(x, param_pos_actors, param_ori_actors, param_c):
    """
    Area-Per-Pixel constraints.
    Args:
        x: Visiting points for each patch.
        param_pos_actors: R^(n * 3). Position of patches.
        param_ori_actors: R^(n * 3). Orientation of patches.
        param_c: R. Threshold for the APP values.

    Returns:

    """
    num_actors, num_dim = param_pos_actors.shape
    x = x.reshape(-1, num_dim)
    vec = x - param_pos_actors
    cos = np.divide(np.sum(vec * param_ori_actors, axis=1),
                    np.linalg.norm(vec, axis=1) * np.linalg.norm(param_ori_actors, axis=1))
    app = np.divide(cos, np.linalg.norm(vec, axis=1))
    return app - param_c


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


def height_cons(x):
    """
    Height constraints. Drones have to maintain positive height value.
    Args:
        x: R^(n * 3). Visiting points for each patch.

    Returns:

    """
    x = x.reshape(-1, 3)
    return x[:, -1]


def height_neg_cons(x):
    """
    Height constraints. Drones have to maintain positive height value.
    Args:
        x: R^(n * 3). Visiting points for each patch.

    Returns:

    """
    x = x.reshape(-1, 3)
    # return -x[:, -1] - np.ones_like(x[:, -1])
    return x[:, -1] - np.zeros_like(x[:, -1])


def lambda_calculator(v1, v2, theta, r, d1, d2):
    lamb2 = torch.divide(r, d1 * d2 * torch.sin(theta)) * torch.sqrt(d1 ** 2 - r ** 2)
    lamb1 = torch.divide(r ** 2 - lamb2 * d1 * d2 * torch.cos(theta), d1 ** 2)
    v3 = lamb1 * v1 + lamb2 * v2
    return lamb1, lamb2, v3


def seg_separator(v1, v2, t):
    v1, v2 = v1.reshape(-1, 3).float(), v2.reshape(-1, 3).float()
    vec = v2 - v1
    n_seg = int(torch.floor(torch.norm(vec, dim=1) / t))
    line_space = torch.linspace(0, n_seg, n_seg + 1).reshape(1, -1).float()
    vec_unit = torch.divide(vec, torch.norm(vec, dim=1))
    path = t * (vec_unit.T @ line_space) + v1.T
    return path.T


def arc_separator(v1, v2, r, t):
    v1, v2 = v1.reshape(-1, 3), v2.reshape(-1, 3)
    v_axis = torch.cross(v1, v2, dim=1) / torch.norm(torch.cross(v1, v2), dim=1)
    rot_ang = t / r
    rot_func = Rotation.from_rotvec(v_axis * rot_ang)
    v_start = v1.detach()
    vec_list = [v1]
    while torch.acos(v_start @ v2.T / (torch.norm(v_start) * torch.norm(v2))) > rot_ang:
        v_start = torch.from_numpy(rot_func.apply(v_start)).float()
        vec_list.append(v_start)
    vec_list = torch.vstack(vec_list)
    return vec_list


def get_tangent_point(p1, c1, c2, r, binary=False):
    v_c1_p1, v_c1_c2 = p1 - c1, c2 - c1
    v_n1 = torch.cross(v_c1_p1, v_c1_c2, dim=1)
    v_c1_p5 = (torch.cross(v_c1_c2, v_n1, dim=1).T / (torch.norm(torch.cross(v_c1_c2, v_n1, dim=1), dim=1) + 1e-6).T).T
    idx = torch.sum(v_c1_p5 * v_c1_p1, dim=1) < 0
    v_c1_p5[idx] = - v_c1_p5[idx]
    if binary:
        return [v_c1_p5 * r + c1, - v_c1_p5 * r + c1]
    else:
        p5 = v_c1_p5 * r + c1
        return p5


def dist_center(pos1_c, pos2_c, origin):
    # origin = origin.double()
    vec_12 = (pos2_c - pos1_c).reshape(-1, 3)
    vec_10 = (origin - pos1_c).reshape(-1, 3)

    # Distance using cross dot.
    vec_cross = torch.cross(vec_10, vec_12, dim=1)
    dist_l = torch.divide(torch.norm(vec_cross, dim=1), torch.norm(vec_12, dim=1))
    return dist_l


def index_not_intersection(pos1_c, pos2_c, origin, r_min, tol=0.0):
    # origin = origin.double()
    vec_12 = (pos2_c - pos1_c).reshape(-1, 3)
    vec_10 = (origin - pos1_c).reshape(-1, 3)

    dist12 = torch.norm(vec_12, dim=1)
    dist10 = torch.norm(vec_10, dim=1)

    # Distance using cross dot.
    vec_cross = torch.cross(vec_10, vec_12, dim=1)
    dist_l = torch.divide(torch.norm(vec_cross, dim=1), torch.norm(vec_12, dim=1))

    cos_theta = torch.divide(torch.sum(vec_12 * vec_10, dim=1), dist12 * dist10)
    lamb = torch.divide(cos_theta * dist10, dist12)

    idx_not_inside_sphere = torch.logical_and(torch.norm(pos1_c - origin, dim=1) + tol >= r_min,
                                              torch.norm(pos2_c - origin, dim=1) + tol >= r_min)
    idx_seg_not_intersection = torch.logical_or(dist_l + tol >= r_min, torch.logical_or(lamb > 1 - tol, lamb < 0 + tol))
    idx_not_intersection = torch.logical_and(idx_not_inside_sphere, idx_seg_not_intersection)
    return idx_not_intersection


def decompose_vector(base1, base2, vec):
    assert len(base1) == len(base2)
    lambs_list = []
    for i in range(len(base1)):
        A = torch.vstack((base1[i, :2], base2[i, :2])).T
        lambs_i = torch.inverse(A) @ vec[i, :2].T
        lambs_list.append(lambs_i)
    lambs_list = torch.vstack(lambs_list)
    return lambs_list[:, 0].reshape(-1, 1), lambs_list[:, 1].reshape(-1, 1)


def tangent_points_single_sphere(p1, p2, c, r):
    n, _ = p1.shape
    r_n = r.repeat((n, 1))

    vec1 = p1 - c
    vec2 = p2 - c
    # Calculate waypoints. ----------------------------------------
    dist1, dist2 = torch.norm(vec1, dim=1), torch.norm(vec2, dim=1)

    # Solve inside sphere problem.
    if dist1 < r:
        vec1 = vec1 / dist1 * (r + 0.00001)
        dist1 = torch.norm(vec1, dim=1)

    if dist2 < r:
        vec2 = vec2 / dist2 * (r + 0.00001)
        dist2 = torch.norm(vec2, dim=1)

    theta_12 = torch.acos(torch.divide(torch.sum(vec1 * vec2, dim=1), (dist1 * dist2)))

    lamb11, lamb12, vec3 = lambda_calculator(vec1, vec2, theta_12, r_n, dist1, dist2)
    lamb21, lamb22, vec4 = lambda_calculator(vec2, vec1, theta_12, r_n, dist2, dist1)
    return vec3 + c, vec4 + c


def get_common_tangent_full(p1, p2, c1, c2, r_min):
    p5_init = get_tangent_point(p1, c1, c2, r_min, binary=True)
    p6_init = get_tangent_point(p2, c2, c1, r_min, binary=True)
    p_init_idx = [[p5_init[0], p6_init[0]], [p5_init[0], p6_init[1]],
                  [p5_init[1], p6_init[0]], [p5_init[1], p6_init[1]]]
    p_res = []
    for i in range(4):
        p5_init_i, p6_init_i = p_init_idx[i][0], p_init_idx[i][1]
        vec_c1_p1, vec_c2_p2, vec_c1_c2 = p1 - c1, p2 - c2, c2 - c1
        lamb1_init, lamb2_init = decompose_vector(vec_c1_p1, p6_init_i - c1, p5_init_i - c1)
        lamb3_init, lamb4_init = decompose_vector(vec_c2_p2, p5_init_i - c2, p6_init_i - c2)
        x_init = torch.cat((p5_init_i, p6_init_i, lamb1_init, lamb2_init, lamb3_init, lamb4_init), dim=1).flatten().numpy()
        tang_res = optimize.fsolve(f_common_tang, x_init,
                                   args=(vec_c1_p1.numpy(), vec_c2_p2.numpy(), vec_c1_c2.numpy(), r_min.numpy()))

        eqs = f_common_tang(tang_res, vec_c1_p1.numpy(), vec_c2_p2.numpy(), vec_c1_c2.numpy(), r_min.numpy())
        if (np.abs(eqs) > 1e-3).any():
            # print("Failed to find solution. Equations results", eqs)
            pass
        else:
            # print("Find solution. Equations results", eqs)
            points_tang = torch.from_numpy(tang_res).reshape(-1, 10).float()
            # print("Equation solution", points_tang)
            p5, p6 = points_tang[:, :3] + c1, points_tang[:, 3:6] + c2
            if (p5[:, 2] < 0).any() or (p6[:, 2] < 0).any():
                pass
            else:
                assert torch.norm(p5 - c1, dim=1) < r_min + 1e-3, "Not on the sphere"
                assert torch.norm(p5 - c1, dim=1) >= r_min - 1e-3, "Inside the sphere"
                assert torch.norm(p6 - c2, dim=1) < r_min + 1e-3, "Not on the sphere"
                assert torch.norm(p6 - c2, dim=1) >= r_min - 1e-3, "Inside the sphere"
                p_res.append([p5, p6])

    # return p5, p6
    return p_res


def geo_path_type(pos1, pos2, center1, center2, r):
    idx_not_intersect1 = index_not_intersection(pos1, pos2, center1, r)
    idx_not_intersect2 = index_not_intersection(pos1, pos2, center2, r)
    intersect_none = torch.logical_and(idx_not_intersect1, idx_not_intersect2)
    intersect_only1 = torch.logical_and(idx_not_intersect2, torch.logical_not(idx_not_intersect1))
    intersect_only2 = torch.logical_and(idx_not_intersect1, torch.logical_not(idx_not_intersect2))
    intersect_both = torch.logical_not(torch.logical_or(idx_not_intersect1, idx_not_intersect2))
    # print("Intersection situation: First sphere", idx_not_intersect1, "Second sphere", idx_not_intersect2, "Both", intersect_both)
    return intersect_none, intersect_only1, intersect_only2, intersect_both


def geo_path_type_both(p1_both, p2_both, c1_both, c2_both, r):
    # Determine whether the geodesic path only surrounds one sphere.
    p5_c1, p6_c1 = tangent_points_single_sphere(p1_both, p2_both, c1_both, r)
    p5_c2, p6_c2 = tangent_points_single_sphere(p1_both, p2_both, c2_both, r)
    idx_geo_only1 = torch.logical_and(index_not_intersection(p1_both, p5_c1, c2_both, r),
                                      index_not_intersection(p6_c1, p2_both, c2_both, r))
    idx_geo_only2 = torch.logical_and(index_not_intersection(p1_both, p5_c2, c1_both, r),
                                      index_not_intersection(p6_c2, p2_both, c1_both, r))
    if torch.logical_and(idx_geo_only1, idx_geo_only2).any():
        idx_geo_only1 = torch.logical_and(index_not_intersection(p1_both, p5_c1, c2_both, r),
                                          index_not_intersection(p6_c1, p2_both, c2_both, r))
        idx_geo_only2 = torch.logical_and(index_not_intersection(p1_both, p5_c2, c1_both, r),
                                          index_not_intersection(p6_c2, p2_both, c1_both, r))
    assert torch.logical_not(
        torch.logical_and(idx_geo_only1, idx_geo_only2)).any(), "Exception for not intersecting both spheres."
    idx_geo_both = torch.logical_not(torch.logical_or(idx_geo_only1, idx_geo_only2))
    return idx_geo_only1, idx_geo_only2, idx_geo_both


def dist_geodesic(pos1, pos2, center1, center2, r):
    """
    Geodesic distance between two points with different sphere centers.
    Args:
        pos1: n * 3
        pos2: n * 3
        center1: n * 3
        center2: n * 3
        r: scalar

    Returns:

    """
    assert len(pos1) == len(pos2)
    assert len(center1) == len(center2)
    tol = 1e-3

    # First extract pairs that has same center  --------------------------------------------
    dist_c1c2 = torch.norm(center1 - center2, dim=1)
    dist_same_center_idx = dist_c1c2 < 0.1
    dist_diff_center_idx = torch.logical_not(dist_same_center_idx)
    dist_center_apprx = 0.5 * (center1 + center2)

    dist12 = torch.norm(pos1 - pos2, dim=1)
    dist_geo = torch.zeros_like(dist12)
    dist_geo[dist_same_center_idx] = dist_sphere(pos1, pos2, r, dist_center_apprx)[dist_same_center_idx]

    # Second we deal with pairs has two different centers   --------------------------------
    pos1_diff, pos2_diff = pos1[dist_diff_center_idx], pos2[dist_diff_center_idx]
    center1_diff, center2_diff = center1[dist_diff_center_idx], center2[dist_diff_center_idx]
    dist_geo_diff = dist_geo[dist_diff_center_idx]

    # We detect the intersection situation with safe regions.
    intersect_none, intersect_only1, intersect_only2, intersect_both = geo_path_type(pos1_diff, pos2_diff, center1_diff, center2_diff, r)

    # Assign distance on simple cases.
    dist_geo_diff[intersect_none] = dist12[dist_diff_center_idx][intersect_none]
    dist_geo_diff[intersect_only1] = dist_sphere(pos1_diff, pos2_diff, r, center1_diff)[intersect_only1]
    dist_geo_diff[intersect_only2] = dist_sphere(pos1_diff, pos2_diff, r, center2_diff)[intersect_only2]

    # If there are situations that p1p2 intersect with both safe regions.
    # Find p5 and p6 in the first step. (common tangent line).
    if intersect_both.any():
        # Retrieve corresponding points.
        p1_both, p2_both, c1_both, c2_both = pos1_diff[intersect_both], pos2_diff[intersect_both], center1_diff[intersect_both], center2_diff[intersect_both]

        idx_geo_only1, idx_geo_only2, idx_geo_both = geo_path_type_both(p1_both, p2_both, c1_both, c2_both, r)

        # print("Geodesic situation if intersect with both: ", idx_geo_only1, idx_geo_only2, idx_geo_both)
        dist_geo_both = torch.zeros_like(p1_both[:, 0])
        dist_geo_both[idx_geo_only1] = dist_sphere(p1_both[idx_geo_only1] - c1_both[idx_geo_only1], p2_both[idx_geo_only1] - c1_both[idx_geo_only1], r - tol)
        dist_geo_both[idx_geo_only2] = dist_sphere(p1_both[idx_geo_only2] - c2_both[idx_geo_only2], p2_both[idx_geo_only2] - c2_both[idx_geo_only2], r - tol)

        if idx_geo_both.any():
            # print("#### Going into situation that has common tangent lines...")
            p1_both_geo_both, p2_both_geo_both = p1_both[idx_geo_both], p2_both[idx_geo_both]
            c1_both_geo_both, c2_both_geo_both = c1_both[idx_geo_both], c2_both[idx_geo_both]

            # print("Start to solve equations...")
            p_res = get_common_tangent_full(p1_both_geo_both, p2_both_geo_both, c1_both_geo_both, c2_both_geo_both, torch.as_tensor(r))
            # print("Obtain solutions...", p_res)

            dist_both_list = []
            for i in range(len(p_res)):
                p5, p6 = p_res[i]
                p5_neg_height_idx = p5[:, -1] < 0
                p5[p5_neg_height_idx, -1] = torch.zeros_like(p5[p5_neg_height_idx, -1])
                p6_neg_height_idx = p6[:, -1] < 0
                p6[p6_neg_height_idx, -1] = torch.zeros_like(p6[p6_neg_height_idx, -1])

                # Calculate geodesic distance if intersecting with two region.
                dist_tang = torch.norm(p5 - p6, dim=1).reshape(-1, 1)
                dist16 = dist_sphere(p1_both_geo_both - c1_both_geo_both, p6 - c1_both_geo_both, r - tol).reshape(-1, 1)
                dist52 = dist_sphere(p2_both_geo_both - c2_both_geo_both, p5 - c2_both_geo_both, r - tol).reshape(-1, 1)
                dist_both = dist16 + dist52 - dist_tang
                dist_both_list.append(dist_both)

            dist_both_min_sol = torch.min(torch.as_tensor(dist_both_list), dim=0)
            dist_both_min = dist_both_min_sol[0]

            # print("#### Choose solution idx", int(dist_both_min_sol[1]), "solution: ", p_res[dist_both_min_sol[1]])
            # print("#### Choose solution distance", dist_both_min)

            dist_geo_both[idx_geo_both] = dist_both_min

        dist_geo_diff[intersect_both] = dist_geo_both

    # Now we combine them together. --------------------------------------------
    dist_geo[dist_diff_center_idx] = dist_geo_diff
    # print("Returning geodesic distance...", dist_geo)
    return dist_geo


def f_common_tang(x, vec_c1_p1, vec_c2_p2, vec_c1_c2, r):
    x = x.reshape(-1, 10)
    vec_c1_p5 = x[:, :3]
    vec_c2_p6 = x[:, 3:6]
    lamb1, lamb2, lamb3, lamb4 = x[:, 6], x[:, 7], x[:, 8], x[:, 9]

    eq1 = vec_c1_p5 - lamb1[:, np.newaxis] * vec_c1_p1 - lamb2[:, np.newaxis] * (vec_c1_c2 + vec_c2_p6)
    eq2 = vec_c2_p6 - lamb3[:, np.newaxis] * (-vec_c1_c2 + vec_c1_p1) + lamb4[:, np.newaxis] * vec_c2_p2
    eq3 = np.linalg.norm(vec_c1_p5, axis=1).reshape(-1, 1) - r
    eq4 = np.linalg.norm(vec_c2_p6, axis=1).reshape(-1, 1) - r
    eq5 = np.sum(vec_c1_p5 * (vec_c2_p6 - vec_c1_p5 + vec_c1_c2), axis=1).reshape(-1, 1)
    eq6 = np.sum(vec_c2_p6 * (vec_c2_p6 - vec_c1_p5 + vec_c1_c2), axis=1).reshape(-1, 1)
    # print("Solve common tangent equation with results", np.hstack((eq1, eq2, eq3, eq4, eq5, eq6)).flatten())
    return np.hstack((eq1, eq2, eq3, eq4, eq5, eq6)).flatten()


def f_geodesic(x, param_pos_drone, param_pos_centers, r_min):
    """Objective function with spherical distance."""
    # print("New iteration for geodesic path...---------------------------------------")
    num_drones, num_dim = param_pos_drone.shape
    x = x.reshape(-1, num_dim)
    center = param_pos_centers.reshape(-1, num_dim)
    r_min = torch.as_tensor(r_min).double()

    L1 = dist_sphere(torch.from_numpy(param_pos_drone.reshape(-1, 3)).double(),
                    torch.from_numpy(x[0].reshape(-1, 3)).double(),
                    r_min, origin=torch.from_numpy(center[0].reshape(-1, 3)).double())
    L2 = torch.sum(
        dist_geodesic(torch.from_numpy(x[:-1, :].reshape(-1, 3)).double(),
                      torch.from_numpy(x[1:, :].reshape(-1, 3)).double(),
                      torch.from_numpy(center[:-1, :]).reshape(-1, 3).double(),
                      torch.from_numpy(center[1:, :]).reshape(-1, 3).double(), r_min))
    if L2 > 100 or torch.isnan(L2):
        dist_geodesic(torch.from_numpy(x[:-1, :].reshape(-1, 3)).double(),
                      torch.from_numpy(x[1:, :].reshape(-1, 3)).double(),
                      torch.from_numpy(center[:-1, :]).reshape(-1, 3).double(),
                      torch.from_numpy(center[1:, :]).reshape(-1, 3).double(), r_min)
    L = L1 + L2
    # L = L + torch.sum(
    #     dist_geodesic(torch.from_numpy(x[:-1, :].reshape(-1, 3)), torch.from_numpy(x[1:, :].reshape(-1, 3)),
    #                   torch.from_numpy(center[:-1, :]).reshape(-1, 3), torch.from_numpy(x[1:, :]).reshape(-1, 3), r_min))
    # print("Optimizing objective function with length %f" % L)
    # print("Optimizing objective function with length", float(L1), float(L2), float(L))
    # print("Current viewpoints", x)
    return L
