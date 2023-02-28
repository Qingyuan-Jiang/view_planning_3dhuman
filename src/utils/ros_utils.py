import ros_numpy
import numpy as np
import open3d as o3d


def pcd_ros2o3d(pcd_ros, remove_nans=True):
    """ covert ros point cloud to open3d point cloud
    Refer to: https://github.com/SeungBack/open3d-ros-helper
    Args:
        pcd_ros (sensor.msg.PointCloud2): ros point cloud message
        remove_nans (bool): if true, ignore the NaN points
    Returns:
        pcd_o3d (open3d.geometry.PointCloud): open3d point cloud
    """
    field_names = [field.name for field in pcd_ros.fields]
    is_rgb = 'rgb' in field_names
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_ros).ravel()
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    if is_rgb:
        cloud_npy = np.zeros(cloud_array.shape + (4,), dtype=np.float)
    else:
        cloud_npy = np.zeros(cloud_array.shape + (3,), dtype=np.float)

    cloud_npy[..., 0] = cloud_array['x']
    cloud_npy[..., 1] = cloud_array['y']
    cloud_npy[..., 2] = cloud_array['z']

    pcd_o3d = o3d.geometry.PointCloud()

    if len(np.shape(cloud_npy)) == 3:
        cloud_npy = np.reshape(cloud_npy[:, :, :3], [-1, 3], 'F')
    pcd_o3d.points = o3d.utility.Vector3dVector(cloud_npy[:, :3])

    if is_rgb:
        rgb_npy = cloud_array['rgb']
        rgb_npy.dtype = np.uint32
        r = np.asarray((rgb_npy >> 16) & 255, dtype=np.uint8)
        g = np.asarray((rgb_npy >> 8) & 255, dtype=np.uint8)
        b = np.asarray(rgb_npy & 255, dtype=np.uint8)
        rgb_npy = np.asarray([r, g, b])
        rgb_npy = rgb_npy.astype(np.float) / 255
        rgb_npy = np.swapaxes(rgb_npy, 0, 1)
        pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_npy)
    return pcd_o3d