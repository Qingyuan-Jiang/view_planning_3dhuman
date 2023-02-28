# Utility functions for Airsim Simulation.

import os

import airsim as sim
import cv2
import numpy as np


def get_RGBImage(client, filename, camera_name='0'):
    # scene vision image in uncompressed RGBA array
    responses = client.simGetImages([sim.ImageRequest(camera_name, sim.ImageType.Scene, False, False)])
    # print('Retrieved images: %d', len(responses))

    assert len(responses) == 1, "Obtain more than one response."
    response = responses[0]

    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channel image array H X W X 3
    # plt.imshow(img_rgb)
    # plt.pause(0.001)
    # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb[:, :, [0, 1, 2]])  # write to png
    return img_rgb


def get_SegmentRGBImage(client, filename, camera_name='0'):
    # scene vision image in uncompressed RGBA array
    success = client.simSetSegmentationObjectID("[\w]*", 0, True)
    success = client.simSetSegmentationObjectID("person_actor", 1)
    responses = client.simGetImages([sim.ImageRequest(camera_name, sim.ImageType.Scene, False, False),
                                     sim.ImageRequest(camera_name, sim.ImageType.Segmentation, False, False)])
    # print('Retrieved images: %d', len(responses))

    # assert len(responses) == 1, "Obtain more than one response."
    rgb = responses[0]
    img_rgb1d = np.frombuffer(rgb.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgb = img_rgb1d.reshape(rgb.height, rgb.width, 3)  # reshape array to 3 channel image array H X W X 3
    cv2.imwrite(os.path.normpath(filename.format('/rgb') + '.png'), img_rgb[:, :, [0, 1, 2]])  # write to png

    seg = responses[1]
    img_seg1d = np.fromstring(seg.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_seg = img_seg1d.reshape(seg.height, seg.width, 3)  # reshape array to 3 channel image array H X W X 3
    # img_seg = np.flipud(img_seg)
    cv2.imwrite(os.path.normpath(filename.format('/seg') + '.png'), img_seg[:, :, [0, 1, 2]])  # write to png

    mask = img_seg > 0
    img_masked = img_rgb * mask
    cv2.imwrite(os.path.normpath(filename.format('/mask') + '.png'), img_masked[:, :, [0, 1, 2]])  # write to png

    # plt.imshow(img_rgb)
    # plt.pause(0.001)
    # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    return img_rgb


def get_FullImage(client, filename, camera_name='0', actor_name='person_actor'):
    responses = client.simGetImages([sim.ImageRequest(camera_name, sim.ImageType.Scene, False, False),
                                     sim.ImageRequest(camera_name, sim.ImageType.Segmentation, False, False),
                                     sim.ImageRequest(camera_name, sim.ImageType.DepthPlanar, True, False)])

    rgb = responses[0]
    img_rgb1d = np.frombuffer(rgb.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgb = img_rgb1d.reshape(rgb.height, rgb.width, 3)  # reshape array to 3 channel image array H X W X 3
    cv2.imwrite(os.path.normpath(filename.format('/rgb/rgb') + '.png'), img_rgb[:, :, [0, 1, 2]])  # write to png

    seg = responses[1]
    img_seg1d = np.fromstring(seg.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_seg = img_seg1d.reshape(seg.height, seg.width, 3)  # reshape array to 3 channel image array H X W X 3
    cv2.imwrite(os.path.normpath(filename.format('/seg/seg') + '.png'), img_seg[:, :, [0, 1, 2]])  # write to png

    mask = img_seg > 0
    img_masked = img_rgb * mask
    cv2.imwrite(os.path.normpath(filename.format('/mask/mask') + '.png'), img_masked[:, :, [0, 1, 2]])  # write to png

    depth = responses[2]
    img_depth = np.array(depth.image_data_float, dtype=np.float64)
    img_depth = img_depth.reshape((depth.height, depth.width, -1))
    img_depth_visual = np.array(img_depth * 255, dtype=np.uint8)
    cv2.imwrite(os.path.normpath(filename.format('/depth/depth') + '.png'), img_depth_visual)
    np.save(os.path.normpath(filename.format('/depth/depth') + '.npy'), img_depth)

    return img_rgb[:, :, [2, 1, 0]], img_seg, img_masked, img_depth


def get_Kmatrix(fov_deg, x, y):
    fx = x / np.tan(np.deg2rad(fov_deg / 2))
    fy = y / np.tan(np.deg2rad(fov_deg / 2))
    return np.array([[fx, 0, x],
                     [0, fy, y],
                     [0, 0, 1]])


def cpose2msg(pos, ori):
    pos, ori = pos.flatten(), ori.flatten()
    cpose_msg = sim.Pose(sim.Vector3r(pos[0], pos[1], pos[2]), sim.Quaternionr(ori[0], ori[1], ori[2], ori[3]))
    return cpose_msg
