import os
import sys
import copy
import numpy as np
import open3d as o3d
from math import pi
from PIL import Image


def expand_img_from_pred(img_np, exp_size):
    h, w = exp_size
    img = Image.fromarray(img_np)
    img_expand = img.resize((h, h))
    img_new = Image.new(img.mode, (w, h))
    img_new.paste(img_expand, (int((w - h) / 2), 0))
    return img_new


def depth_map_regulate(depth_map):
    min_depth = np.amin(depth_map[depth_map > 0])
    max_depth = np.amax(depth_map[depth_map > 0])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    return depth_map


def vis_depth_map(depth_map):
    formatted = (depth_map * 255 / np.max(depth_map)).astype('uint8')