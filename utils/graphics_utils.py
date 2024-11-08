#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#CCHSTUDIO提供本版本的中文注释
# 文件作用说明：
# 本文件提供与图形学相关的实用函数和类，包括点云的基本结构、几何变换、世界坐标到视图坐标的转换矩阵、
# 投影矩阵的生成以及视场角和焦距的相互转换。这些工具函数用于渲染、相机视角转换和计算视角相关的参数。

import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    """
    定义基本点云数据结构，包含点、颜色和法线信息。

    属性：
    - points (np.array): 点的坐标数组。
    - colors (np.array): 点的颜色数组。
    - normals (np.array): 点的法线数组。
    """
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    """
    对点集合应用几何变换矩阵。

    参数：
    - points (torch.Tensor): 点的坐标张量，大小为 (P, 3)，其中 P 是点的数量。
    - transf_matrix (torch.Tensor): 4x4 变换矩阵。

    返回：
    - torch.Tensor: 经过变换的点坐标。
    """
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    denom = points_out[..., 3:] + 1e-7  # 避免除零
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    """
    生成从世界坐标到视图坐标的转换矩阵。

    参数：
    - R (np.array): 旋转矩阵。
    - t (np.array): 平移向量。

    返回：
    - np.array: 4x4 世界到视图坐标的转换矩阵。
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    生成从世界坐标到视图坐标的转换矩阵，并支持额外的平移和缩放。

    参数：
    - R (np.array): 旋转矩阵。
    - t (np.array): 平移向量。
    - translate (np.array): 额外平移向量。
    - scale (float): 缩放因子。

    返回：
    - np.array: 4x4 世界到视图坐标的转换矩阵。
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    根据近、远平面距离和视角计算投影矩阵。

    参数：
    - znear (float): 近平面距离。
    - zfar (float): 远平面距离。
    - fovX (float): 水平方向视场角。
    - fovY (float): 垂直方向视场角。

    返回：
    - torch.Tensor: 4x4 投影矩阵。
    """
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    """
    将视场角转换为焦距。

    参数：
    - fov (float): 视场角（弧度制）。
    - pixels (int): 图像分辨率（像素数）。

    返回：
    - float: 焦距。
    """
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    """
    将焦距转换为视场角。

    参数：
    - focal (float): 焦距。
    - pixels (int): 图像分辨率（像素数）。

    返回：
    - float: 视场角（弧度制）。
    """
    return 2 * math.atan(pixels / (2 * focal))
