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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

#CCHSTUDIO提供本版本的中文注释
# 文件作用说明：
# 本文件定义了 Camera 和 MiniCam 类，用于存储和管理相机的参数和投影矩阵。
# 这些类用于处理渲染场景中的相机视角和投影转换。

class Camera(nn.Module):
    """
    Camera 类表示场景中的相机，提供投影和视角转换矩阵。

    参数：
    - colmap_id (int): 相机在 COLMAP 数据集中的 ID。
    - R (np.array): 旋转矩阵。
    - T (np.array): 平移向量。
    - FoVx, FoVy (float): 相机的水平和垂直视场角。
    - image (torch.Tensor): 输入图像张量。
    - gt_alpha_mask (torch.Tensor): 可选的 alpha 蒙版。
    - image_name (str): 图像名称。
    - uid (int): 唯一标识符。
    - trans (np.array): 可选的平移向量。
    - scale (float): 缩放因子。
    - data_device (str): 设备类型，默认为 "cuda"。
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # 初始化图像，裁剪值并应用设备
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 应用 alpha 蒙版（如果存在）
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # 设置相机的远近平面
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # 计算视角转换矩阵和投影矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    """
    MiniCam 类是一个简化的相机类，包含基本的相机参数，用于快速生成视图和投影矩阵。

    参数：
    - width (int): 图像宽度。
    - height (int): 图像高度。
    - fovy, fovx (float): 垂直和水平视场角。
    - znear, zfar (float): 相机的近、远裁剪平面。
    - world_view_transform (torch.Tensor): 视角转换矩阵。
    - full_proj_transform (torch.Tensor): 完整的投影转换矩阵。
    """
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
