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
# 本文件包含多个通用工具函数，支持逆 sigmoid 计算、PIL 图像转换为 Torch 张量、学习率调节函数生成、
# 矩阵变换构建和控制台输出封装等操作。这些函数在深度学习模型训练和图像处理工作流中被广泛使用。

import torch
import sys
from datetime import datetime
import numpy as np
import random


def inverse_sigmoid(x):
    """
    计算输入 x 的逆 sigmoid 函数。

    参数：
    - x (torch.Tensor): 输入张量。

    返回：
    - torch.Tensor: 逆 sigmoid 变换后的张量。
    """
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    """
    将 PIL 图像转换为给定分辨率的 Torch 张量。

    参数：
    - pil_image (PIL.Image): 输入 PIL 图像。
    - resolution (tuple): 转换后的图像分辨率 (width, height)。

    返回：
    - torch.Tensor: 转换后的 Torch 图像张量。
    """
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    创建学习率的指数衰减函数。初始步数使用延迟因子进行控制，随后指数衰减至最终学习率。
    连续学习率衰减函数。源自JaxNeRF。
    返回的学习率在step=0时为lr_init，在step=max_steps时为lr_final，在其他地方则采用对数线性插值（相当于指数衰减）。
    如果lr_delay_steps>0，则学习率将根据lr_delay_mult的某个平滑函数进行缩放，使得在优化开始时的初始学习率为lr_init*lr_delay_mult，但在steps>lr_delay_steps之后将逐渐恢复到正常的学习率。
    参数：
    - lr_init (float): 初始学习率。
    - lr_final (float): 最终学习率。
    - lr_delay_steps (int): 延迟步数，延迟步数内减缓学习率衰减速度。
    - lr_delay_mult (float): 延迟系数，用于平滑学习率过渡。
    - max_steps (int): 最大训练步数。

    返回：
    - function: 生成学习率的调节函数。
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # 如果学习率被设为 0，则不启用该参数
            return 0.0
        if lr_delay_steps > 0:
            # 使用反向余弦衰减
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    """
    从对角矩阵中提取 6 个唯一元素，生成新的矩阵。

    参数：
    - L (torch.Tensor): 输入对角张量。

    返回：
    - torch.Tensor: 包含唯一元素的张量，表示不对称矩阵的下三角部分。
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    """
    提取对称矩阵的下三角元素，生成表示不对称性的张量。

    参数：
    - sym (torch.Tensor): 输入对称张量。

    返回：
    - torch.Tensor: 表示不对称性的张量。
    """
    return strip_lowerdiag(sym)


def build_rotation(r):
    """
    生成旋转矩阵。

    参数：
    - r (torch.Tensor): 包含四元数 (r, x, y, z) 的张量。

    返回：
    - torch.Tensor: 旋转矩阵。
    """
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    """
    根据缩放和旋转四元数生成旋转矩阵 L。

    参数：
    - s (torch.Tensor): 缩放张量。
    - r (torch.Tensor): 四元数旋转张量。

    返回：
    - torch.Tensor: 包含缩放和旋转的最终矩阵 L。
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L


def safe_state(silent):
    """
    设置控制台输出格式和随机种子，方便在模型训练中复现实验。

    参数：
    - silent (bool): 是否禁止控制台输出。

    返回：
    - None
    """
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
