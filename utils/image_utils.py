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
# 本文件提供图像质量评估的实用函数，包括均方误差 (MSE) 和峰值信噪比 (PSNR) 计算，用于衡量图像之间的差异。

import torch


def mse(img1, img2):
    """
    计算两个图像之间的均方误差 (Mean Squared Error, MSE)。

    参数：
    - img1 (torch.Tensor): 第一个图像张量。
    - img2 (torch.Tensor): 第二个图像张量。

    返回：
    - torch.Tensor: 图像之间的均方误差，每个图像的 MSE 作为独立值。
    """
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    """
    计算两个图像之间的峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)。

    参数：
    - img1 (torch.Tensor): 第一个图像张量，值范围在 [0, 1]。
    - img2 (torch.Tensor): 第二个图像张量，值范围在 [0, 1]。

    返回：
    - torch.Tensor: 图像之间的 PSNR 值，每个图像的 PSNR 作为独立值。
    """
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
