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
# 本文件提供常用的损失函数，包括 L1 损失、L2 损失和结构相似性 (SSIM) 指数，用于衡量模型预测输出与真实图像之间的差异。

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    """
    计算 L1 损失，即网络输出与真实值之间的绝对误差的平均值。

    参数：
    - network_output (torch.Tensor): 网络输出的张量。
    - gt (torch.Tensor): 真实值的张量。

    返回：
    - torch.Tensor: L1 损失值。
    """
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    """
    计算 L2 损失，即网络输出与真实值之间的均方误差。

    参数：
    - network_output (torch.Tensor): 网络输出的张量。
    - gt (torch.Tensor): 真实值的张量。

    返回：
    - torch.Tensor: L2 损失值。
    """
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    """
    生成 1D 高斯分布向量。

    参数：
    - window_size (int): 高斯窗口的大小。
    - sigma (float): 高斯分布的标准差。

    返回：
    - torch.Tensor: 归一化后的高斯分布向量。
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    创建 2D 高斯窗口，用于 SSIM 计算。

    参数：
    - window_size (int): 窗口的大小。
    - channel (int): 图像通道数。

    返回：
    - torch.Tensor: 用于 SSIM 计算的 2D 高斯窗口。
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性 (SSIM) 指数，用于评估两幅图像之间的相似性。

    参数：
    - img1 (torch.Tensor): 第一个图像张量。
    - img2 (torch.Tensor): 第二个图像张量。
    - window_size (int): 高斯窗口的大小，默认为 11。
    - size_average (bool): 是否对 SSIM 结果取平均，默认为 True。

    返回：
    - torch.Tensor: SSIM 值。
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    计算 SSIM 的内部辅助函数，使用给定窗口执行卷积操作。

    参数：
    - img1 (torch.Tensor): 第一个图像张量。
    - img2 (torch.Tensor): 第二个图像张量。
    - window (torch.Tensor): 用于 SSIM 计算的高斯窗口。
    - window_size (int): 窗口的大小。
    - channel (int): 图像的通道数。
    - size_average (bool): 是否对 SSIM 结果取平均。

    返回：
    - torch.Tensor: SSIM 值。
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
