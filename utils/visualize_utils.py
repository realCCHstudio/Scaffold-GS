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
# 本文件提供用于图像和深度数据可视化的辅助函数，包括添加标签、深度图转换为相机坐标、法线计算等功能。

from typing import Callable, Dict, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F


def add_label_centered(
        img: np.ndarray,
        text: str,
        font_scale: float = 1.0,
        thickness: int = 2,
        alignment: str = "top",
        color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    在图像顶部或底部居中添加文本标签。

    参数：
    - img (np.ndarray): 输入图像。
    - text (str): 要添加的文本内容。
    - font_scale (float): 字体大小比例。
    - thickness (int): 文本厚度。
    - alignment (str): 文本对齐方式，"top" 或 "bottom"。
    - color (Tuple[int, int, int]): 文本颜色 (B, G, R)。

    返回：
    - np.ndarray: 带标签的图像。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, font_scale, thickness=thickness)[0]
    img = img.astype(np.uint8).copy()

    if alignment == "top":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, 50),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    elif alignment == "bottom":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, img.shape[0] - textsize[1]),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        raise ValueError("Unknown text alignment")

    return img


def tensor2rgbjet(
        tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    """
    将张量转换为伪彩色图像（JET 颜色映射）。

    参数：
    - tensor (th.Tensor): 输入张量。
    - x_max (Optional[float]): 数据最大值。
    - x_min (Optional[float]): 数据最小值。

    返回：
    - np.ndarray: JET 彩色图像。
    """
    return cv2.applyColorMap(tensor2rgb(tensor, x_max=x_max, x_min=x_min), cv2.COLORMAP_JET)


def tensor2rgb(
        tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    """
    将张量转换为 RGB 格式的图像。

    参数：
    - tensor (th.Tensor): 输入张量。
    - x_max (Optional[float]): 数据最大值。
    - x_min (Optional[float]): 数据最小值。

    返回：
    - np.ndarray: RGB 图像。
    """
    x = tensor.data.cpu().numpy()
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()

    gain = 255 / np.clip(x_max - x_min, 1e-3, None)
    x = (x - x_min) * gain
    x = x.clip(0.0, 255.0)
    x = x.astype(np.uint8)
    return x


def tensor2image(
        tensor: th.Tensor,
        x_max: Optional[float] = 1.0,
        x_min: Optional[float] = 0.0,
        mode: str = "rgb",
        mask: Optional[th.Tensor] = None,
        label: Optional[str] = None,
) -> np.ndarray:
    """
    将张量转换为图像，并可以选择添加掩膜和标签。

    参数：
    - tensor (th.Tensor): 输入张量。
    - x_max (Optional[float]): 最大像素值。
    - x_min (Optional[float]): 最小像素值。
    - mode (str): 显示模式，"rgb" 或 "jet"。
    - mask (Optional[th.Tensor]): 可选掩膜。
    - label (Optional[str]): 可选标签文本。

    返回：
    - np.ndarray: 转换后的图像。
    """
    tensor = tensor.detach()

    if mask is not None:
        tensor = tensor * mask

    if len(tensor.size()) == 2:
        tensor = tensor[None]

    assert len(tensor.size()) == 3, tensor.size()
    n_channels = tensor.shape[0]
    if n_channels == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif n_channels != 3:
        raise ValueError(f"Unsupported number of channels {n_channels}.")

    img = tensor.permute(1, 2, 0)

    if mode == "rgb":
        img = tensor2rgb(img, x_max=x_max, x_min=x_min)
    elif mode == "jet":
        img[:, :, :3] = img[:, :, [2, 1, 0]]
        img = tensor2rgbjet(img, x_max=x_max, x_min=x_min)
        img[:, :, :3] = img[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"Unsupported mode {mode}.")

    if label is not None:
        img = add_label_centered(img, label)

    return img


def depthImgToPosCam_Batched(d, screenCoords, focal, princpt):
    """
    将深度图像转换为相机坐标。

    参数：
    - d (Tensor): 深度图像，尺寸为 b x 1 x H x W。
    - screenCoords (Tensor): 屏幕坐标，尺寸为 b x 2 x H x W。
    - focal (Tensor): 焦距矩阵，尺寸为 b x 2 x 2。
    - princpt (Tensor): 主点坐标，尺寸为 b x 2。

    返回：
    - Tensor: 相机坐标，尺寸为 b x 3 x H x W。
    """
    p = screenCoords - princpt[:, :, None, None]
    x = (d * p[:, 0:1, :, :]) / focal[:, 0:1, 0, None, None]
    y = (d * p[:, 1:2, :, :]) / focal[:, 1:2, 1, None, None]
    return th.cat([x, y, d], dim=1)


def computeNormalsFromPosCam_Batched(p):
    """
    从相机坐标计算法线。

    参数：
    - p (Tensor): 相机坐标，尺寸为 b x 3 x H x W。

    返回：
    - Tensor: 法线图，尺寸为 b x 3 x H x W。
    """
    p = F.pad(p, (1, 1, 1, 1), "replicate")
    d0 = p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]
    d1 = p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]
    n = th.cross(d0, d1, dim=1)
    norm = th.norm(n, dim=1, keepdim=True)
    norm = norm + 1e-5
    norm[norm < 1e-5] = 1
    return -n / norm


def visualize_normal(inputs, depth_p):
    """
    可视化法线，从深度图计算法线并生成法线图像。

    参数：
    - inputs (Dict): 包含焦距和主点信息的字典。
    - depth_p (Tensor): 深度图像。

    返回：
    - np.ndarray: 法线图像。
    """
    uv = th.stack(
        th.meshgrid(
            th.arange(depth_p.shape[2]), th.arange(depth_p.shape[1]), indexing="xy"
        ),
        dim=0,
    )[None].float().cuda()
    position = depthImgToPosCam_Batched(
        depth_p[None, ...], uv, inputs["focal"], inputs["princpt"]
    )
    normal = 0.5 * (computeNormalsFromPosCam_Batched(position) + 1.0)
    normal = normal[0, [2, 1, 0], :, :]
    normal_p = tensor2image(normal, label="normal_p")

    return normal_p
