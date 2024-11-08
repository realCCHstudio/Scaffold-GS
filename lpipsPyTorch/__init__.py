import torch
from .modules.lpips import LPIPS


# 文件作用说明：
# 本文件用于实现 LPIPS (Learned Perceptual Image Patch Similarity) 指标计算。
# 该指标用于衡量图像之间的感知相似度，提供 AlexNet、SqueezeNet 和 VGG 三种特征提取网络类型。

def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""
    计算两幅图像的感知相似度 (LPIPS)，用于图像质量评估。

    参数：
    - x, y (torch.Tensor): 待比较的输入图像张量。
    - net_type (str): 用于提取特征的网络类型，可选 'alex'、'squeeze' 或 'vgg'。默认值为 'alex'。
    - version (str): LPIPS 的版本，默认值为 '0.1'。

    返回：
    - torch.Tensor: 表示 x 和 y 之间的感知距离。
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)  # 初始化 LPIPS 计算标准
    return criterion(x, y)  # 返回计算出的感知距离
