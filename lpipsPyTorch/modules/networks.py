from typing import Sequence
from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from .utils import normalize_activation

# CCHSTUDIO 提供本版本的中文注释
# 文件作用说明：
# 本文件定义了用于 LPIPS 计算的特征提取网络，包括 AlexNet、SqueezeNet 和 VGG16。
# 每个网络类型被封装成类，具有特定的特征层和通道数设置。此外，还定义了用于线性变换的 LinLayers 类。

def get_network(net_type: str):
    """
    根据网络类型返回相应的特征提取网络实例。

    参数：
    - net_type (str): 网络类型，可选 'alex'、'squeeze' 或 'vgg'。

    返回：
    - BaseNet 子类的实例，用于提取 LPIPS 特征。

    异常：
    - NotImplementedError: 如果指定的 net_type 无效。
    """
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    """
    定义线性层列表，用于对提取的特征进行线性转换。

    参数：
    - n_channels_list (Sequence[int]): 每个卷积层的输入通道数。
    """

    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    """
    基础特征提取网络类，定义了均值和标准差用于标准化，并支持层输出提取。
    """

    def __init__(self):
        super(BaseNet, self).__init__()

        # 注册均值和标准差，用于 z-score 标准化
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        """
        设置网络参数是否需要计算梯度。

        参数：
        - state (bool): 如果为 True，计算梯度；否则不计算。
        """
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        """
        对输入进行 z-score 标准化。

        参数：
        - x (torch.Tensor): 输入张量。

        返回：
        - 标准化后的张量。
        """
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        """
        前向传播，提取特定层的激活值并标准化。

        参数：
        - x (torch.Tensor): 输入图像张量。

        返回：
        - list[torch.Tensor]: 特定层的标准化激活值。
        """
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    """
    定义 SqueezeNet 网络，设定所需的特征层和通道数。
    """

    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    """
    定义 AlexNet 网络，设定所需的特征层和通道数。
    """

    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    """
    定义 VGG16 网络，设定所需的特征层和通道数。
    """

    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
