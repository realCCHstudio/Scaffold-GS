import torch
import torch.nn as nn
from .networks import get_network, LinLayers
from .utils import get_state_dict

# CCHSTUDIO 提供本版本的中文注释
# 文件作用说明：
# 本文件实现了 LPIPS (Learned Perceptual Image Patch Similarity) 计算标准，使用指定的预训练网络对图像进行特征提取，
# 然后通过线性层计算图像的感知差异，输出最终的感知距离。

class LPIPS(nn.Module):
    r"""
    定义 LPIPS 类，用于计算两幅图像之间的感知相似度。

    参数：
    - net_type (str): 用于提取特征的网络类型，可选 'alex'、'squeeze' 或 'vgg'。默认值为 'alex'。
    - version (str): LPIPS 的版本，目前仅支持 '0.1'。
    """

    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        assert version in ['0.1'], '仅支持版本 v0.1'

        super(LPIPS, self).__init__()

        # 初始化预训练的特征提取网络
        self.net = get_network(net_type)

        # 初始化线性层
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        前向传播计算 LPIPS 距离。

        参数：
        - x, y (torch.Tensor): 输入的图像张量。

        返回：
        - torch.Tensor: 表示输入图像之间的 LPIPS 感知距离。
        """
        # 获取 x 和 y 的特征
        feat_x, feat_y = self.net(x), self.net(y)

        # 计算特征差异
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]

        # 通过线性层计算距离并对每层结果取均值
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        # 合并所有层结果并求和
        return torch.sum(torch.cat(res, 0), 0, True)
