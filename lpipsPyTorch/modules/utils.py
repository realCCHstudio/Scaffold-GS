from collections import OrderedDict
import torch

#CCHSTUDIO提供本版本的中文注释
# 文件作用说明：
# 本文件包含用于 LPIPS 计算的辅助工具函数，包括激活标准化和预训练模型权重下载。

def normalize_activation(x, eps=1e-10):
    """
    归一化输入张量的激活值，用于消除不同特征通道的尺度差异。

    参数：
    - x (torch.Tensor): 输入张量。
    - eps (float): 防止除零的极小值。

    返回：
    - torch.Tensor: 归一化后的张量。
    """
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    """
    下载并处理指定网络类型的预训练权重，用于 LPIPS 计算。

    参数：
    - net_type (str): 网络类型，默认 'alex'，可选 'alex'、'squeeze'、'vgg'。
    - version (str): LPIPS 版本，默认 '0.1'。

    返回：
    - OrderedDict: 下载并处理后的模型权重字典。
    """
    # 构建下载链接
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
          + f'master/lpips/weights/v{version}/{net_type}.pth'

    # 下载模型权重
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # 重命名字典中的键
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
