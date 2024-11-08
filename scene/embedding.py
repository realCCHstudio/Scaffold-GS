# 本文件参考了 https://github.com/nerfstudio-project/nerfstudio/blob/a8e6f8fa3fd6c0ad2f3e681dcf1519e74ad2230f/nerfstudio/field_components/embedding.py
# 感谢原作者的出色工作！

import torch
from abc import abstractmethod
from typing import Optional
from jaxtyping import Shaped
from torch import Tensor, nn


# 文件作用说明：
# 本文件定义了字段组件和嵌入类，支持输入数据的嵌入表示，提供通用的字段组件框架以及可用于场景生成的嵌入层。

class FieldComponent(nn.Module):
    """
    通用字段组件类，可组合以存储和计算字段。

    参数：
    - in_dim (Optional[int]): 模块的输入维度。
    - out_dim (Optional[int]): 模块的输出维度。
    """

    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """
        如果模块内没有 torch.nn 成员则什么都不做。否则初始化它们。
        """
        pass

    def set_in_dim(self, in_dim: int) -> None:
        """
        设置编码的输入维度。

        参数：
        - in_dim (int): 输入维度。
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """
        返回编码的输出维度。

        返回：
        - int: 输出维度。
        """
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """
        返回处理后的张量。

        参数：
        - in_tensor (Shaped[Tensor, "*bs input_dim"]): 要处理的输入张量。

        返回：
        - Shaped[Tensor, "*bs output_dim"]: 处理后的张量。
        """
        raise NotImplementedError


class Embedding(FieldComponent):
    """
    嵌入类，用于索引嵌入。

    参数：
    - in_dim (int): 嵌入的数量。
    - out_dim (int): 嵌入向量的维度。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """
        初始化嵌入层，创建指定大小的嵌入矩阵。
        """
        self.embedding = torch.nn.Embedding(self.in_dim, self.out_dim)

    def mean(self, dim=0):
        """
        返回嵌入权重在指定维度上的平均值。

        参数：
        - dim (int): 计算均值的维度。

        返回：
        - torch.Tensor: 嵌入权重的均值。
        """
        return self.embedding.weight.mean(dim)

    def forward(self, in_tensor: Shaped[Tensor, "*batch input_dim"]) -> Shaped[Tensor, "*batch output_dim"]:
        """
        执行前向传播，返回嵌入向量。

        参数：
        - in_tensor (Shaped[Tensor, "*batch input_dim"]): 输入张量。

        返回：
        - Shaped[Tensor, "*batch output_dim"]: 嵌入后的张量。
        """
        return self.embedding(in_tensor)
