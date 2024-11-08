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
# 本文件包含系统相关的辅助函数，用于目录创建和查找指定文件夹中迭代保存的最高编号的文件。

from errno import EEXIST
from os import makedirs, path
import os


def mkdir_p(folder_path):
    """
    创建指定路径的目录，若目录已存在则不做任何操作，类似于命令行中的 `mkdir -p`。

    参数：
    - folder_path (str): 要创建的目录路径。

    异常处理：
    - 如果目录已存在，则忽略异常。
    """
    try:
        makedirs(folder_path)
    except OSError as exc:  # 兼容 Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    """
    查找指定文件夹中文件名包含的最大迭代数。
    假设文件名的格式为 "prefix_number"，本函数提取这些数字并返回最大值。

    参数：
    - folder (str): 包含迭代文件的文件夹路径。

    返回：
    - int: 最高的迭代编号。
    """
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
