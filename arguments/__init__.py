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

from argparse import ArgumentParser, Namespace
import sys
import os


# 文件作用说明：
# 本文件用于定义参数组类及相关方法，提供加载、解析和管理程序的运行参数功能。
# 包括模型、优化、以及管道执行相关的参数组，支持从命令行和配置文件中获取参数。
# 可以动态地构建参数组并将其值提取到一个新的参数实例中。

class GroupParams:
    """一个空类，用于在不同参数组之间传递和管理参数。"""
    pass


class ParamGroup:
    """
    ParamGroup 类的作用是提供参数组管理功能。每个参数组对应特定的运行设置，并支持参数默认值填充。
    """

    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        """
        初始化一个参数组，将类属性添加为命令行参数。

        参数：
        - parser (ArgumentParser): 用于解析命令行参数的 ArgumentParser 实例。
        - name (str): 参数组的名称。
        - fill_none (bool): 是否将参数默认值设为 None。
        """
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        """
        从命令行参数中提取参数值，将其存入 GroupParams 实例。

        参数：
        - args (Namespace): 命令行参数。

        返回：
        - GroupParams 实例，其中包含提取的参数。
        """
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    """
    定义模型加载相关参数，包括文件路径、背景处理、特征维度等。
    """

    def __init__(self, parser, sentinel=False):
        # 定义各种模型参数，初始化时提供默认值
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size = 0.001  # 若 voxel_size<=0，则使用 1nn 距离
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        # 其他模型相关参数
        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        # 外观和分辨率控制
        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1
        self.undistorted = False

        # Bungeenerf 数据集中特有参数配置
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        """
        从命令行参数中提取并返回配置参数，确保 source_path 为绝对路径。

        参数：
        - args (Namespace): 命令行参数。

        返回：
        - GroupParams 实例，其中包含提取的模型参数。
        """
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    """
    定义管道执行相关参数，如是否在 Python 中转换 SH 或计算 3D 协方差等。
    """

    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    """
    定义优化相关参数，包括学习率、梯度控制和锚点密集化的相关设置。
    """

    def __init__(self, parser):
        # 设置训练中的各类学习率、延迟乘数、最大步骤数等参数
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # 其他学习率和参数初始化
        ...
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    """
    合并命令行参数和配置文件参数，以确保最终配置项的完整性和优先级。

    参数：
    - parser (ArgumentParser): 用于解析命令行的 ArgumentParser 实例。

    返回：
    - Namespace：合并后的参数命名空间，优先使用命令行参数。
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
