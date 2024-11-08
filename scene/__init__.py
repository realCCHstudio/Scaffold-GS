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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

#CCHSTUDIO提供本版本的中文注释
# 文件作用说明：
# 本文件定义了 Scene 类，用于加载和管理场景数据，包括训练和测试相机的加载。
# 根据不同数据集类型（如 Colmap 和 Blender）配置场景，并在指定迭代时加载已训练的高斯模型。

class Scene:
    """
    Scene 类用于加载和管理场景的相关信息，包括训练和测试相机、加载迭代模型等。
    """
    gaussians: GaussianModel  # 场景中包含的高斯模型对象

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], ply_path=None):
        """
        初始化场景类，根据数据源加载相机和点云等信息。

        参数：
        - args (ModelParams): 模型参数。
        - gaussians (GaussianModel): 场景使用的高斯模型实例。
        - load_iteration (int): 加载指定的训练迭代，-1 表示加载最后一次迭代。
        - shuffle (bool): 是否随机打乱相机顺序。
        - resolution_scales (list[float]): 加载的分辨率缩放比例。
        - ply_path (str): 可选的 PLY 文件路径，用于加载自定义点云。
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 加载指定迭代的已训练模型
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化训练和测试相机字典
        self.train_cameras = {}
        self.test_cameras = {}

        # 识别场景类型并加载相应信息
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,
                                                           ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"

        self.gaussians.set_appearance(len(scene_info.train_cameras))

        # 初始化或加载点云和相机数据
        if not self.loaded_iter:
            if ply_path is not None:
                # 将自定义 PLY 文件复制到模型路径
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                            'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                       'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # 将相机信息转换为 JSON 并保存
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 随机打乱训练和测试相机的顺序
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # 设置相机的空间范围
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        # 如果加载了特定迭代，则加载对应的点云和模型检查点
        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                                 "point_cloud",
                                                                 "iteration_" + str(self.loaded_iter),
                                                                 "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                             "point_cloud",
                                                             "iteration_" + str(self.loaded_iter)))
        else:
            # 如果未加载特定迭代，则从点云创建新的高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前迭代的点云和模型检查点。

        参数：
        - iteration (int): 当前迭代数，用于文件命名。
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        """
        获取指定分辨率比例的训练相机列表。

        参数：
        - scale (float): 分辨率比例。

        返回：
        - list: 指定比例的训练相机。
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """
        获取指定分辨率比例的测试相机列表。

        参数：
        - scale (float): 分辨率比例。

        返回：
        - list: 指定比例的测试相机。
        """
        return self.test_cameras[scale]
