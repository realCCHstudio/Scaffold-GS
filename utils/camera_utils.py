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


from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

#CCHSTUDIO提供本版本的中文注释
# 文件作用说明：
# 本文件定义了相机工具函数，用于场景中的相机加载、分辨率缩放和相机信息序列化。
# 它提供了将相机信息从不同的格式转换为标准化 Camera 对象的功能，并支持将相机信息
# 输出为 JSON 格式，用于存储或进一步处理。

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    """
    加载相机信息并创建 Camera 对象。

    参数：
    - args: 命令行参数对象，包含分辨率和设备信息。
    - id (int): 相机的唯一标识符。
    - cam_info: 包含相机信息的对象。
    - resolution_scale (float): 分辨率缩放比例。

    返回：
    - Camera: 包含指定分辨率图像、相机姿态和内参的相机对象。
    """
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "Specify '--resolution/-r' as 1 if rescaling is not desired.")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """
    从相机信息创建 Camera 对象列表。

    参数：
    - cam_infos: 相机信息列表。
    - resolution_scale (float): 分辨率缩放比例。
    - args: 命令行参数对象。

    返回：
    - list: 包含 Camera 对象的列表。
    """
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera: Camera):
    """
    将 Camera 对象转换为 JSON 序列化格式。

    参数：
    - id (int): 相机的唯一标识符。
    - camera (Camera): 需要序列化的 Camera 对象。

    返回：
    - dict: 包含相机位置、旋转矩阵、焦距等信息的字典，适用于 JSON 格式。
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry