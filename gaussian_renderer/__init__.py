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

import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

#CCHSTUDIO提供本版本的中文注释
# 文件作用说明：
# 本文件用于定义基于高斯模型的渲染功能，包括生成视角适应的神经高斯分布并进行场景渲染。
# 主要函数有 generate_neural_gaussians 和 render，用于创建高斯对象和渲染场景，
# 以及 prefilter_voxel 用于对体素的预筛选操作。

def generate_neural_gaussians(viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False):
    """
    生成适应视角的神经高斯分布，计算视锥体内的特征、距离、颜色和旋转等属性。

    参数：
    - viewpoint_camera: 表示当前视角的相机参数。
    - pc (GaussianModel): 包含高斯模型和相关数据的对象。
    - visible_mask: 用于视锥体加速过滤的可见性掩码。
    - is_training (bool): 指定是否为训练模式。

    返回：
    - xyz, color, opacity, scaling, rot 等高斯分布的特性，用于渲染。
    """
    # 视锥体过滤，初始化为全可见
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    # 计算观察视角和距离
    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    # 视角适应特征
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
               feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
               feat[:, ::1, :1] * bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)

    # 合并本地视角特征
    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long,
                                          device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # 获取偏移量的不透明度
    neural_opacity = pc.get_opacity_mlp(cat_local_view if pc.add_opacity_dist else cat_local_view_wodist)

    # 不透明度掩码
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > 0.0).view(-1)
    opacity = neural_opacity[mask]

    # 获取偏移量的颜色
    color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1) if pc.add_color_dist else torch.cat(
        [cat_local_view_wodist, appearance], dim=1))
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])

    # 获取偏移量的协方差
    scale_rot = pc.get_cov_mlp(cat_local_view if pc.add_cov_dist else cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])

    # 偏移量处理
    offsets = grid_offsets.view([-1, 3])

    # 组合数据以进行并行掩码处理
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # 处理协方差和旋转
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])

    # 处理偏移量以获得高斯中心
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, visible_mask=None,
           retain_grad=False):
    """
    渲染场景。

    参数：
    - viewpoint_camera: 当前视角相机。
    - pc (GaussianModel): 高斯模型实例。
    - pipe: 管道参数，包含渲染配置。
    - bg_color (torch.Tensor): 背景颜色。
    - scaling_modifier (float): 缩放修正系数。
    - visible_mask: 可见性掩码。
    - retain_grad (bool): 是否保留梯度。

    返回：
    - 字典，包含渲染图像、空间点和选择掩码等信息。
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc,
                                                                                            visible_mask,
                                                                                            is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask,
                                                                      is_training=is_training)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 渲染高斯到图像中
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None)

    if is_training:
        return {"render": rendered_image, "viewspace_points": screenspace_points, "visibility_filter": radii > 0,
                "radii": radii, "selection_mask": mask, "neural_opacity": neural_opacity, "scaling": scaling}
    else:
        return {"render": rendered_image, "viewspace_points": screenspace_points, "visibility_filter": radii > 0,
                "radii": radii}


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    对体素进行预筛选，基于可见性过滤体素。

    参数：
    - viewpoint_camera: 当前视角相机。
    - pc (GaussianModel): 高斯模型实例。
    - pipe: 管道参数，包含渲染配置。
    - bg_color (torch.Tensor): 背景颜色。
    - scaling_modifier (float): 缩放修正系数。
    - override_color: 可选的覆盖颜色。

    返回：
    - 体素可见性过滤结果。
    """
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor

    scales, rotations, cov3D_precomp = None, None, None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D=means3D, scales=scales[:, :3], rotations=rotations,
                                           cov3D_precomp=cov3D_precomp)

    return radii_pure > 0
