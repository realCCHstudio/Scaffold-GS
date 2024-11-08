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
import traceback
import socket
import json
from scene.cameras import MiniCam

#CCHSTUDIO提供本版本的中文注释
# 文件作用说明：
# 本文件提供了一个 GUI 网络接口，用于监听来自客户端的连接，并接收和解析摄像机参数。
# 通过 socket 连接，接受客户端传入的渲染配置信息，以实现实时渲染控制。

host = "127.0.0.1"  # 默认主机地址
port = 6009  # 默认端口

conn = None  # 连接对象
addr = None  # 地址对象

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 初始化监听套接字


def init(wish_host, wish_port):
    """
    初始化网络监听器，绑定指定的主机和端口。

    参数：
    - wish_host (str): 监听的主机地址。
    - wish_port (int): 监听的端口号。
    """
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)  # 设置非阻塞模式


def try_connect():
    """
    尝试接受客户端连接。如果未连接则跳过异常处理。
    """
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)  # 设置连接超时时间
    except Exception as inst:
        pass


def read():
    """
    从连接中读取消息的长度和内容。

    返回：
    - dict: 从 JSON 格式解码的消息内容。
    """
    global conn
    messageLength = conn.recv(4)  # 接收消息长度
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)  # 接收消息内容
    return json.loads(message.decode("utf-8"))


def send(message_bytes, verify):
    """
    发送消息和验证数据到客户端。

    参数：
    - message_bytes (bytes): 要发送的消息内容。
    - verify (str): 验证字符串，用于确保数据完整性。
    """
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))


def receive():
    """
    接收并解析来自客户端的摄像机配置信息，返回 MiniCam 实例和其他配置信息。

    返回：
    - custom_cam (MiniCam): 表示摄像机参数的 MiniCam 实例。
    - do_training (bool): 指示是否处于训练模式。
    - do_shs_python (bool): 是否在 Python 中处理 SH。
    - do_rot_scale_python (bool): 是否在 Python 中进行旋转缩放处理。
    - keep_alive (bool): 是否保持连接。
    - scaling_modifier (float): 缩放修正系数。
    """
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            # 解析消息中的各项参数
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]

            # 设置视角变换矩阵并应用相应的旋转方向
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:, 1] = -world_view_transform[:, 1]
            world_view_transform[:, 2] = -world_view_transform[:, 2]

            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:, 1] = -full_proj_transform[:, 1]

            # 创建 MiniCam 实例，代表摄像机在当前视角的参数
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)

        except Exception as e:
            print("")
            traceback.print_exc()
            raise e

        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None
