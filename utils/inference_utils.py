import time

import torch
import torch.nn as nn

import models.EasyModel as Easynet
import models.InceptionBlock as Inception
import models.InceptionBlockV2 as Inception_v2
import models.SemExpModel as SemExpNet
from models.AlexNet import AlexNet
from models.AutoEncoderConv import Autoencoder
from models.EasyModel import EasyModel
from models.InceptionBlock import InceptionBlock
from models.InceptionBlockV2 import InceptionBlockV2
from models.SemExpLinearModel import Sem_Exp
from models.VggNet import VggNet
from utils.excel_utils import *


def get_dnn_model(arg: str):
    """
    根据名称获取对应的DNN模型
    :param arg: 模型名字
    :return: 对应的名字
    """
    # 设置默认通道数
    input_channels = 3
    if arg == "alex_net":
        return AlexNet(input_channels=input_channels)
    elif arg == "vgg_net":
        return VggNet(input_channels=input_channels)
    elif arg == "easy_net":
        return EasyModel(in_channels=input_channels)
    elif arg == "inception":
        return InceptionBlock(in_channels=input_channels)
    elif arg == "inception_v2":
        return InceptionBlockV2(in_channels=input_channels)
    elif arg == "AutoEncoderConv":
        return Autoencoder()
    elif arg == "Sem_Exp":
        return Sem_Exp()
    else:
        raise RuntimeError("没有对应的DNN模型")


def show_model_constructor(model, skip=True):
    """
    展示DNN各层结构
    :param model: DNN模型
    :param skip: 是否需要跳过 ReLU BatchNorm Dropout等层
    :return: 展示DNN各层结构
    """
    print("show model constructor as follows: ")
    if len(model) > 0:
        idx = 1
        for layer in model:
            if skip is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue
            print(f'{idx}-{layer}')
            idx += 1
    else:
        print("this model is a empty model")


def show_features(model, input_data, device, epoch_cpu=50, epoch_gpu=100, skip=True, save=False, sheet_name="model",
                  path=None):
    """
    可以输出DNN各层的性质,并将其保存在excel表格中,输出的主要性质如下：
    ["index", "layerName", "computation_time(ms)", "output_shape", "transport_num", "transport_size(MB)","accumulate_time(ms)"]
    [DNN层下标，层名字，层计算时延，层输出形状，需要传输的浮点数数量，传输大小，从第1层开始的累计推理时延]
    :param model: DNN模型
    :param input_data: 输入数据
    :param device: 指定运行设备
    :param epoch_cpu: cpu循环推理次数
    :param epoch_gpu: gpu循环推理次数
    :param skip: 是否跳过不重要的DNN层
    :param save: 是否将内容保存在excel表格中
    :param sheet_name: excel中的表格名字
    :param path: excel路径
    :return: None
    """
    if device == "cuda":
        if not torch.torch.cuda.is_available():
            raise RuntimeError("运行设备上没有cuda 请调整device参数为cpu")

    # 推理之前对设备进行预热
    warmUp(model, input_data, device)

    if save:
        sheet_name = sheet_name
        value = [["index", "layerName", "computation_time(ms)", "output_shape", "transport_num",
                  "transport_size(MB)", "accumulate_time(ms)"]]
        create_excel_xsl(path, sheet_name, value)

    if len(model) > 0:
        idx = 1
        accumulate_time = 0.0
        for layer in model:
            if skip is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue
            temp_x = input_data
            # 记录DNN单层的推理时间
            input_data, layer_time = recordTime(layer, temp_x, device, epoch_cpu, epoch_gpu)
            accumulate_time += layer_time
            # 计算中间传输占用大小为多少MB
            total_num = 1
            for num in input_data.shape:
                total_num *= num
            size = total_num * 4 / 1000 / 1000
            print("------------------------------------------------------------------")
            print(f'{idx}-{layer} \n'
                  f'computation time: {layer_time :.3f} ms\n'
                  f'output shape: {input_data.shape}\t transport_num:{total_num}\t transport_size:{size:.3f}MB\t accumulate time:{accumulate_time:.3f}ms\n')
            # 保存到excel表格中
            if save:
                sheet_name = input_data
                value = [[idx, f"{layer}", round(layer_time, 3), f"{input_data.shape}", total_num, round(size, 3),
                          round(accumulate_time, 3)]]
                write_excel_xls_append(path, sheet_name, value)
            idx += 1
        return input_data
    else:
        print("this model is a empty model")
        return input_data


def warmUp(model, input_data, device):
    """
    预热操作：不对设备进行预热的话，收集的数据会有时延偏差
    :param model: DNN模型
    :param input_data: 输入数据
    :param device: 运行设备类型
    :return: None
    """
    # 预热10轮
    epoch = 10
    model = model.to(device)
    for i in range(1):
        # GPU设备的预热
        if device == "cuda":
            warmUpGpu(model, input_data, device, epoch)
        # CPU设备的预热
        elif device == "cpu":
            warmUpCpu(model, input_data, device, epoch)


def warmUpGpu(model, input_data, device, epoch):
    """ GPU 设备预热"""
    dummy_input = torch.rand(input_data.shape).to(device)
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_input)
        avg_time = 0.0
        for i in range(epoch):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            avg_time += curr_time
        avg_time /= epoch
        # print(f"GPU Warm Up : {curr_time:.3f}ms")
        # print("==============================================")


def warmUpCpu(model, input_data, device, epoch):
    """ CPU 设备预热"""
    dummy_input = torch.rand(input_data.shape).to(device)
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_input)
        avg_time = 0.0
        for i in range(epoch):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            curr_time = end - start
            avg_time += curr_time
        avg_time /= epoch
        # print(f"CPU Warm Up : {curr_time * 1000:.3f}ms")
        # print("==============================================")


def recordTime(model, input_data, device, epoch_cpu, epoch_gpu):
    """
    记录DNN模型或者DNN层的推理时间，根据设备分发到不同函数上进行计算
    :param model: DNN模型
    :param input_data: 输入数据
    :param device: 运行设备
    :param epoch_cpu: cpu循环推理次数
    :param epoch_gpu: gpu循环推理次数
    :return: 输出结果以及推理时延
    """
    model = model.to(device)
    res_x, computation_time = None, None
    # 根据设备的不同分发到不同的函数中进行处理
    if device == "cuda":
        res_x, computation_time = recordTimeGpu(model, input_data, device, epoch_gpu)
    elif device == "cpu":
        res_x, computation_time = recordTimeCpu(model, input_data, device, epoch_cpu)
    # 返回计算结果和计算时延
    return res_x, computation_time


def recordTimeGpu(model, input_data, device, epoch):
    # 记录运算的平均时间
    all_time = 0.0
    with torch.no_grad():
        for i in range(epoch):
            # 测试使用的随机生成数据
            # if torch.is_tensor(input_data):
            #     input_data = torch.rand(input_data.shape).to(device)
            # init loggers
            # 记录时间
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                starter.record()
                # 获取模型执行的结果
                res_x = model(input_data)
                ender.record()
            # wait for GPU SYNC
            # 关于GPU的计算机制 一定要有下面这一行才能准确测量在GPU上的推理时延
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            all_time += curr_time
        all_time /= epoch
    # 返回结果和计算时延
    return res_x, all_time


def recordTimeCpu(model, input_data, device, epoch):
    # 记录运算的平均时间
    all_time = 0.0
    for i in range(epoch):
        # 测试使用的随机生成数据
        # if torch.is_tensor(input_data):
        #     input_data = torch.rand(input_data.shape).to(device)
        # 记录运行的时间信息
        with torch.no_grad():
            start_time = time.perf_counter()
            # 获取模型执行的结果
            res_x = model(input_data)
            end_time = time.perf_counter()
        # 记录时间
        curr_time = end_time - start_time
        all_time += curr_time
    # 记录平均时间
    all_time /= epoch
    # 返回运算结果和平均计算时间
    return res_x, all_time * 1000


def model_partition(model, model_partition_edge):
    """
    根据 model_partition_edge 对DNN模型-model进行划分
    :param model: 传入的DNN模型
    :param model_partition_edge:模型分层点
    :return: 边缘端模型 edge_model, 云端模型 cloud_model
    """
    # 如果model_partition_edge是[]，代表模型全部部署在边缘执行
    if len(model_partition_edge) == 0:
        return model, nn.Sequential()

    # 开始构建边端模型和云端模型
    edge_model, cloud_model = nn.Sequential(), nn.Sequential()
    # 下面三种情况下进行DADS划分DAG结构，其他的情况按照链式结构划分
    if isinstance(model, Inception.InceptionBlock):
        return Inception.construct_edge_cloud_inception_block(model, model_partition_edge)
    if isinstance(model, Inception_v2.InceptionBlockV2):
        return Inception_v2.construct_edge_cloud_inception_block(model, model_partition_edge)
    if isinstance(model, Easynet.EasyModel):
        return Easynet.construct_edge_cloud_inception_block(model, model_partition_edge)
    if isinstance(model, SemExpNet.Sem_Exp):
        return SemExpNet.construct_edge_cloud_inception_block(model, model_partition_edge)

    # 使用链式结构的划分
    if len(model_partition_edge) == 1:
        partition_point = model_partition_edge[0][0]
        idx = 1
        for layer in model:
            if idx <= partition_point:
                edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            else:
                cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            idx += 1
        return edge_model, cloud_model


def model_partition_linear(model, index):
    """
    model_partition函数可以将一个整体的model,划分成两个部分
    划分的大致思路：
        如选定第index层对模型进行划分 ，则代表在第index后对模型进行划分
        将index层之前的层包括第index层 - 封装进edge_model中交给边缘端设备推理
        将index之后的层 - 封装进cloud_model交给云端设备推理
    举例：在第index层之后对模型进行划分
    index = 0 - 代表在初始输入进行划分
    index = 1 - 代表在第1层后对模型进行划分,edge_cloud包括第1层

    :param model: 传入模型
    :param index: 模型划分点
    :return: 划分之后的边端模型和云端模型
    """
    edge_model = nn.Sequential()
    cloud_model = nn.Sequential()
    idx = 1
    for layer in model:
        if idx <= index:
            edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        else:
            cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        idx += 1
    return edge_model, cloud_model
