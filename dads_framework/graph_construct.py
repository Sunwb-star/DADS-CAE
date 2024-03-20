import pickle
import sys

import networkx as nx
import torch
from matplotlib import pyplot as plt

from net.net_utils import get_speed
from utils.inference_utils import recordTime

inf = sys.maxsize
construction_time = 0.0
predictor_dict = {}


def get_layers_latency(model, device):
    """
    获取模型model在云端设备或边端设备上的各层推理时延，用于构建有向图
    :param model: DNN模型
    :param device: 推理设备
    :return: layers_latency[] 代表各层的推理时延
    """
    # 一个用来记录每一层的编号对应的输出，用来做计算延迟测试用的
    dict_layer_output = {}
    # 初始输入数据，随机初始化，只是用来进行延迟测试的数据
    # 这就相当于是第一层神经网络的输入数据
    input_x = torch.rand((1, 24, 240, 240)).to(device)
    orientation = torch.tensor(30).to(device)
    goal_index = torch.tensor(0).to(device)
    input = [input_x, orientation, goal_index]
    input_bak = [input_x, orientation, goal_index]
    # 记录每一层的延迟
    layers_latency = []
    # 对于model中迭代的每一层网络来进行操作
    for layer_index, layer in enumerate(model):
        # 对于某一层先检查其输入是否要进行修改
        # 如果该模型需要进行DAG拓扑，并且当前层是分支网络开始那个点或者汇总点的话，就进行下面的操作：
        if model.has_dag_topology and (layer_index + 1) in model.dag_dict.keys():
            # 从model.dag_dict[layer_index + 1]中取出分支层的前一层的index
            pre_input_cond = model.dag_dict[layer_index + 1]
            # 如果当前层所依赖的前一层信息的是一个列表（当当前层是网络的回合点的时候会出现这样的状况），代表当前层有多个输入
            if isinstance(pre_input_cond, list):
                input = []
                # 对于concat操作,输入应为一个列表
                for pre_index in pre_input_cond:
                    input.append(dict_layer_output[pre_index])
                input[1] = input[1].unsqueeze(0)
                input[2] = input[2].unsqueeze(0)
            else:
                # 当前层的的输入从其依赖层中获得
                input = dict_layer_output[pre_input_cond]
        # 当input不是一个list的时候，就将数据放在相应device中，第一层网络就是这样
        if not isinstance(input, list):
            input = input.to(device)
        # 将该层神经网络放到相应device中
        layer = layer.to(device)
        if (layer_index == 1):
            input = input_bak[0]
        elif (layer_index == 16):
            input = input_bak[1]
        elif (layer_index == 17):
            input = input_bak[2]
        # 记录推理时延和当前层的随机输出结果
        input, lat = recordTime(layer, input, device, epoch_cpu=10, epoch_gpu=10)
        # 如果模型需要进行DAG拓扑并且当前层是网络的分支点或者聚合点的话就更新dict_layer_output，用来为分支点或者聚合点后的网络层提供数据
        if model.has_dag_topology and (layer_index + 1) in model.record_output_list:
            dict_layer_output[layer_index + 1] = input
        # 记录当前层的网络延迟
        layers_latency.append(lat)
    # 返回网络延迟
    return layers_latency


def add_graph_edge(graph, vertex_index, input, layer_index, layer,
                   bandwidth, net_type, edge_latency, cloud_latency,
                   dict_input_size_node_name, dict_node_layer, dict_layer_input_size, dict_layer_output,
                   record_flag):
    """
    向一个有向图中添加节点和边的信息
    :param graph: 向哪个有向图中添加
    :param vertex_index: 当前构建的顶点编号
    :param input: 当前层的输入
    :param layer_index: 当前层的标号
    :param layer: 当前层类型
    :param bandwidth: 互联网网络带宽
    :param net_type: 网络连接类型
    :param edge_latency: 在边缘设备上该层的推理时延
    :param cloud_latency: 在云端设备上该层的推理时延
    :param dict_input_size_node_name:   字典：key:输入 value:对应的顶点编号
    :param dict_node_layer:             字典：key:顶点编号 value:对应DNN中第几层
    :param dict_layer_input_size:       字典：key:DNN中第几层 value:对应的输入大小
    :param dict_layer_output:            字典：key:DNN中第几层 value:对应的输出
    :param record_flag: 只有某些关键层才会记录层的输出--->只有分支点和汇合点的前驱节点会记录输出
    :return: 当前构建的顶点数目vertex_index ，以及当前层的输出（会用于作为下一层的输入）
    """

    # 云端设备节点
    cloud_vertex = "cloud"
    # 边缘设备节点
    edge_vertex = "edge"
    # 获取当前层在边缘端设备上的推理时延以及在云端设备上的推理时延
    # edge_lat = predict_model_latency(input, layer, device="edge", predictor_dict=predictor_dict)
    # cloud_lat = predict_model_latency(input, layer, device="cloud", predictor_dict=predictor_dict)
    # 获取当前层需要的传输时延
    # 获取当前节点的输入数据在网络上的传输大小
    transport_size = len(pickle.dumps(input))
    # 计算传输速度
    speed = get_speed(network_type=net_type, bandwidth=bandwidth)
    # 计算传输时延
    transmission_lat = transport_size / speed
    # 一层DNN layer可以构建一条边，而构建一条边需要两个顶点
    # dict_input_size_node_name 可以根据输入数据大小构建对应的图顶点
    # 所以可以在执行DNN layer的前后分别构建 start_node以及end_node
    start_node, end_node, record_input = None, None, None
    # 如果当前层是汇入点，也即当前层的输入是一个列表，存储着多个数据的话
    if isinstance(input, list) and layer_index == 18:
        # 初始化layer_out
        layer_out = None
        # 赋值record_input
        record_input = input
        # 对于input列表中的每一个输入来构建
        for one_input in input:
            # 利用输入列表中的单个输入获取到对应的上一层DNN网络的对应表示的节点start_node，一般会在上一层的输出数据时就进行构建，
            # 一般是之前就存在的节点
            vertex_index, start_node = get_node_name(one_input, vertex_index, dict_input_size_node_name)
            # 计算当前网络层的输出
            layer_out = layer(input)
            # 利用当前网络层的输出layer_out来唯一的创建一个节点作为end_node，就算有多个输入但是汇入点的输出只有一个，所以会建立一个汇入点
            vertex_index, end_node = get_node_name(layer_out, vertex_index, dict_input_size_node_name)
            # 例如input是长度为n的列表，则需要构建n个start_node到同一个end_node的n条边
            # 添加从前一个节点到当前层对应的节点的边，每个DNN层会被表示为利用其输出数据创建的唯一节点，也就是说end_node是表示当前层的
            graph.add_edge(start_node, end_node, capacity=transmission_lat)
        # 重新赋值input作为下一层的输入的值
        input = layer_out
    else:
        # 常规构建方式，根据上一层的网络的输出数据（也即当前网络的输入数据）获取到上一层网络节点在DAG图中的表示，作为start_node
        vertex_index, start_node = get_node_name(input, vertex_index, dict_input_size_node_name)
        record_input = input
        # 计算当前层的输入
        input = layer(input)
        # 创建当前层在DAG拓扑图中的根据输出结果创建的唯一标识节点，作为end_node
        vertex_index, end_node = get_node_name(input, vertex_index, dict_input_size_node_name)
        # 避免无效层覆盖原始数据，下面的方式可以过滤掉relu层或dropout层
        if start_node == end_node:
            # relu层或dropout层不需要进行构建
            return vertex_index, input
        # 添加从前一个节点到当前节点的边
        graph.add_edge(start_node, end_node, capacity=transmission_lat)
    print(start_node, end_node)
    # 注意：end_node可以用来在有向图中表示当前层，也就是DNN layer
    # 添加从边缘节点到dnn层的边
    # graph.add_edge(edge_vertex, end_node, capacity=cloud_latency)
    # 添加从dnn层到云端设备的边
    # graph.add_edge(end_node, cloud_vertex, capacity=edge_latency)
    # 记录有向图中的顶点node_name对应的DNN的第几层
    dict_node_layer[end_node] = layer_index + 1
    # 如果record_flag为true的话就记录DNN层中当前层对应的输出
    if record_flag:
        dict_layer_output[layer_index + 1] = input
    # 返回更新后的vertex_index以及当前网络层计算之后的输出
    return vertex_index, input


def graph_construct(model, input, edge_latency_list, cloud_latency_list, bandwidth, net_type="wifi"):
    """
    传入一个DNN模型，construct_digraph_by_model将DNN模型构建成具有相应权重的有向图
    构建过程主要包括三个方面：
    (1) 从边缘设备到DNN层的边，权重设置为云端推理时延
    (2) DNN层之间的边，权重设置为传输时延
    (3) 从DNN层到云端设备的边，权重设置为边端推理时延

    :param model: 传入DNN模型
    :param input: DNN模型的初始输入
    :param edge_latency_list: 边缘设备上各层的推理时延
    :param cloud_latency_list: 云端设备上各层的推理时延
    :param bandwidth: 当前网络时延带宽，可由带宽监视器获取，单位是MB/s
    :param net_type: 当前网络类型，默认为wifi
    :return: 构建好的有向图graph, dict_vertex_layer, dict_layer_input

    由于GoogleNet和ResNet不能用简单地x = layer(x)进行下一步执行
    所以需要自定义新的 get_min_cut_value_for_ResBlock
    所以用户如果有新的DAG结构（1）完善已有创建结构（2）iterable api 需要自定义
    """
    # nx.DiGraph()表示创建一个无多重边有向图
    graph = nx.DiGraph()

    """
    dict_for_input 字典的作用：
        :key是tuple(input.size,input_slice)，字典的键是输入数据的形状以及输入的切片(取输入中的前3个数据)，
            使用数据的形状以及切片可以确定输入是否是同一输入
        :value为与之对应的构建好的有向图中的顶点node_name
    通过dict_for_input可以将DNN layer转化为有向图中的顶点node_name
    原理：对于每一个DNN中的layer其输入数据是唯一的
    """
    dict_input_size_node_name = {}

    """
    dict_vertex_layer字典的作用：
        :key是node_name，也就是有向图中顶点的名称
        :value是该顶点对应原DNN中第几层，也即layer_index
    可以通过有向图的顶点node_name找到其对应原DNN模型中第几层
    注意：
        layer_index = 0 代表初始输入
        layer_index > 0 表示目前顶点代表原DNN层的第layer_index层，若想取出原DNN层应使用model[layer_index-1]
    """
    dict_node_layer = {"v0": 0}  # 初始化v0，对应的原生层是为初始输入

    """
    dict_layer_input以及dict_layer_output 字典的作用：
        :key表示原DNN中第几层，也即layer_index 
        :value表示DNN中第layer_index的层输入以及输出是什么
    第layer_index层的输入与输出，可以使用shape以及前三个元素确定是否为同一输入
    注意：
        layer_index = 0 代表初始输入 
        layer_index > 0 获取的是原模型中model[layer_index-1]层的输入
    """
    # 第0层为初始输入，其输入记录为None，第层不代表任何的神经网络层
    dict_layer_input = {0: None}
    # 第0层为初始输入，其输出即为input
    dict_layer_output = {0: input}
    # 云端设备节点
    cloud_vertex = "cloud"
    # 边缘设备节点
    edge_vertex = "edge"
    # 开始为传入模型搭建DAG拓扑结构
    print(f"start construct graph for model...")
    # 构建模型初始输入v0，在"edge"和"v0"之间建立一条边，capacity是无穷大
    graph.add_edge(edge_vertex, "v0", capacity=inf)
    # 构建图的顶点序号
    vertex_index = 0
    input_bak = input
    # 遍历模型中的所有的层
    for layer_index, layer in enumerate(model):
        # print(layer_index, layer)
        # 对于某一层先检查其输入是否要进行修改
        # 需要更改的层是有DAG拓扑需求的层中分支点和汇总点
        if model.has_dag_topology and (layer_index + 1) in model.dag_dict.keys():
            # 从model.dag_dict[layer_index + 1]中取出分支层的所依赖的前一层的index
            pre_input_cond = model.dag_dict[layer_index + 1]
            # 如果当前层依赖的数据是一个列表，代表当前层有多个输入
            if isinstance(pre_input_cond, list):
                input = []
                # 有多个输入的层对应的是DAG拓扑中的汇合点也就是concat操作，这个的输入应为一个列表
                for pre_index in pre_input_cond:
                    # 当前层的输入也就是对应的列表中的前一层的输出
                    input.append(dict_layer_output[pre_index])
                input[1] = input[1].unsqueeze(0)
                input[2] = input[2].unsqueeze(0)
            else:
                # 当前层的的输入如果不是一个列表的话，就从其他层的输出获得
                input = dict_layer_output[pre_input_cond]
        # if (layer_index == 1):
        #     input = input_bak[0]
        # elif (layer_index == 16):
        #     input = input_bak[1]
        # elif (layer_index == 17):
        #     input = input_bak[2]
        # model.record_output_list中标记的层数对应的DNN层的输出需要被记录
        # record_flag表示当前层的输出需不需要被记录
        record_flag = model.has_dag_topology and (layer_index + 1) in model.record_output_list
        # 根据判断是否需要修改后的input进行有向图中边的构建
        vertex_index, input = add_graph_edge(graph, vertex_index, input, layer_index, layer,
                                             bandwidth, net_type,
                                             edge_latency_list[layer_index], cloud_latency_list[layer_index],
                                             dict_input_size_node_name, dict_node_layer,
                                             dict_layer_input, dict_layer_output, record_flag=record_flag)

    print(dict_input_size_node_name)
    # 利用nx将图绘制出来
    nx.draw(graph, with_labels=graph.nodes)
    # 展示图
    plt.show()
    # 主要负责处理出度大于1的顶点
    prepare_for_partition(graph, vertex_index, dict_node_layer)
    # 返回构建好的有向图、节点及对应的网络层的字典、网络层对应的输入的字典，dict_layer_input没用
    # 利用nx将图绘制出来
    nx.draw(graph, with_labels=graph.nodes)
    # 展示图
    plt.show()
    return graph, dict_node_layer, dict_layer_input


def get_node_name(input, vertex_index, dict_input_size_node_name):
    """
    根据输入input构建对应的顶点名称 node_name
    :param input: 当前层的输入
    :param vertex_index: 顶点编号，即目前应该创建哪个顶点
    :param dict_input_size_node_name: 通过dict_for_input可以将DNN layer转化为有向图中的顶点，对应node_name
    :return: node_name，构建DAG边所需要的首位节点name
    """
    # 获取输入数据相对应的信息 ---> 做一下差异化
    if (isinstance(input, list)):
        input = input[0] + vertex_index
    # print(input)
    len_of_shape = len(input.shape)
    input_shape = str(input.shape)  # 获取当前input的大小
    # print(len_of_shape, input_shape)
    # input_slice用于根据传入数据的唯一性来校验当前DNN层的唯一性
    input_slice = input
    if len_of_shape == 0:
        input_slice = str(input_slice)
    for _ in range(len_of_shape - 1):
        input_slice = input_slice[0]
    # 获取input的前3个数据，保证数据的唯一性
    input_slice = str(input_slice[:3])
    # 如果当前的输入数据唯一标识元组不在dict_input_size_node_name的keys当中的话，就表明当前层没有被构建过
    if (input_shape, input_slice) not in dict_input_size_node_name.keys():
        # 生成当前层的node_name
        node_name = "v" + str(vertex_index)
        # 创建一个新的节点，使用dict_input_size_node_name存储唯一性标识以及对应的层的名称
        dict_input_size_node_name[(input_shape, input_slice)] = node_name
        # 对应的编号+1
        vertex_index += 1
    else:
        # 如果当前层已经存储在了dict_input_size_node_name中的话就表明当前层已经被构建过了
        # 从字典中取出原有节点，保证正确构建有向图
        node_name = dict_input_size_node_name[(input_shape, input_slice)]
    return vertex_index, node_name


def prepare_for_partition(graph, vertex_index, dict_node_layer):
    """
    对根据DNN模型已经构建好的DAG图进行下一步工作：
    1 将出度不为1的顶点记录为start_vex
    2 生成新节点为node_name，从node_name -> start_vex 的边代表传输速度，原来从start vex出发的边改为inf
    3 找到需要删除的边 ：指原终点为start vex的边，将其改成到新节点node name的边
    4 删除cloud和edge到原节点的边

    :param graph : 已经构建好的DAG图
    :param vertex_index : 指定下一个生成的节点编号
    :param dict_node_layer : 记录有向图中的顶点对应的DNN的第几层
    :return:
    """

    # 处理graph-1个顶点指向多个其他顶点的情况
    map_for_vex = []
    # 保存有多个出度的顶点
    multiple_out_vex = []
    # 对每条边进行操作
    for edge in graph.edges.data():
        # 边的起始顶点
        start_vex = edge[0]
        # 边的终止顶点
        end_vex = edge[1]
        # 如果是"edge"到DNN层的边或者DNN层到"cloud"的边，继续循环
        if start_vex == "edge" or end_vex == "cloud":
            continue
        # 如果当前顶点的前置顶点是第一次出现则进行保存
        if start_vex not in map_for_vex:
            map_for_vex.append(start_vex)
        # 如果前置顶点已经在map_for_vex中出现过，再出现的话说明start_vex出度大于1，
        # 将其记录在multiple_out_vex中
        elif start_vex not in multiple_out_vex:
            multiple_out_vex.append(start_vex)
    # 对于multiple_out_vex中的出度大于1的顶点进行操作
    for start_vex in multiple_out_vex:
        # 生成一个新的节点
        node_name = "v" + str(vertex_index)
        # 更新vertex_index
        vertex_index += 1
        # 新节点与原节点start_vex对应相同的DNN layer
        dict_node_layer[node_name] = dict_node_layer[start_vex]
        # 对旧的节点start_vex进行改正
        # 记录需要修改的边，即起点为start_vex的节点，将其修改为inf
        modify_edges = []
        for edge in graph.edges.data():
            # 如果是"edge"到DNN层的边或者DNN层到"cloud"的边，继续循环
            if edge[0] == "edge" or edge[1] == "cloud":
                continue
            # 将起点是start_vex的边放入到modify_edges当中
            if edge[0] == start_vex:
                modify_edges.append(edge)

        # 增加新edge
        for edge in modify_edges:
            # 新增一条从start_vex到新的node_name的边
            graph.add_edge(edge[0], node_name, capacity=edge[2]["capacity"])
            # 新增从新的node_name到edge[1]的边，权重为inf
            graph.add_edge(node_name, edge[1], capacity=inf)
            # 删除原有的边
            graph.remove_edge(edge[0], edge[1])

        # 删除 edge - old node
        # if graph.has_edge("edge", start_vex):
        #     data = graph.get_edge_data("edge", start_vex)["capacity"]
        #     graph.add_edge("edge", node_name, capacity=data)
        #     graph.remove_edge("edge", start_vex)
        # 删除 old node - cloud
        # if graph.has_edge(start_vex, "cloud"):
        #     data = graph.get_edge_data(start_vex, "cloud")["capacity"]
        #     graph.add_edge(node_name, "cloud", capacity=data)
        #     graph.remove_edge(start_vex, "cloud")

    # 简化edge的数值，保留三位小数足够计算
    for edge in graph.edges.data():
        graph.add_edge(edge[0], edge[1], capacity=round(edge[2]["capacity"], 3))
    return vertex_index
