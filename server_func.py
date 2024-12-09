import torch

import net.net_utils as net
from dads_framework.dads import algorithm_DSL, get_partition_points
from dads_framework.graph_construct import get_layers_latency
from models import SemExpModel
from utils import inference_utils


def start_server(socket_server, device):
    """
    开始监听客户端传来的消息
    一般仅在 cloud_api.py 中直接调用
    :param socket_server: socket服务端
    :param device: 使用本地的cpu运行还是cuda运行
    :return: None
    """
    # 等待客户端连接
    conn, client = net.wait_client(socket_server)
    # 接收模型类型
    model_type = net.get_short_data(conn)
    print(f"get model type successfully.")
    if model_type == "AutoEncoderConv":
        start_server_linear(conn, client, model_type, device)
    elif model_type == "Sem_Exp":
        start_server_sem(conn, client, model_type, device)
    else:
        # 根据模型的类型读取和生成模型实例
        model = inference_utils.get_dnn_model(model_type)
        # 获取云端设备的各层时延
        cloud_latency_list = get_layers_latency(model, device=device)
        # 发送云端设备的时延信息到边缘端
        net.send_short_data(conn, cloud_latency_list, "model latency on the cloud device.")
        # 接收边缘设备计算出来的模型分层点
        model_partition_edge = net.get_short_data(conn)
        print(f"get partition point successfully.")
        # 获取划分后的边缘端模型和云端模型
        _, cloud_model = inference_utils.model_partition(model, model_partition_edge)
        # 将分割后的云端部分的模型传入到device中
        cloud_model = cloud_model.to(device)
        # 接收边缘端设备传输的中间数据并返回传输时延
        edge_output, transfer_latency = net.get_data(conn)
        # 避免连续发送两个消息 防止消息粘包
        conn.recv(40)
        print(f"get edge_output and transfer latency successfully.")
        # 发送传输时延
        net.send_short_data(conn, transfer_latency, "transfer latency")
        # 避免连续发送两个消息 防止消息粘包
        conn.recv(40)
        # 进行设备预热，保持数据的一致性
        inference_utils.warmUp(cloud_model, edge_output, device)
        # 记录云端推理时延
        cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output, device, epoch_cpu=30,
                                                                 epoch_gpu=100)
        print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
        net.send_short_data(conn, cloud_latency, "cloud latency")
        # print(cloud_output.shape)
        print("================= DNN Collaborative Inference Finished. ===================")


def start_client(ip, port, input_x, model_type, upload_bandwidth, device):
    """
    启动一个client客户端 向server端发起推理请求
    一般仅在 edge_api.py 中直接调用
    :param ip: server端的ip地址
    :param port: server端的端口地址
    :param input_x: 初始输入
    :param model_type: 选用的模型类型
    :param upload_bandwidth 上传带宽
    :param device: 在本地cpu运行还是cuda运行
    :return: None
    """
    # 读取模型
    model = inference_utils.get_dnn_model(model_type)
    # 和云端建立连接
    conn = net.get_socket_client(ip, port)
    # 发送一个model_type数据请求云端的各层推理时延
    net.send_short_data(conn, model_type, msg="model type")
    # 计算出边缘端的时延参数
    edge_latency_list = get_layers_latency(model, device=device)
    # 接受到云端的时延参数
    cloud_latency_list = net.get_short_data(conn)
    # 获得图中的割集以及dict_node_layer字典
    graph_partition_edge, dict_node_layer = algorithm_DSL(model, input_x,
                                                          edge_latency_list, cloud_latency_list,
                                                          bandwidth=upload_bandwidth)
    # 获得在DNN模型哪层之后划分
    model_partition_edge = get_partition_points(graph_partition_edge, dict_node_layer)
    print(f"partition edges : {model_partition_edge}")
    # 发送划分点
    net.send_short_data(conn, model_partition_edge, msg="partition strategy")
    # 获取划分后的边缘端模型和云端模型
    edge_model, _ = inference_utils.model_partition(model, model_partition_edge)
    # 将模型转移到device中
    edge_model = edge_model.to(device)
    # 开始边缘端的推理 首先进行预热
    inference_utils.warmUp(edge_model, input_x, device)
    # 获取边缘端延迟以及边缘端数据
    edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x, device, epoch_cpu=30, epoch_gpu=100)
    print(f"{model_type} 在边缘端设备上推理完成 - {edge_latency:.3f} ms")
    # 发送边缘端推理得到的中间数据
    net.send_data(conn, edge_output, "edge output")
    # 避免连续接收两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())
    transfer_latency = net.get_short_data(conn)
    print(f"{model_type} 传输完成 - {transfer_latency:.3f} ms")
    # 避免连续接收两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())
    cloud_latency = net.get_short_data(conn)
    # 打印推理完成的消息
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
    print("================= DNN Collaborative Inference Finished. ===================")
    conn.close()


def start_client_linear(ip, port, input_x, model_type, partition_point, device, orientation, goal_index):
    """
    启动一个client客户端 向server端发起推理请求
    一般仅在 edge_api.py 中直接调用
    :param ip: server端的ip地址
    :param port: server端的端口地址
    :param model_type: 选用的模型类型
    :param input_x: 初始输入
    :param partition_point 模型划分点
    :param device: 在本地cpu运行还是cuda运行
    :return: None
    """

    conn = net.get_socket_client(ip, port)
    # 发送模型类型
    net.send_short_data(conn, model_type, msg="model type")
    # 读取模型
    model = inference_utils.get_dnn_model(model_type)
    if model_type == "AutoEncoderConv":
        model.load_state_dict(torch.load('pretrained_model_pths/autoencoder_model.pth'))
    model.eval()
    # 发送划分点
    net.send_short_data(conn, partition_point, msg="partition strategy")
    edge_model, _ = inference_utils.model_partition_linear(model, partition_point)
    edge_model = edge_model.to(device)
    edge_model.eval()

    # 传入数据的shape是(1, 24, 240, 240)
    last_map = input_x[0, -1, :, :].unsqueeze(0).unsqueeze(0)
    # 形状变为(24, 1, 240, 240)
    input_x = input_x.view(-1, 1, 240, 240)
    # 开始边缘端的推理，首先进行预热，传入的数据维度是(1, 1, 240, 240)
    inference_utils.warmUp(edge_model, input_x[0].unsqueeze(0), device)
    # 传入前23个地图进行推理，然后获取结果
    # edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x[:23], device, epoch_cpu=30,
    #                                                        epoch_gpu=100)
    edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x, device, epoch_cpu=30,
                                                           epoch_gpu=100)
    # edge_output_list = [edge_output, last_map]
    edge_output_list = [edge_output, orientation, goal_index]
    print(f"{model_type} 在边缘端设备上推理完成 - {edge_latency:.3f} ms")
    flag = net.get_short_data(conn)
    # 发送中间数据
    # net.send_data(conn, edge_output, "edge output")
    net.send_data(conn, edge_output_list, "edge output")
    # 连续发送两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())
    transfer_latency = net.get_short_data(conn)
    print(f"{model_type} 传输完成 - {transfer_latency:.3f} ms")
    # 连续发送两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())
    # 发送方向和目标等信息
    # extra_list = [orientation, goal_index]
    # net.send_short_data(conn, extra_list, "edge output")
    cloud_latency = net.get_short_data(conn)
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
    print("================= DNN Collaborative Inference Finished. ===================")
    conn.close()


def start_server_linear(_conn, _client, _model_type, device):
    """
    开始监听客户端传来的消息
    一般仅在 cloud_api.py 中直接调用
    :param _model_type: 模型的类型
    :param _conn: socket连接
    :param _client: TCP连接的客户端
    :param device: 使用本地的cpu运行还是cuda运行
    :return: None
    """
    # 等待客户端连接
    conn, client = _conn, _client
    # 接收模型类型
    model_type = _model_type
    # print(f"get model type successfully.")
    # 读取模型
    model = inference_utils.get_dnn_model(model_type)
    if model_type == "AutoEncoderConv":
        model.load_state_dict(torch.load('pretrained_model_pths/autoencoder_model.pth'))
    model.eval()
    # 接收模型分层点
    partition_point = net.get_short_data(conn)
    print(f"get partition point successfully.")
    flag = 1
    net.send_short_data(conn, flag, "flag")
    _, cloud_model = inference_utils.model_partition_linear(model, partition_point)
    cloud_model = cloud_model.to(device)
    cloud_model.eval()
    # 接收中间数据并返回传输时延
    # 结束到的edge_output是一个list，第一个数据的形状是(23,1,15,15)，第二个形状是(1,1,240,240)
    edge_output, transfer_latency = net.get_data(conn)
    print(edge_output)
    # print(edge_output[0].shape)
    # 连续接收两个消息 防止消息粘包
    conn.recv(40)
    print(f"get edge_output and transfer latency successfully.")
    net.send_short_data(conn, transfer_latency, "transfer latency")
    # 连续接收两个消息 防止消息粘包
    conn.recv(40)
    # 进行预热
    print(edge_output[0].shape)
    inference_utils.warmUp(cloud_model, edge_output[0][0].unsqueeze(0), device)
    # 记录云端推理时延
    cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output[0], device, epoch_cpu=30,
                                                             epoch_gpu=100)

    print("================= Start Navigation Decision Evaluating ===================")
    # 得到的推理结果形状是(23, 1, 240, 240)，进行重新变换
    encoder_decoder_data = cloud_output.view(1, 24, 240, 240)
    # edge_output[0] = cloud_output
    # encoder_decoder_data = torch.cat(edge_output, dim=1)
    # print(encoder_decoder_data.shape)
    # 可视化一下
    # plt.figure()
    # plt.imshow(encoder_decoder_data[0, 3, :, :].cpu().detach().numpy())
    # plt.show()
    # print(encoder_decoder_data[0, -1].max(), encoder_decoder_data[0, -1].min())
    # 获取方向和目标Index信息
    # extra_list = net.get_short_data(conn)
    orientation = edge_output[1]
    goal_index = edge_output[2]
    # 开始记录时延信息
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for i in range(2):
        # 首先进行二值化处理，得到新的地图(1,24,240,240)
        threshold = 0.25
        encoder_decoder_data[0, :24] = (encoder_decoder_data[0, :24] > threshold).float()
        # plt.figure()
        # plt.imshow(encoder_decoder_data[0, 1, :, :].cpu().detach().numpy())
        # plt.show()
        input_data = [encoder_decoder_data, orientation, goal_index]
        sem_exp_model = SemExpModel.Sem_Exp(input_shape=(1, 24, 240, 240)).to(device)
        output = sem_exp_model(input_data)
        print(output)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    all_time = curr_time / 2
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency + all_time:.3f} ms")
    net.send_short_data(conn, cloud_latency + all_time, "cloud latency")
    print("================= DNN Collaborative Inference Finished. ===================")


def start_client_sem(ip, port, input_x, model_type, partition_point, device):
    """
    启动一个client客户端 向server端发起推理请求
    一般仅在 edge_api.py 中直接调用
    :param ip: server端的ip地址
    :param port: server端的端口地址
    :param model_type: 选用的模型类型
    :param input_x: 初始输入
    :param partition_point 模型划分点
    :param device: 在本地cpu运行还是cuda运行
    :return: None
    """

    conn = net.get_socket_client(ip, port)
    # 发送模型类型
    net.send_short_data(conn, model_type, msg="model type")
    # 读取模型
    model = inference_utils.get_dnn_model(model_type)
    # if model_type == "AutoEncoderConv":
    #     model.load_state_dict(torch.load('models/autoencoder_model.pth'))
    model.eval()
    # 发送划分点
    net.send_short_data(conn, partition_point, msg="partition strategy")
    edge_model, _ = inference_utils.model_partition_linear(model, partition_point)
    edge_model = edge_model.to(device)
    edge_model.eval()
    # 传入数据的shape是(1, 24, 240, 240)
    # 开始边缘端的推理，首先进行预热，传入的数据维度是(1, 24, 240, 240)
    inference_utils.warmUp(edge_model, input_x[0], device)
    edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x[0], device, epoch_cpu=30,
                                                           epoch_gpu=100)

    edge_output_list = [edge_output, input_x[1], input_x[2]]
    print(f"{model_type} 在边缘端设备上推理完成 - {edge_latency:.3f} ms")
    flag = net.get_short_data(conn)
    # 发送中间数据
    # net.send_data(conn, edge_output, "edge output")
    net.send_data(conn, edge_output_list, "edge output")
    # 连续发送两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())
    transfer_latency = net.get_short_data(conn)
    print(f"{model_type} 传输完成 - {transfer_latency:.3f} ms")
    # 连续发送两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())
    cloud_latency = net.get_short_data(conn)
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
    print("================= DNN Collaborative Inference Finished. ===================")
    conn.close()


def start_server_sem(_conn, _client, _model_type, device):
    """
    开始监听客户端传来的消息
    一般仅在 cloud_api.py 中直接调用
    :param _model_type: 模型的类型
    :param _conn: socket连接
    :param _client: TCP连接的客户端
    :param device: 使用本地的cpu运行还是cuda运行
    :return: None
    """
    # 等待客户端连接
    conn, client = _conn, _client
    # 接收模型类型
    model_type = _model_type
    # print(f"get model type successfully.")
    # 读取模型
    model = inference_utils.get_dnn_model(model_type)
    # if model_type == "AutoEncoderConv":
    #     model.load_state_dict(torch.load('models/autoencoder_model.pth'))
    model.eval()
    # 接收模型分层点
    partition_point = net.get_short_data(conn)
    print(f"get partition point successfully.")
    flag = 1
    net.send_short_data(conn, flag, "flag")
    _, cloud_model = inference_utils.model_partition_linear(model, partition_point)
    cloud_model = cloud_model.to(device)
    cloud_model.eval()
    # 接收中间数据并返回传输时延
    # 结束到的edge_output是一个list，第一个数据的形状是(23,1,15,15)，第二个形状是(1,1,240,240)
    edge_output, transfer_latency = net.get_data(conn)
    print(edge_output)
    # print(edge_output[0].shape)
    # 连续接收两个消息 防止消息粘包
    conn.recv(40)
    print(f"get edge_output and transfer latency successfully.")
    net.send_short_data(conn, transfer_latency, "transfer latency")
    # 连续接收两个消息 防止消息粘包
    conn.recv(40)
    # 进行预热
    inference_utils.warmUp(cloud_model, edge_output[0].to(device), device)
    # 记录云端推理时延
    cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output[0].to(device), device,
                                                             epoch_cpu=30,
                                                             epoch_gpu=100)
    print("================= Start Navigation Decision Evaluating ===================")
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for i in range(30):
        orientation = edge_output[1].to(device)
        goal_index = edge_output[2].to(device)
        model.orientation_emb.to(device)
        model.goal_emb.to(device)
        model.classifier.to(device)
        orientation_emb = model.orientation_emb(orientation)
        goal_index_emb = model.goal_emb(goal_index)
        input_data = [cloud_output, orientation_emb.unsqueeze(0), goal_index_emb.unsqueeze(0)]
        input_data = torch.cat(input_data, dim=1).to(device)
        output = model.classifier(input_data)
        print(output)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    # all_time += curr_time
    all_time = curr_time / 30
    # 可视化一下
    # plt.figure()
    # plt.imshow(encoder_decoder_data[0, 3, :, :].cpu().detach().numpy())
    # plt.show()
    # print(encoder_decoder_data[0, -1].max(), encoder_decoder_data[0, -1].min())
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency + all_time:.3f} ms")
    net.send_short_data(conn, cloud_latency + all_time, "cloud latency")
    print("================= DNN Collaborative Inference Finished. ===================")
