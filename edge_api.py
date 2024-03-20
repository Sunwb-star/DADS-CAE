import getopt
import multiprocessing
import sys
import warnings

import h5py
import numpy
import torch

from net.monitor_client import MonitorClient
from server_func import start_client_linear
from server_func import start_client_sem

warnings.filterwarnings("ignore")

"""
    边缘设备api，用于启动边缘设备，进行前半部分计算后，将中间数据传递给云端设备
    client 启动指令 python edge_api.py -i 127.0.0.1 -p 9999 -d cpu -t easy_net
    "-t", "--type"          模型种类参数 "alex_net" "vgg_net" "easy_net" "inception" "inception_v2"
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-d", "--device"        是否开启客户端GPU计算 cpu or cuda
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:i:p:d:", ["type=", "ip=", "port=", "device_on="])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    model_type = ""
    ip, port = "127.0.0.1", 999
    device = "cpu"
    for opt, arg in opts:
        if opt in ("-t", "--type"):
            model_type = arg
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-d", "--device"):
            device = arg

    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("本机器上不可以使用cuda")

    # 开启：带宽监测客户端
    # 如果没有两个设备测试的条件 可以使用下面的方式 将带宽自定义
    # bandwidth_value = 10  #Mbps
    bandwidth_value = multiprocessing.Value('d', 0.0)
    monitor_cli = MonitorClient(ip=ip, bandwidth_value=bandwidth_value)
    monitor_cli.start()
    # 等待子进程结束后获取到带宽数据
    monitor_cli.join()

    # 根据带宽和其他的一些信息来选择走哪一条分支
    print(f"get bandwidth value : {bandwidth_value.value} MB/s")
    # if (bandwidth_value.value < 10):
    #     model_type = "AutoEncoderConv"
    # else:
    #     model_type = "Sem_Exp"
    # step2 准备input数据
    hdf5_file_path = "models/autoencoder_data.h5"
    with h5py.File(hdf5_file_path, 'r') as f:
        # 读取HDF5文件中的数据集
        loaded_tensor = torch.tensor(numpy.array(f['autoencoder_datasets']))
        input_data = loaded_tensor[0].unsqueeze(0)

    input_data = input_data.to(device)
    orientation = torch.tensor(30).to(device)
    goal_index = torch.tensor(0).to(device)
    # 部署阶段 - 选择优化分层点
    upload_bandwidth = bandwidth_value.value  # MBps
    # MBps 为确保程序正确运行 这里设置为10；实机运行使用上面那行
    # upload_bandwidth = 10
    # 使用云边协同的方式进行模拟，传入相关的参数进行设置
    if model_type == "Sem_Exp":
        # 传入的参数input_data的形状是(1, 24, 240, 240)
        input_x = [input_data, orientation, goal_index]
        partition_point = 15
        start_client_sem(ip, port, input_x, model_type, partition_point, device)
    elif model_type == "AutoEncoderConv":
        # input_data = input_data.view(-1, 1, 240, 240)
        partition_point = 7
        # plt.figure()
        # plt.imshow(input_data[0, 1, :, :].cpu().detach().numpy())
        # plt.show()
        # 传入的数据形状是(1, 24, 240, 240)
        start_client_linear(ip, port, input_data, model_type, partition_point, device, orientation, goal_index)
