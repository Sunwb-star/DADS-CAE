import torch

# host = 'www.baidu.com'
# delay = ping(host)
# if delay is not None:
#     print('Ping %s 延迟为 %0.2f ms' % (host, delay))
# else:
#     print('Ping %s 失败' % host)
#
# hdf5_file_path = "../models/autoencoder_data.h5"
# with h5py.File(hdf5_file_path, 'r') as f:
#     # 读取HDF5文件中的数据集
#     loaded_tensor = torch.tensor(numpy.array(f['autoencoder_datasets']))
#     input_data = loaded_tensor[0].unsqueeze(0)
#     print(input_data.shape)
#     # 创建一个包含多个子图的画布
#     fig, axes = plt.subplots(1, 2)
#     # 将每个矩阵放入对应的子图中
#     for i in range(2):
#         axes[0].imshow(input_data[0, 0, :, :].numpy())
#         axes[1].imshow(input_data[0, 1, :, :].numpy())
#     # 调整布局，以免子图重叠
#     plt.tight_layout()
#     # 显示图像
#     plt.show()
state_dict = torch.load('../models/sem_exp_origin.pth')
print(state_dict.keys())
del state_dict['dist.fc_mean.bias']
del state_dict['dist.logstd._bias']
del state_dict['dist.fc_mean.weight']
del state_dict['network.critic_linear.weight']
del state_dict['network.critic_linear.bias']
state_dict['network.second.0.weight'] = state_dict.pop('network.linear1.weight')
state_dict['network.second.0.bias'] = state_dict.pop('network.linear1.bias')
state_dict['network.second.2.weight'] = state_dict.pop('network.linear2.weight')
state_dict['network.second.2.bias'] = state_dict.pop('network.linear2.bias')
print(state_dict.keys())
