import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3) ---> (1, 3)，接收的是一个归一化后的pose，其中的x和y的范围都在[-1,1]之间
        `grid_size` 4-tuple (bs, _, grid_h, grid_w) ---> (1, 20，240，240)
        `device` torch.device (cpu or gpu) ---> GPU
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    返回两个仿射变换矩阵，用于局部地图的构建 ----> 在Semantic_Mapping模块中被使用
    """
    # 获取机器人当前归一化后的位置和角度
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    # 将角度转化为弧度
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()
    # 沿一个新维度对输入张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度，
    theta11 = torch.stack([cos_t, -sin_t, torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t, torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device), -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device), torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    # affine_grid()函数是根据仿射变换矩阵，生成该变换相对应的逐像素偏移矩阵；
    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))
    # 返回两个变换矩阵
    return rot_grid, trans_grid


class ChannelPool(nn.MaxPool1d):
    """
        在Semantic_Mapping模块中被使用
    """

    def forward(self, x):
        n, c, w, h = x.size()
        # view()相当于reshape、resize，重新调整Tensor的形状。https://blog.csdn.net/weixin_41377182/article/details/120808310
        x = x.view(n, c, w * h).permute(0, 2, 1)
        # contiguous()函数的作用：把tensor变成在内存中连续分布的形式。
        # 1 transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
        # 所以需要contiguous来返回一个contiguous copy；https://blog.csdn.net/weixin_43332715/article/details/124749348
        x = x.contiguous()
        # max_pool1d()对于一维矩阵进行池化
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        # 重新对张量进行形状调整
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py#L32
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        # nn.Parameter()是 PyTorch 中的一个类，用于创建可训练的参数（权重和偏置），这些参数会在模型训练过程中自动更新
        # 传入的bias的形状是torch.Size([2])，进行升维
        # torch.unsqueeze()函数起到升维的作用，在dim=1的维度上增加一个维度信息，self._bias中包含的内容是：
        # tensor([[0.],
        #         [0.]], requires_grad=True)
        # 形状是(2, 1)，升维以后得dim()变为2
        # 这个self._bias也是一个可以训练的参数
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        # 传入的是一个形状为torch.Size([num_scenes, 2])的torch.zeros(),维度dim()是等于2的
        if x.dim() == 2:
            # .t()意义就是将Tensor进行转置，将self._bias的形状变为torch.Size([1, 2])
            # 在原本的预训练模型中该参数为tensor([[-0.0139, -0.0104]], device='cuda:0', requires_grad=True)
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        # 返回x加上偏置bias的值
        return x + bias


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    # 将矩阵展平，变为0行
    def forward(self, x):
        return x.view(x.size(0), -1)


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L82
class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):

        super(NNBase, self).__init__()
        # input_size(recurrent_input_size)：输入数据X的特征值的数目。
        # hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
        # 传入的参数中：recurrent--->0,recurrent_input_size--->256,hidden_size--->256
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        # 门控循环单元（Gated Recurrent Unit(GRU))
        # 如果设置中关于是否启动循环神经网络为True的话进行下方的GRU网络，但是由于传入的recurrent是0，所以不进行下方的代码
        if recurrent:
            # torch.nn.GRUCell(input_size, hidden_size, bias=True)
            # input_size：输入数据X的特征值的数目。
            # hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
            # bias：默认为 True，如果为 false 则表示神经元不使用偏置
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            # torch.nn.init.orthogonal_(tensor)正交初始化，使得tensor是正交的，
            # weight_ih表示输入到隐藏层的权重，weight_hh表示隐藏层到隐藏层的权重，
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            # bias_ih表示输入到隐藏层的偏置，bias_hh表示隐藏层到隐藏层的偏置。
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    # 使用@property装饰器来创建只读属性，@property装饰器会将方法转换为相同名称的只读属性,可以与所定义的属性配合使用，这样可以防止属性被修改。
    @property
    def is_recurrent(self):
        """
        返回神经网络是否是循环的神经网络，在本代码中不是循环，也即self._recurrent = 0
        """
        return self._recurrent

    # 使用@property装饰器来创建只读属性，@property装饰器会将方法转换为相同名称的只读属性,可以与所定义的属性配合使用，这样可以防止属性被修改。
    @property
    def rec_state_size(self):
        """
        返回循环状态的大小
        如果是使用循环神经网络GRU的话返回self._hidden_size = 256，否则返回1
        """
        if self._recurrent:
            return self._hidden_size
        return 1

    # 使用@property装饰器来创建只读属性，@property装饰器会将方法转换为相同名称的只读属性,可以与所定义的属性配合使用，这样可以防止属性被修改。
    @property
    def output_size(self):
        """
        返回输出状态大小,256
        """
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        """
        如果使用了循环神经网络的话，使用GRU进行计算，但是本代码中self._recurrent = 0，所以不进入该函数
        """
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten 逆展平
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs
