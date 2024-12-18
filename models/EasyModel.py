from collections import abc
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def getBlockIndex(item, accumulate_len):
    """
    通过传入的下标item，提供该item代表的层应该在哪个模块中选择
    :param item: item or index 层的下标，从0开始计数
    :param accumulate_len: 代表各部分累加和的列表
    :return: 对应的模块下标 part_index part_index = 0 代表features 以此类推 part_index = 1 代表inception3
    """
    # 遍历accumulate_len列表
    for part_index in range(len(accumulate_len)):
        # 获取当前branch时的累积长度
        part_len = accumulate_len[part_index]
        # 找到属于哪个branch
        if item < part_len:
            # 返回所在第几个branch当中
            return part_index
    return len(accumulate_len)


class Operation_Concat(nn.Module):
    """
    Operation_Concat 用于最后的拼接操作
    """

    def __init__(self):
        super().__init__()
        self.res = 0

    def forward(self, outputs):
        # 对于输出的outputs列表进行维度拼接
        self.res = torch.cat(outputs, 1)
        return self.res


class EasyModel(nn.Module):
    """
    构建一个InceptionBlock结构：是一个DAG拓扑结构的模型
    """

    def __init__(self, in_channels: int = 3) -> None:
        super(EasyModel, self).__init__()
        # 使用一个nn.Sequential构造一个单独的卷积层，使用的kernel_size是正方形卷积7*7，步幅是2*2
        self.preInference = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(7, 7), stride=(2, 2))
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1)
        )
        # 用于最后的拼接工作的模块
        self.concat = Operation_Concat()
        # 将三层的branch放入一个列表中进行迭代
        self.branch_list = [self.preInference, self.branch1, self.branch2]
        # 记录累积的长度
        self.accumulate_len = []
        # 对于self.branch_list的列表长度进行
        for i in range(len(self.branch_list)):
            # 如果是branch列表中的第一个Sequential的话就直接添加该Sequential内部的layer的长度到累计长度当中
            if i == 0:
                self.accumulate_len.append(len(self.branch_list[i]))
            # 如果不是branch列表中的第一个Sequential的话，就将前一个branch的数值加上当前branch的长度作为累积长度
            else:
                self.accumulate_len.append(self.accumulate_len[i - 1] + len(self.branch_list[i]))

        # 如果是DAG拓扑结构需要自己设计好下面几个设定
        # self.has_dag_topology表示该网络需要变成DAG的拓扑结构
        self.has_dag_topology = True
        # self.record_output_list记录整个模型中的哪几层需要保存输出，也就是说在每一个branch的结尾处需要保存输出
        self.record_output_list = [self.accumulate_len[0], self.accumulate_len[1], self.accumulate_len[2]]
        # 定义DAG拓扑相关层的输入
        # 这样的字典结构可能用于描述一个有向无环图（DAG）中节点之间的依赖关系或某种范围映射。
        # 键表示范围的上界，值表示范围的下界或依赖节点。在这个特定的例子中，这样的映射可能在某种图形算法或数据结构中发挥作用。
        # 最终self.accumulate_len的内容是[1, 2, 3]
        # self.dag_dict的内容是{2: 1, 3: 1, 4: [2, 3]}，这样就像是一个DAG拓扑结构，是一种总分总的结构，
        # 每一个数值标识模型中唯一的一层网络，表示第二层依赖于第一层，第三层依赖于第一层，第四层依赖于第二层和第三层
        self.dag_dict = {
            self.accumulate_len[0] + 1: self.accumulate_len[0],
            self.accumulate_len[1] + 1: self.accumulate_len[0],
            self.accumulate_len[2] + 1: [self.accumulate_len[1], self.accumulate_len[2], ],
        }

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1, branch2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        # 相当于传入了原始数据到网络中来，self.preInference是第一个“总”
        x = self.preInference(x)
        # 进行两个“分”的计算得到一个输出列表
        outputs = self._forward(x)
        # 最后将这个分离的tensor列表进行合并
        return self.concat(outputs)

    def __len__(self):
        # 最终的长度等于self.accumulate_len累积长度总长度+1
        return self.accumulate_len[-1] + 1

    def __getitem__(self, item):
        # 如果超出模型整体的累积长度范围的话，就停止迭代
        if item >= self.accumulate_len[-1] + 1:
            raise StopIteration()

        # 根据传入的item推断出所在的Block的位置
        part_index = getBlockIndex(item, self.accumulate_len)
        # 直接从第一个branch当中去除DNN层
        if part_index == 0:
            layer = self.branch_list[part_index][item]
        # 如果还能在self.branch_list当中找到这个index对应的值的话就获取到该层
        elif part_index < len(self.accumulate_len):
            layer = self.branch_list[part_index][item - self.accumulate_len[part_index - 1]]
        else:
            layer = self.concat
        return layer

    def __iter__(self):
        return Inception_SentenceIterator(self.branch_list, self.concat, self.accumulate_len)


class Inception_SentenceIterator(abc.Iterator):
    def __init__(self, branch_list, concat, accumulate_len):
        # branch对应的列表
        self.branch_list = branch_list
        # 累积长度列表
        self.accumulate_len = accumulate_len
        # 最后用于拼接整个tensor的layer
        self.concat = concat
        # 迭代时用于标记当前层数的index
        self._index = 0

    def __next__(self):
        # 如果超出范围，就停止迭代
        if self._index >= self.accumulate_len[-1] + 1:
            raise StopIteration()
        # 根据传入的item取出正确的DNN层
        part_index = getBlockIndex(self._index, self.accumulate_len)
        # 直接从第一个branch当中去除DNN层
        if part_index == 0:
            layer = self.branch_list[part_index][self._index]
        # 如果还能在self.branch_list当中找到这个index对应的值的话就获取到该层
        elif part_index < len(self.accumulate_len):
            layer = self.branch_list[part_index][self._index - self.accumulate_len[part_index - 1]]
        else:
            layer = self.concat
        # 每迭代一次就将self._index的数值+1
        self._index += 1
        return layer


class easy_dag_part(nn.Module):
    def __init__(self, branches):
        super(easy_dag_part, self).__init__()
        # self.branch1表示分支网络1
        self.branch1 = branches[0]
        # self.branch2表示分支网络2
        self.branch2 = branches[1]
        # self.concat表示最后的合并网络Operation_Concat
        self.concat = Operation_Concat()

    def forward(self, x):
        # 两个分支分别根据传入的数据进行计算
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        # 两个的输出放入到一个列表当中进行维度合并生成一个新的特征数据作为输出结果
        outputs = [branch1, branch2]
        return self.concat(outputs)


class EdgeInception(nn.Module):
    """
    edge Inception 用于构建划分好的边端Inception
    """

    def __init__(self, edge_branches):
        super(EdgeInception, self).__init__()
        # 边端模型部分的两个分支
        self.branch1 = edge_branches[0]
        self.branch2 = edge_branches[1]

    def forward(self, x):
        # 边端中模型的分支分别进行计算结果
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        # 作为中间结果进行输出
        outputs = [branch1, branch2]
        return outputs


class CloudInception(nn.Module):
    """
    cloud Inception 用于构建划分好的云端Inception
    """

    def __init__(self, cloud_branches):
        super(CloudInception, self).__init__()
        # 云端会有一个用来合并最终输出结果的层self.concat
        self.branch1 = cloud_branches[0]
        self.branch2 = cloud_branches[1]
        self.concat = Operation_Concat()

    def forward(self, x):
        # 在两条分支上分别进行继续推理
        branch1 = self.branch1(x[0])
        branch2 = self.branch2(x[1])
        outputs = [branch1, branch2]
        # 合并最终结果
        return self.concat(outputs)


def construct_edge_cloud_inception_block(model: EasyModel, model_partition_edge: list):
    """
    构建Inception的边端模型和云端模型
    :param model: 传入一个需要划分的Inception block
    :param model_partition_edge: Inception的划分点 (start_layer, end_layer)，也就是需要被割断的那条边
    :return: edge_Inception,cloud_Inception
    """
    # 模型的累积长度，也就是所有的网络层的总长度
    accumulate_len = model.accumulate_len
    # 分别初始化边端的网络模型和云端模型
    edge_model, cloud_model = nn.Sequential(), nn.Sequential()
    # 如果只有一个地方需要划分的话
    if len(model_partition_edge) == 1:
        # 提取划分点
        partition_point = model_partition_edge[0][0]
        # 因为分割点只有一个的话，那就只会出现在第一个分支点出现之前，所以就断言分割点的范围如下
        assert partition_point <= accumulate_len[0] + 1
        idx = 1
        for layer in model:
            # 如果index大于第一段累积长度的话直接跳出
            if idx > accumulate_len[0]:
                break
            # 将分支之前的总的网络枝干进行边缘端和云端分割并分别存入到edge_model和cloud_model当中
            if idx <= partition_point:
                edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            else:
                cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            idx += 1
        # 利用easy_dag_part构建分支之后的所有分支网络
        layer = easy_dag_part(model.branch_list[1:])
        # 将后面的分支网络作为一个整体来加入到cloud当中
        cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
    else:
        # 如果model_partition_edge的长度是2的话就代表需要在几个个branch之间进行划分
        assert len(model_partition_edge) == 2
        # 抽离出分支网络
        branches = model.branch_list[1:]
        # 首先在edge模型端加入最开始的总的分支网络部分
        edge_model.add_module(f"1-preInference", model.preInference)
        # 对边缘端的分支和云端的分支分别进行列表存储
        edge_branches = []
        cloud_branches = []
        # 对于model_partition_edge中存储的需要被分割的边，分别进行操作
        for edge in model_partition_edge:
            # 初始化边端和云端的模型分支
            edge_branch = nn.Sequential()
            cloud_branch = nn.Sequential()
            # 找到要分割的边所在的branch块传入给block，tmp_point是用来记录分割点在对应的block块中的层数的
            block, tmp_point = None, None
            # 如果当前这条edge中的起始网络层或者终止网络层在第一个branch块的范围的话，就将分割点设定为第一个branch
            if (edge[0] in range(accumulate_len[0] + 1, accumulate_len[1] + 1) or
                    edge[1] in range(accumulate_len[0] + 1, accumulate_len[1] + 1)):
                block = branches[0]
                tmp_point = edge[0] - accumulate_len[0]
            # 否则就将分割点设定为第二个branch，计算出要分割的点
            elif (edge[0] in range(accumulate_len[1] + 1, accumulate_len[2] + 1) or
                  edge[1] in range(accumulate_len[1] + 1, accumulate_len[2] + 1)):
                block = branches[1]
                tmp_point = edge[0] - accumulate_len[1]
            # 对block对应的branch进行分割，分别存入到edge_branch和cloud_branch中
            idx = 1
            for layer in block:
                if idx <= tmp_point:
                    edge_branch.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                else:
                    cloud_branch.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                idx += 1
            # 在edge_branches和cloud_branches中分别进行存储
            edge_branches.append(edge_branch)
            cloud_branches.append(cloud_branch)

        # 最终在不同的branch上进行了不同的分割点分割。
        # 使用edge_branches以及cloud_branches构建EdgeInception以及CloudInception两个类
        edge_Inception = EdgeInception(edge_branches)
        cloud_Inception = CloudInception(cloud_branches)
        # 将分割好的模型分别放入到对应的边端上
        edge_model.add_module(f"2-edge-inception", edge_Inception)
        cloud_model.add_module(f"1-cloud-inception", cloud_Inception)
    return edge_model, cloud_model


if __name__ == "__main__":
    model = EasyModel()
    model.to("cpu")
    print(model.accumulate_len)
    print(model.record_output_list)
    print(model.dag_dict)
    print(len(model))
