from collections import abc

import torch
import torch.nn as nn


# 做长期目标决策的类
class Sem_Exp(nn.Module):
    def __init__(self, hidden_size=256, num_sem_categories=16):
        super(Sem_Exp, self).__init__()
        # input_shape的大小是(1, 24, 240, 240)
        # input_shape[1]的值是local_w:240，input_shape[2]的值是local_h:240
        out_size = int(240 / 16.0) * int(240 / 16.0)
        # 构建神经网络
        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 定义一个两个线性层，包含一个隐藏层，隐藏层神经元数量是256，输入中的8 * 2是两个emb输入
        self.classifier = nn.Sequential(
            nn.Linear(out_size * 32 + 8 * 2, hidden_size),
            # nn.Linear(out_size * 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        # 在训练过程中，需要用到词嵌入，而torch.nn.Embedding就提供了这样的功能。
        # 我们只需要初始化torch.nn.Embedding(num_embeddings,embedding_dim)，n是单词数，m就是词向量的维度
        # num_embeddings：词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
        # embedding_dim：嵌入向量的维度，即用多少维的词向量来表示一个符号
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        # Sets the module in training mode.设置该模块为训练模式
        self.train()

        self.len = len(self.main)

    # 前向传播函数
    def forward(self, inputs):
        # 将传入的数据通过self.main神经网络
        x = self.main(inputs[0])
        # 其为一个简单的存储固定大小的词典的嵌入向量的查找表，意思就是说，给一个编号，
        # 嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。
        # 生成一个表示空间信息的8维度的特征向量
        orientation_emb = self.orientation_emb(inputs[1])
        # 生成一个可以表示目标信息的8维度的特征向量
        goal_emb = self.goal_emb(inputs[2])
        # 将上面三个tensor在dim=1的维度上进行拼接
        x = torch.cat((x, orientation_emb, goal_emb), 1)
        y = self.classifier(x)
        # 返回三个参数，.squeeze(-1)用来删除最后一个维度
        return y

    def __iter__(self):
        """用于遍历AlexNet模型的每一层"""
        return SentenceIterator(self.main)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        layer = nn.Sequential()
        try:
            if item < self.len:
                layer = self.main[item]
        except IndexError:
            raise StopIteration()
        return layer


class SentenceIterator(abc.Iterator):
    """
    AlexNet迭代器
    下面是AlexNet网络的迭代参数调整
    将下面的设置传入到AlexNet的 __iter__ 中可以完成对于AlexNet网络的层级遍历
    """

    def __init__(self, net):
        self.features = net
        self._index = 0
        self.len = len(net)

    def __next__(self):
        layer = nn.Sequential()
        try:
            if self._index <= self.len:
                layer = self.features[self._index]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer


if __name__ == "__main__":
    model = Sem_Exp()
    # print(model.state_dict())
    for name in model.state_dict():
        print(name)
    edge_model = nn.Sequential()
    cloud_model = nn.Sequential()
    index = 3
    idx = 1
    for layer in model:
        if idx <= index:
            edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        else:
            cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        idx += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 把模型转为test模式
    model.eval()

    # step2 准备要传送的input数据
    input_0 = torch.rand(size=(1, 24, 240, 240), requires_grad=False).to(device)
    orientation = torch.tensor(30).unsqueeze(0).to(device)
    goal_index = torch.tensor(0).unsqueeze(0).to(device)
    input = [input_0, orientation, goal_index]
    output = model(input)
    prob = nn.Sigmoid()(output)
    print("概率：", prob)
    print(len(model))
