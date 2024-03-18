import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义自动编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义模型和优化器
input_size = 28 * 28  # MNIST图像大小
hidden_size = 64  # 潜在空间的维度
autoencoder = Autoencoder(input_size, hidden_size)
criterion = nn.BCELoss()  # 二元交叉熵损失用于二值化图像
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练自动编码器
num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        images, _ = data
        images = images.view(images.size(0), -1)

        # 前向传播
        outputs = autoencoder(images)

        # 计算损失并反向传播
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印每个epoch的损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试自动编码器
with torch.no_grad():
    test_data = iter(train_loader).next()
    test_images, _ = test_data
    test_images = test_images.view(test_images.size(0), -1)
    reconstructed_images = autoencoder(test_images)

# 将原始图像和重建图像可视化
import matplotlib.pyplot as plt


def plot_images(images, title):
    plt.figure(figsize=(10, 2))
    for i in range(20):
        plt.subplot(2, 10, i + 1)
        plt.imshow(images[i].view(28, 28).numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


plot_images(test_images, 'Original Images')
plot_images(reconstructed_images, 'Reconstructed Images')
