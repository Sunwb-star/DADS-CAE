import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        # 1x1 Convolution Layer
        self.conv1x1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1)

        # 3x3 Convolution Layer
        self.conv3x3 = nn.Conv2d(in_channels, out_channels[1], kernel_size=3, padding=1)

        # 5x5 Convolution Layer
        self.conv5x5 = nn.Conv2d(in_channels, out_channels[2], kernel_size=5, padding=2)

        # Max Pooling Layer
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass through each branch
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(x)
        branch3 = self.conv5x5(x)
        branch4 = self.max_pool(x)

        # Concatenate the outputs along the channel dimension
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return output


# Example usage
input_channels = 3  # Assuming RGB image input
output_channels = [64, 128, 32, 32]  # Number of output channels for each branch
inception_block = InceptionBlock(input_channels, output_channels)

# Dummy input tensor
dummy_input = torch.randn(1, input_channels, 224, 224)

# Forward pass through Inception Block
output = inception_block(dummy_input)
print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
