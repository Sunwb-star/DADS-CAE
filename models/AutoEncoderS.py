from collections import abc

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            # output_size=(input_size - kernel_size + 2 * padding) / stride + 1
            # 传入数据(1, 240, 240)，输出数据是(8, 240, 240)
            nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # 传入数据(8, 240, 240)，输出数据是(8, 120, 120)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            # 传入数据(8, 120, 120)，输出数据是(16, 120, 120)
            # nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # # 输入数据是(16, 120, 120)，输出数据是(16, 60, 60)
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(16),
            # 传入数据(16, 60, 60)，输出数据是(32, 60, 60)
            # nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # # 输入数据是(32, 60, 60)，输出数据是(32, 30, 30)
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(32),
            # 传入数据(32, 30, 30)，输出数据是(8, 30, 30)
            # nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # # 输入数据是(8, 30, 30)，输出数据是(8, 15, 15)
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(8),
            # 传入数据(8, 15, 15)，输出数据是(1, 15, 15)
            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            # Input: (1, 15, 15), Output: (8, 15, 15)
            nn.ConvTranspose2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # Input: (8, 15, 15), Output: (32, 30, 30)
            # nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            # # Input: (32, 30, 30), Output: (16, 60, 60)
            # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(16),
            # # Input: (16, 60, 60), Output: (8, 120, 120)
            # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(8),
            # Input: (8, 120, 120), Output: (1, 240, 240)
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.encoder_len = len(self.encoder)
        self.decoder_len = len(self.decoder)
        self.len = self.encoder_len + self.decoder_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def __iter__(self):
        return SentenceIterator(self.encoder, self.decoder)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        try:
            if item < self.encoder_len:
                layer = self.encoder[item]
            else:
                layer = self.decoder[item - self.encoder_len]
        except IndexError:
            raise StopIteration()
        return layer


class SentenceIterator(abc.Iterator):
    def __init__(self, encoder, decoder):
        self.encoder_part = encoder
        self.decoder_part = decoder
        self._index = 0
        self.encoder_len = len(encoder)
        self.decoder_len = len(decoder)

    def __next__(self):
        try:
            if self._index < self.encoder_len:
                layer = self.encoder_part[self._index]
            else:
                layer = self.decoder_part[self._index - self.encoder_len]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer


if __name__ == '__main__':
    model = Autoencoder()
    print(len(model))
