import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


## Base Model
class MiniCNN_1(nn.Module):
    def __init__(self):
        super(MiniCNN_1, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 28,1>28,8|RF:3,J:1
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 28,8>28,8|RF:5,J:1
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 28,8>28,8|RF:7,J:1
            nn.ReLU(),
        )
        self.pool_1 = nn.MaxPool2d(2, 2)  # 28,16>14,16|RF:8,J:2
        self.transition_1 = nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=(1, 1), padding=0, bias=False
        )  # 14,8>14,4|RF:8,J:2
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 14,4>14,8|RF:12,J:2
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 14,8>14,16|RF:16,J:2
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 14,16>14,16|RF:20,J:2
            nn.ReLU(),
        )
        self.pool_2 = nn.MaxPool2d(2, 2)  # 14,16>7,16|RF:21,J:4
        self.transition_2 = nn.Conv2d(
            in_channels=16, out_channels=4, kernel_size=(1, 1), padding=0, bias=False
        )  # 14,16>7,4|RF:21,J:4
        self.fc = nn.Linear(4 * 7 * 7, 10)  # 7,4>10

    def forward(self, x):
        x = self.block_1(x)
        x = self.pool_1(x)
        x = self.transition_1(x)
        x = self.block_2(x)
        x = self.pool_2(x)
        x = self.transition_2(x)
        x = x.view(-1, 4 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


## Increasing number of Params
class MiniCNN_2(nn.Module):
    def __init__(self):
        super(MiniCNN_2, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 28,1>28,8|RF:3,J:1
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 28,8>28,8|RF:5,J:1
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 28,8>28,16|RF:7,J:1
            nn.ReLU(),
        )
        self.pool_1 = nn.MaxPool2d(2, 2)  # 28,16>14,16|RF:8,J:2
        self.transition_1 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False
        )  # 14,16>14,8|RF:8,J:2
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 14,8>14,8|RF:12,J:2
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 14,8>14,16|RF:16,J:2
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 14,16>14,16|RF:20,J:2
            nn.ReLU(),
        )
        self.pool_2 = nn.MaxPool2d(2, 2)  # 14,16>7,16|RF:21,J:4
        self.transition_2 = nn.Conv2d(
            in_channels=16, out_channels=4, kernel_size=(1, 1), padding=0, bias=False
        )  # 14,16>7,4|RF:21,J:4
        self.fc = nn.Linear(4 * 7 * 7, 10)  # 7,4>10

    def forward(self, x):
        x = self.block_1(x)
        x = self.pool_1(x)
        x = self.transition_1(x)
        x = self.block_2(x)
        x = self.pool_2(x)
        x = self.transition_2(x)
        x = x.view(-1, 4 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


## Adding BatchNorm
class MiniCNN_3(nn.Module):
    def __init__(self):
        super(MiniCNN_3, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 28,1>28,8|RF:3,J:1
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 28,8>28,8|RF:5,J:1
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 28,8>28,16|RF:7,J:1
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.pool_1 = nn.MaxPool2d(2, 2)  # 28,16>14,16|RF:8,J:2
        self.transition_1 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False
        )  # 14,16>14,8|RF:8,J:2
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False
            ),  # 14,8>14,8|RF:12,J:2
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 14,8>14,16|RF:16,J:2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 14,16>14,16|RF:20,J:2
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.pool_2 = nn.MaxPool2d(2, 2)  # 14,16>7,16|RF:21,J:4
        self.transition_2 = nn.Conv2d(
            in_channels=16, out_channels=3, kernel_size=(1, 1), padding=0, bias=False
        )  # 14,16>7,3|RF:21,J:4
        self.fc = nn.Linear(3 * 7 * 7, 10)  # 7,3>10

    def forward(self, x):
        x = self.block_1(x)
        x = self.pool_1(x)
        x = self.transition_1(x)
        x = self.block_2(x)
        x = self.pool_2(x)
        x = self.transition_2(x)
        x = x.view(-1, 3 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
