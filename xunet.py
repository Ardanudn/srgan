import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
norm = torch.nn.BatchNorm2d

class XuNet(nn.Module):
    def __init__(self, kernel_size=5, padding=2):
        super(XuNet, self).__init__()
        # self.preprocessing = kv_conv2d()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(5, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(5, 2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        # self.pool5 = nn.AvgPool2d(kernel_size=16, stride=1)
        # self.pool6 = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.pool7 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128*(1+2*2+4*4), 128)
        self.fc2 = nn.Linear(128, 2)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x  # self.preprocessing(x)
        out = self.conv1(out)
        out = torch.abs(out)
        # print(out)
        out = self.relu(self.bn1(out))
        out = self.pool1(out)
        out = self.tanh(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.pool3(out)
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.pool4(out)
        out = self.relu(self.bn5(self.conv5(out)))
        _, _, x, y = out.size()
        out1 = F.avg_pool2d(out, kernel_size=(x/4, y/4), stride=(x/4, y/4))
        out2 = F.avg_pool2d(out, kernel_size=(x/2, y/2), stride=(x/2, y/2))
        out3 = F.avg_pool2d(out, kernel_size=(x, y), stride=1)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out3 = out3.view(out3.size(0), -1)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.sigmoid(out)

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, kv_conv2d) or \
                    isinstance(mod, norm) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Conv2d):
                nn.init.xavier_uniform(mod.weight)
            elif isinstance(mod, norm):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal(mod.weight, 0., 0.01)
                mod.bias.data.zero_()

class Stegoanalyser(nn.Module):
    def __init__(self, *, in_channels=3):
        super().__init__()
        # Input Dimension: (nc) x 64 x 64
        weights = 1 / 12 * torch.tensor([
            [-1., 2., -2., 2, -1],
            [2., -6, 8, -6, 2],
            [-2., 8., -12, 8, -2],
            [2., -6, 8, -6, 2],
            [-1., 2., -2., 2, -1]])
        self.weights = weights.view(1, 1, 5, 5).repeat(3, in_channels, 1, 1)

        self.conv1 = nn.Conv2d(in_channels, 10, 7)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 30, 3)
        self.conv4 = nn.Conv2d(30, 40, 3)
        self.mp1 = nn.MaxPool2d(4, 4, padding=1)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.dense1 = nn.Linear(1000, 100)
        self.dense2 = nn.Linear(100, 1)

    def forward(self, x):
        self.weights = self.weights.to(x)
        x = F.relu(F.conv2d(x, self.weights, padding=2))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mp1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp2(x)
        x = x.view(x.size(0), -1)
        x = F.tanh(self.dense1(x))
        x = self.dense2(x).view(x.size(0), -1)
        x = x[..., None, None]
        return x