import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(9216, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 4)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)  # 输入通道数为1，输出通道数为16，卷积核大小为3
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)  # 最大池化层，池化核大小为2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)  # 输入通道数为16，输出通道数为32，卷积核大小为3
        self.maxpool = nn.MaxPool1d(kernel_size=2)  # 最大池化层，池化核大小为2
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)  # 输入通道数为16，输出通道数为32，卷积核大小为3
        self.fc = nn.Linear(12544, 2048)  # 输入特征数为32*24，输出特征数为36
        self.fc2 = nn.Linear(2048, 512)  # 输入特征数为32*24，输出特征数为36
        self.fc3 = nn.Linear(512, 128)  # 输入特征数为32*24，输出特征数为36
        self.fc4 = nn.Linear(128,3)  # 输入特征数为32*24，输出特征数为36

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)  # 展开特征图
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

