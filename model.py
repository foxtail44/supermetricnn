import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_filters, out_filters, size = 3):
        super(Conv, self).__init__()
        self.bn = nn.BatchNorm1d(in_filters)
        padding = size / 2
        self.conv = nn.Conv1d(in_filters, out_filters, size, padding=padding)
        self.lrelu = nn.LeakyReLU()

    def forward(self, input):
        return process(input, self.bn, self.conv, self.lrelu)


class CNNSimple(torch.nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 kernel_size):
        super(CNNSimple, self).__init__()

        self.conv1 = Conv(in_filters, 10, kernel_size)
        self.conv2 = Conv(10, 12, kernel_size)
        self.conv3 = Conv(12, 14, kernel_size)
        self.conv4 = Conv(14, 16, kernel_size)
        self.conv5 = Conv(16, 18, kernel_size)
        self.conv6 = Conv(18, out_filters, kernel_size)
        self.act = nn.Sigmoid()

    def forward(self, input):
        input = process(input, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6)
        output = self.act(input)
        return output


def process(*func_input):
    if len(func_input) == 0:
        return None
    else:
        output = func_input[0]
        for i in range(1, len(func_input)):
            output = func_input[i](output)
        return output