import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_filters, out_filters, size = 3):
        super(Conv, self).__init__()
        self.bn = nn.BatchNorm1d(in_filters)
        padding = size
        self.conv = nn.Conv1d(in_filters, out_filters, size, padding=padding)
        self.lrelu = nn.LeakyReLU()

    def forward(self, input):
        return process(input, self.bn, self.conv, self.lrelu)


class CNNSimple(torch.nn.Module):
    def __init__(self, in_filters):
        super(CNNSimple, self).__init__()

        # input Nx10x1
        self.conv1 = Conv(in_filters, 10, 3)    # Nx10x5
        self.conv2 = Conv(10, 64, 3)  # Νx10x9
        self.conv3 = Conv(64, 128, 3)  # Nx128x13
        self.conv4 = Conv(128, 256, 3)  # Nx256x17
        self.conv5 = Conv(256, 1, 2)    # Nx1x20
        self.act = nn.Sigmoid()

    def forward(self, input):
        return process(input,
                       self.conv1, self.conv2, self.conv3,
                       self.conv4, self.conv5, self.act).squeeze(1)


def process(*func_input):
    if len(func_input) == 0:
        return None
    else:
        output = func_input[0]
        for i in range(1, len(func_input)):
            output = func_input[i](output)
        return output