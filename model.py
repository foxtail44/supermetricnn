import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_filters, out_filters, size=3, bn_on=True):
        super(Conv, self).__init__()
        self.bn_on = bn_on
        self.bn = nn.BatchNorm1d(in_filters)
        padding = size
        self.conv = nn.Conv1d(in_filters, out_filters, size, padding=padding)
        self.lrelu = nn.LeakyReLU()

    def forward(self, input):
        if self.bn_on:
            return process(input, self.bn, self.conv, self.lrelu)
        return process(input, self.conv, self.lrelu)


class CNNSimple(torch.nn.Module):
    def __init__(self, input_size):
        super(CNNSimple, self).__init__()

        # input Nx10x1
        self.linear = nn.Linear(input_size, 128)
        self.conv1 = Conv(128, 256, 3)    # Nx10x5
        self.conv2 = Conv(256, 512, 3)  # Νx10x9
        self.conv3 = Conv(512, 1024, 3)  # Nx128x13
        self.conv4 = Conv(1024, 1024, 3)  # Nx256x17
        self.conv5 = Conv(1024, 1, 2)    # Nx1x20
        self.act = nn.Sigmoid()

    def forward(self, input):
        input = process(input, self.linear)
        input = process(input.unsqueeze(2), self.conv1, self.conv2, self.conv3)
        return process(input, self.conv4, self.conv5).squeeze(1)


def process(*func_input):
    if len(func_input) == 0:
        return None
    else:
        output = func_input[0]
        for i in range(1, len(func_input)):
            output = func_input[i](output)
        return output