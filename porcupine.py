import torch
import torch.nn as nn
import torch.nn.functional as F

class PLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(PLinear, self).__init__(in_features, out_features, bias)
        self.scale = nn.Parameter(torch.ones(out_features))
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False

    def forward(self, input):
        return F.linear(input, self.weight, self.bias) * self.scale


class PConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        self.scale = nn.Parameter(torch.ones(out_channels))
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return (output.transpose(1, -1) * self.scale).transpose(1, -1)
