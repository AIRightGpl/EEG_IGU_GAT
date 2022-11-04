import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, shrink_ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // shrink_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // shrink_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        add_out = avg_out + max_out
        out = self.sigmoid(add_out)
        return out
