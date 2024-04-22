import torch
import torch.nn as nn


class STMConv(nn.Module):
    def __init__(self):
        super(STMConv, self).__init__()
        self.layer1 = nn.ModuleList([nn.Conv2d(256, 4, kernel_size=1, stride=1) for _ in range(32)])
        self.layer2 = nn.ModuleList([nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)for _ in range(32)])
        self.layer3 = nn.ModuleList([nn.Conv2d(4, 256, kernel_size=1, stride=1)for _ in range(32)])
    def forward(self, x):
        result = []
        for i in range(32):
            x = torch.nn.functional.relu(self.layer1[i](x))
            x = torch.nn.functional.relu(self.layer2[i](x))
            x = torch.nn.functional.relu(self.layer3[i](x))
            result.append(x)
        return sum(result)