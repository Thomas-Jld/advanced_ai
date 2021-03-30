import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

class superdupermodel(nn.Module):
    def __init__(self):
        super(superdupermodel, self).__init__()

        self.conv = nn.Sequential(
            #1, 28, 28
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            #4, 14, 14

            nn.Conv2d(8, 16, kernel_size=5),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #16, 7, 7

        )

        self.lin = nn.Sequential(
            nn.Linear(16*5*5, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 10),
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y.view(x.shape[0], -1)
        return self.lin(y)
