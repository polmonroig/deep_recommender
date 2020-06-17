import torch.nn as nn


class AutoEncoder(nn.module):

    def __init__(self):
        super().__init__()


    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def forward(self, x):
        return seld.decode(self.encode(x))
