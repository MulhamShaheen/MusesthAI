import torch
from torch import nn


class AudioProjection(nn.Module):
    def __init__(self, input_dim, output_dim, sequal_len=32, scale_factor=2):
        super(AudioProjection, self).__init__()
        self.scale_factor = scale_factor
        self.sequal_len = sequal_len
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, scale_factor * output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(scale_factor * output_dim, sequal_len * output_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.reshape(x, (B, self.sequal_len, self.output_dim))

        return x