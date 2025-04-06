from torch import nn


class AudioProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AudioProjection, self).__init__()
        self.projection = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x):
        x = self.projection(x)
        x = x.view(2, x.shape[0], 2048)
        return x
