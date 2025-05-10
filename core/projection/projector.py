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


class ImprovedAudioProjection(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=32, num_layers=2, dropout=0.1, activation='gelu', use_positional_encoding=True):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()

        # Sequential Layers
        self.projection_layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(num_layers):
            out_dim_layer = output_dim if i == num_layers - 1 else 2 * output_dim # Example: intermediate layer has higher dim
            self.projection_layers.append(nn.Linear(in_dim, out_dim_layer))
            if i < num_layers - 1:
                self.projection_layers.append(self.activation)
                self.projection_layers.append(nn.Dropout(dropout))
            in_dim = out_dim_layer

        self.final_reshape = nn.Linear(output_dim, seq_len * output_dim)

    def forward(self, x):
        B = x.shape[0]

        # Apply sequential linear and activation layers
        for layer in self.projection_layers:
            x = layer(x)

        # Final projection and reshape
        x = self.final_reshape(x)
        x = x.reshape(B, self.seq_len, self.output_dim)
        return x
