from torch import nn

# TODO: Everything, just a basic placeholder CNN encoder/decoder for now
class BytenetEncoder(nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, dilation_rate):
        super(BytenetEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv1d(num_channels, kernel_size, dilation_rate ** i))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


class BytenetDecoder(nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, dilation_rate):
        super(BytenetDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv1d(num_channels, kernel_size, dilation_rate ** i))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x
