import torch.nn as nn
from collections import OrderedDict


class LinearModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, data_input):
        output = self.activation(self.linear(data_input))
        return output

class RlModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Sequential(OrderedDict([("linear{}".format(i),
                                                 LinearModule(config['input_layer'][i], config['hidden_layer'][i]))
                                                for i in range(len(config['input_layer']))]))

    def forward(self, state):
        # 第一种情况，直接将输入flatten
        input_data = state.reshape(-1)
        return self.model(input_data)
