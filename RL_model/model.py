import torch.nn as nn


class RlModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleList([nn.Linear(config['input_layer'][i], config['hidden_layer'][i])
                                    for i in range(len(config['input_layer']))])

    def forward(self, state):
        # 第一种情况，直接将输入flatten
        input_data = state.reshape(-1)
        return self.model(input_data)
