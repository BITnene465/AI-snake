import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        # 定义全连接层和ReLU激活函数
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_mpl_model(input_size, hidden_size1, hidden_size2, output_size, device='cpu'):
    model = MLP(input_size, hidden_size1, hidden_size2, output_size)
    model.to(device)
    return model

if __name__ == '__main__':
    model = MLP(input_size=24, hidden_size1=128, hidden_size2=32, output_size=3)
    print(len(model.fc1.weight.flatten()) + len(model.fc2.weight.flatten()) + len(model.fc3.weight.flatten()))


