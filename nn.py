import torch
import torch.nn as nn
import random


class Net(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, weights):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.update_weights(weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x

    # def forward(self, x):
    #     x = self.fc1(x)
    #     print("After fc1:", x.shape)
    #     x = self.relu(x)
    #     x = self.fc2(x)
    #     print("After fc2:", x.shape)
    #     x = self.relu(x)
    #     x = self.out(x)
    #     print("After out:", x.shape)
    #     x = self.sigmoid(x)
    #     return x

    def update_weights(self, weights):
        """ 根据提供的权重列表更新网络权重 """
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            # 重新计算权重和偏置的位置
            x = self.fc1.in_features * self.fc1.out_features
            xx = x + self.fc1.out_features
            y = xx + self.fc2.in_features * self.fc2.out_features
            yy = y + self.fc2.out_features
            z = yy + self.out.in_features * self.out.out_features

            self.fc1.weight.data = weights[:x].reshape(self.fc1.out_features, self.fc1.in_features)
            self.fc1.bias.data = weights[x:xx]
            self.fc2.weight.data = weights[xx:y].reshape(self.fc2.out_features, self.fc2.in_features)
            self.fc2.bias.data = weights[y:yy]
            self.out.weight.data = weights[yy:z].reshape(self.out.out_features, self.out.in_features)
            self.out.bias.data = weights[z:]

    def predict(self, input):
        """对输入数据进行预测，返回最大值的索引"""
        input = torch.tensor(input).float().unsqueeze(0)  # 转换输入为适当的Tensor
        output = self.forward(input)
        return torch.argmax(output, dim=1).item()  # 返回最大值索引



if __name__ == '__main__':
    weights = [random.random() for i in range(12 * 20 + 20 * 12 + 12 * 4 + 20 + 12 + 4)]
    model = Net(12, 20, 12, 4, weights)
    input = [random.random() for _ in range(12)]
    input_tensor = torch.tensor(input).float()
    print(input_tensor)# Correctly formatting the input
    print(model(input_tensor))  # Corrected line: pass input_tensor instead of input

