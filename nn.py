import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class Net(nn.Module):
    """Neural Network implementation.

    Attributes:
        a: Size of input layer.
        b: Size of hidden layer 1.
        c: Size of hidden layer 2.
        d: Size of output layer.
        fc1: Full connection layer 1.
        fc2: Full connection layer 2.
        out: Output layer.
        relu: Activation function of fc1 and fc2.
    """

    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, weights):
        super(Net, self).__init__()

        self.a = n_input
        self.b = n_hidden1
        self.c = n_hidden2
        self.d = n_output

        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_output)
        self.relu = nn.ReLU()

        self.update_weights(weights)

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.out(y)
        y = F.softmax(y, dim=1)  # 使用 softmax 作为输出层的激活函数
        return y

    def update_weights(self, weights):
        """Update the weights of the Neural Network.

           weights is a list of size a*b+b + b*c+c + c*d+d.
        """
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            x = self.a * self.b
            xx = x + self.b
            y = xx + self.b * self.c
            yy = y + self.c
            z = yy + self.c * self.d
            self.fc1.weight.data = weights[0:x].reshape(self.b, self.a)
            self.fc1.bias.data = weights[x:xx]
            self.fc2.weight.data = weights[xx:y].reshape(self.c, self.b)
            self.fc2.bias.data = weights[y:yy]
            self.out.weight.data = weights[yy:z].reshape(self.d, self.c)
            self.out.bias.data = weights[z:]

     def predict(self, input, weights):
        self.update_weights(weights)  # 确保每次预测前都更新权重
        # input = torch.tensor([input]).float()  # 确保输入格式正确
        y = self(input)
        print(self.fc1.weight[0][0])  # 打印第一层的权重以验证更新
        return torch.argmax(y, dim=1).item()

# 示例代码
if __name__ == '__main__':
    weights = [random.random() for i in range(32 * 20 + 20 * 12 + 12 * 4 + 20 + 12 + 4)]
    model = Net(32, 20, 12, 4, weights)
    input = [random.random() for _ in range(32)]
    print(model.predict(input))