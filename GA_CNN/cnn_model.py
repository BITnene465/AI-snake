import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_channels, output_dim, input_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算卷积层后特征图的大小
        conv_output_size = self._compute_conv_output_size(input_size, input_channels)

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * conv_output_size * conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def _compute_conv_output_size(self, input_size, input_channels):
        # 计算卷积层后特征图的大小
        dummy_input = torch.zeros((1, input_channels, input_size[0], input_size[1]))
        dummy_output = self.conv_layers(dummy_input)
        return dummy_output.size(2)


def get_model(input_channels, output_dim, input_size, device=torch.device("cpu")):
    model = CNN(input_channels, output_dim, input_size)
    if torch.cuda.is_available():
        model.to(device)
    return model



# 测试代码,看前向传播是否成功
if __name__ == '__main__':
    import torch
    # 定义模型参数
    input_channels = 3
    output_dim = 10
    input_size = (32, 32)  # 输入尺寸为 (height, width)
    # 创建模型
    model = get_model(input_channels, output_dim, input_size)
    # 创建随机输入数据
    input_data = torch.randn(1, input_channels, input_size[0], input_size[1])
    # 前向传播
    output = model(input_data)
    # 打印输出大小
    print("Output", output)