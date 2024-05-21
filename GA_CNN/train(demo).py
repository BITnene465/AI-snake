import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from cnn_model import get_model
from trainer import Trainer

def main():
    # 设置超参数
    input_channels = 1  # 输入通道数
    output_dim = 10  # 输出维度
    batch_size = 64  # 批量大小
    num_epochs = 2  # 训练轮数
    learning_rate = 0.001  # 学习率

    # 准备数据集和数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = get_model(input_channels, output_dim, (28, 28), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建Trainer实例
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device)

    # 训练和测试模型
    for epoch in range(num_epochs):
        train_loss = trainer.train(epoch)
        test_loss, test_acc = trainer.test(epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    trainer.save_model('./data/mnist_model.pth')
    trainer.close_writer()

if __name__ == "__main__":
    main()
