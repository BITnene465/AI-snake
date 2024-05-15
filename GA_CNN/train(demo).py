import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import your_custom_dataset_module  # 导入你自定义的数据集模块
from cnn_model import get_model
from trainer import Trainer

def main():
    # 设置超参数
    input_channels = 3  # 输入通道数
    output_dim = 10  # 输出维度
    batch_size = 64  # 批量大小
    num_epochs = 10  # 训练轮数
    learning_rate = 0.001  # 学习率

    # 准备数据集和数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = your_custom_dataset_module.YourCustomDataset(root='data', train=True, transform=transform)
    test_dataset = your_custom_dataset_module.YourCustomDataset(root='data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = get_model(input_channels, output_dim, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建Trainer实例
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device)

    # 训练和测试模型
    for epoch in range(num_epochs):
        train_loss = trainer.train()
        test_loss, test_acc = trainer.test()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    trainer.save_model('model.pth')
    trainer.close_writer()

if __name__ == "__main__":
    main()
