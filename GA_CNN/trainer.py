import torch
import torch.nn as nn
import torch.optim as optim
import cnn_model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import your_custom_dataset_module  # 导入你自定义的数据集模块

# 定义训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

# 定义测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(test_loader.dataset), correct / total

# 定义主函数
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
    print("模型训练使用： ", device)
    model = cnn_model.get_model(input_channels, output_dim, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和测试模型
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()
