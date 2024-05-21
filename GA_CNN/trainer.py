import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, scheduler=None, log_dir=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir) if log_dir else None

    def train(self, epoch):
        self.model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            if self.writer:
                self.writer.add_scalar('Train Loss', loss.item(), epoch * len(self.train_loader) + i)
        if self.scheduler:
            self.scheduler.step()
        return running_loss / len(self.train_loader.dataset)

    def test(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            if self.writer:
                self.writer.add_scalar('Test Loss', running_loss / len(self.test_loader.dataset), epoch)
                self.writer.add_scalar('Test Accuracy', accuracy, epoch)
        return running_loss / len(self.test_loader.dataset), accuracy

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def close_writer(self):
        if self.writer:
            self.writer.close()