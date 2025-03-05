import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# 定义教师模型（较大）
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 32, 28, 28]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 14, 14]
        x = self.pool(x)  # [B, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # [B, 3136]
        x = F.relu(self.fc1(x))  # [B, 128]
        x = self.fc2(x)  # [B, 10]
        return x

# 定义学生模型（较小）
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 16, 28, 28]
        x = self.pool(x)  # [B, 16, 14, 14]
        x = self.pool(x)  # [B, 16, 7, 7]
        x = x.view(-1, 16 * 7 * 7)  # [B, 784]
        x = F.relu(self.fc1(x))  # [B, 64]
        x = self.fc2(x)  # [B, 10]
        return x

# 定义知识蒸馏损失函数 
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs, teacher_outputs, labels):
        # 交叉熵损失
        loss_ce = self.criterion_ce(student_outputs, labels)
        
        # 蒸馏损失（Kullback-Leibler散度）
        log_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        loss_kd = self.criterion_kd(log_student, soft_teacher) * (self.temperature ** 2)
        
        # 综合损失
        loss = self.alpha * loss_ce + (1. - self.alpha) * loss_kd
        return loss

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.4f} \tAccuracy: {accuracy:.4f}')

# 测试函数
def test(model, device, test_loader, criterion=None, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if criterion is not None:
                test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    if criterion is not None and epoch is not None:
        avg_loss = test_loss / len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        print(f'Test Epoch: {epoch} \tLoss: {avg_loss:.4f} \tAccuracy: {accuracy:.4f}')
    else:
        accuracy = correct / len(test_loader.dataset)
        print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def main():
    # 配置
    batch_size = 512
    epochs = 10
    lr = 0.001
    
    # 指定使用第二张GPU（索引为1），如果系统中有至少两张GPU
    if torch.cuda.is_available():
        if torch.cuda.device_count() < 2:
            print(f"系统中只有 {torch.cuda.device_count()} 张GPU，无法使用第二张GPU（cuda:1）。")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:1")
            print("使用第二张GPU（cuda:1）。")
    else:
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU。")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化教师模型并训练
    teacher_model = TeacherNet().to(device)
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()
    print("\nTraining Teacher Model...")
    for epoch in range(1, epochs + 1):
        train(teacher_model, device, train_loader, optimizer_teacher, criterion_ce, epoch)
        test(teacher_model, device, test_loader, criterion_ce, epoch)
    # 保存教师模型
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    
    # 初始化学生模型
    student_model = StudentNet().to(device)
    optimizer_student = optim.Adam(student_model.parameters(), lr=lr)
    distill_loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)  # alpha=1.0表示完全使用交叉熵损失
    
    # 加载训练好的教师模型（确保教师模型处于评估模式）
    teacher_model.eval()
    # 如果需要再次加载模型，可以取消注释
    # teacher_model.load_state_dict(torch.load('teacher_model.pth'))
    
    print("\nTraining Student Model with Knowledge Distillation...")
    for epoch in range(1, epochs + 1):
        student_model.train()
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer_student.zero_grad()
            student_outputs = student_model(data)
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            loss = distill_loss_fn(student_outputs, teacher_outputs, target)
            loss.backward()
            optimizer_student.step()
            
            total_loss += loss.item() * data.size(0)
            pred = student_outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = correct / len(train_loader.dataset)
        print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.4f} \tAccuracy: {accuracy:.4f}')
        test(student_model, device, test_loader, nn.CrossEntropyLoss(), epoch)
    
    # 保存学生模型
    torch.save(student_model.state_dict(), 'student_model.pth')
    
    # 最终评估
    print("\nFinal Evaluation:")
    test(teacher_model, device, test_loader)
    test(student_model, device, test_loader)
    
if __name__ == '__main__':
    main()