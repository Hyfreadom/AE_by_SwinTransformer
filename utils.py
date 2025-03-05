# 包含辅助功能
#       ---- Class MNISTResize
#       ---- Func  train_swin_ae
#       ---- Func  visualize_reconstruction
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
from torch.nn import functional as F


class MNISTResize(object):
    def __call__(self, img):
        # 将MNIST图像(28x28)填充到32x32
        padding = (2, 2, 2, 2)  # 左、上、右、下各填充2个像素
        return F.pad(img, padding, "constant", 0)

def train_swin_ae(model, train_loader, test_loader, num_epochs=10, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for data, _ in train_loader:
            data = data.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, data)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                test_loss += loss.item()
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Test Loss: {test_loss/len(test_loader):.6f}')
    
    return model


import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

def visualize_reconstruction(model, data_loader, device, num_images=5, output_dir='./results'):
    """
    可视化重建结果并保存到文件 - 在一张图中同时显示原始图像和重建图像
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 使用的设备
        num_images: 要可视化的图像数量
        output_dir: 保存结果的目录
    """
    model.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'reconstruction_results_{timestamp}.png')
    
    # 获取一些测试图像
    dataiter = iter(data_loader)
    images, _ = next(dataiter)
    
    # 选择num_images个图像
    images = images[:num_images].to(device)
    
    # 通过模型获取重建图像
    with torch.no_grad():
        reconstructed = model(images)
    
    # 将图像从设备移回CPU
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # 创建单张图表，上方是原始图像，下方是重建图像
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*3, 6))
    
    for i in range(num_images):
        # 显示原始图像 (裁剪回28x28以便直观比较)
        if num_images > 1:
            axes[0, i].imshow(images[i][0, 2:30, 2:30].numpy(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # 显示重建图像 (同样裁剪回28x28)
            axes[1, i].imshow(reconstructed[i][0, 2:30, 2:30].numpy(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        else:
            # 处理单图情况
            axes[0].imshow(images[i][0, 2:30, 2:30].numpy(), cmap='gray')
            axes[0].set_title(f'Original')
            axes[0].axis('off')
            
            axes[1].imshow(reconstructed[i][0, 2:30, 2:30].numpy(), cmap='gray')
            axes[1].set_title(f'Reconstructed')
            axes[1].axis('off')
    
    # 添加一个总标题
    plt.suptitle('Original vs Reconstructed Images', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题留出空间
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"重建结果已保存到: {save_path}")
    
    # 关闭图像以释放内存
    plt.close(fig)
    
    # 额外保存一张MSE损失值的图像
    mse_losses = []
    for i in range(num_images):
        orig = images[i][0, 2:30, 2:30].numpy()
        recon = reconstructed[i][0, 2:30, 2:30].numpy()
        mse = np.mean((orig - recon) ** 2)
        mse_losses.append(mse)
    
    # 创建MSE损失条形图
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_images+1), mse_losses)
    plt.xlabel('Image Number')
    plt.ylabel('MSE Loss')
    plt.title('Reconstruction MSE Loss')
    plt.xticks(range(1, num_images+1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 显示每个条形上的实际MSE值
    for i, v in enumerate(mse_losses):
        plt.text(i+1, v + max(mse_losses)*0.01, f'{v:.5f}', 
                 ha='center', va='bottom', fontsize=9)
    
    # 保存MSE图
    mse_path = os.path.join(output_dir, f'mse_loss_{timestamp}.png')
    plt.savefig(mse_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MSE损失图已保存到: {mse_path}")
    
    # 返回平均MSE损失
    return np.mean(mse_losses)