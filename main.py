# 此为主程序入口，包含模型参数和训练模型
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SwinTransformerAutoEncoder
from utils import MNISTResize, train_swin_ae, visualize_reconstruction

def main():
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 自定义转换 - 将MNIST(28x28)调整为32x32
    transform = transforms.Compose([
        transforms.ToTensor(),
        MNISTResize()  # 自定义变换添加填充
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 简单检查一个批次，确认尺寸
    sample_batch, _ = next(iter(train_loader))
    print(f"数据尺寸: {sample_batch.shape}")  # 应该是[128, 1, 32, 32]
    
    # 创建模型 - 调整为32x32输入
    model = SwinTransformerAutoEncoder(
        img_size=32,               # 调整为32x32
        patch_size=4,              # 块大小
        in_chans=1,                # 输入通道数
        embed_dim=48,              # 嵌入维度
        depths=[2, 2],             # 编码器深度
        depths_decoder=[2, 2],     # 解码器深度
        num_heads=[3, 6],          # 注意力头数
        window_size=8,             # 窗口大小调整为8（可被32整除）
        bottleneck_dim=64          # 瓶颈维度，实现压缩
    )
    
    # 打印模型
    print("\n--- 模型结构 ---")
    
    # 计算参数量和压缩率
    total_params = sum(p.numel() for p in model.parameters())
    image_size = 32 * 32
    compression_ratio = model.bottleneck_dim / image_size
    
    print(f"模型总参数量: {total_params}")
    print(f"原始图像大小: {image_size}")
    print(f"瓶颈维度: {model.bottleneck_dim}")
    print(f"压缩率: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)")
    
    # 检查尺寸一致性 - 输入一个样本批次经过模型
    with torch.no_grad():
        test_batch = sample_batch[:2].to(device)  # 只取两个样本测试
        model.to(device)
        output = model(test_batch)
        assert output.shape == test_batch.shape, f"输出形状 {output.shape} 与输入形状 {test_batch.shape} 不匹配"
        print("\n尺寸一致性检查通过!")
    
    # 训练模型
    trained_model = train_swin_ae(model, train_loader, test_loader, num_epochs=5, device=device)
    
    # 可视化重建结果
    avg_mse = visualize_reconstruction(trained_model, test_loader, device,num_images=5, 
                             output_dir='./reconstructions')
    print(f"average reconstruction MSE:{avg_mse:.6f}")

if __name__ == "__main__":
    main()