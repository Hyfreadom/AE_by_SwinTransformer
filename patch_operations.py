# 包含Patch相关操作
#       ---- Class PatchEmbed
#       ---- Class PatchMerging
#       ---- Class PatchExpand
#       ---- Func  window_partition
#       ---- Func  window_reverse
import torch
import torch.nn as nn

def window_partition(x, window_size):
    """
    将特征图分割成不重叠的窗口
    Args:
        x: (B, H, W, C)
        window_size: 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # 确保可以被window_size整除
    assert H % window_size == 0, f"高度 {H} 不能被窗口大小 {window_size} 整除"
    assert W % window_size == 0, f"宽度 {W} 不能被窗口大小 {window_size} 整除"
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口合并为特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: 窗口大小
        H, W: 输出特征图的高和宽
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # 检查输入尺寸
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像大小 ({H}*{W}) 与模型不匹配 ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"输入特征大小错误，应为{H*W}，实际为{L}"
        assert H % 2 == 0 and W % 2 == 0, f"x大小 ({H}*{W}) 不是偶数."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B H/2*W/2 2*C

        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)  # 先将通道数扩大为两倍
        self.norm = norm_layer(dim)
        # 新增: 线性层，将通道数降至一半
        self.reduce = nn.Linear(2 * dim, dim // 2, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        
        # 检查尺寸
        assert L == H * W, f"输入特征有错误的大小，应为{H*W}，实际为{L}"
        
        # 应用层归一化
        x = self.norm(x)
        
        # 第一步: 扩展通道 (B, L, C) -> (B, L, 2C)
        x = self.expand(x)
        
        # 将特征重组为空间表示
        x = x.view(B, H, W, 2 * C)
        
        # 将一个像素分为2x2=4个像素
        # 构造新的更大的特征图
        x_new = torch.zeros(B, H*2, W*2, C//2, device=x.device)
        
        # 分配像素 - 将每个像素的前C/2通道放到四个位置
        for i in range(2):
            for j in range(2):
                x_new[:, i::2, j::2, :] = x[:, :, :, (i*2+j)*(C//2):(i*2+j+1)*(C//2)]
        
        # 将特征转回序列形式
        x = x_new.view(B, 4*L, C//2)
        
        return x