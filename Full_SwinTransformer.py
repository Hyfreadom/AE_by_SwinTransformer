# 此为为拆分版的代码，便于连贯直接喂给LLM，包含
#       ---- attention.py
#       ---- patch_operations.py
#       ---- basic_blocks.py
#       ---- model.py
#       ---- utilis.py
#       ---- main.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F


# 窗口自注意力模块 (W-MSA)
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Wh, Ww]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # 生成相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 调整为非负
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, C//num_heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x


# MLP模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 窗口分割功能
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


# 窗口合并功能
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


# Swin Transformer块
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, 
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 如果使用移位窗口，计算注意力掩码
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"输入特征的大小错误，应为{H*W}，实际为{L}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 移位窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 移位回来
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


# Patch Embedding层
# input : (1,32,32)
# patches_number: (32//4) * (32//4) = 64
# output: (64,embed_dim=96)
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


# Patch Merging层 (用于下采样)
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


# 修复的Patch Expanding层 (用于上采样)
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


# 基本的Swin Transformer层
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, upsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 构建Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # 交替使用W-MSA和SW-MSA
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样或上采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            new_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
            print(f"下采样: {input_resolution} -> {new_resolution}, 通道: {dim} -> {dim*2}")
        else:
            self.downsample = None
            
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
            new_resolution = (input_resolution[0] * 2, input_resolution[1] * 2)
            print(f"上采样: {input_resolution} -> {new_resolution}, 通道: {dim} -> {dim//2}")
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        if self.upsample is not None:
            x = self.upsample(x)
            
        return x


# Swin Transformer自编码器
class SwinTransformerAutoEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=1, embed_dim=96,
                 depths=[2, 2], depths_decoder=[2, 2], num_heads=[3, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 bottleneck_dim=64):  # 添加瓶颈维度参数，实现压缩
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        # 分割图像为非重叠的块并嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
            
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 打印每层的分辨率和通道数
        print(f"初始分辨率: {patches_resolution}, 嵌入维度: {embed_dim}")

        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 编码器层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                 patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                upsample=None)
            self.layers.append(layer)

        # 瓶颈层
        self.bottleneck_size = patches_resolution[0] // (2 ** (self.num_layers - 1)) * \
                               patches_resolution[1] // (2 ** (self.num_layers - 1))
        self.bottleneck_dim = bottleneck_dim
        bottleneck_in_features = int(embed_dim * 2 ** (self.num_layers - 1))
        bottleneck_in_size = bottleneck_in_features * self.bottleneck_size
        
        print(f"瓶颈层尺寸: 输入=[{self.bottleneck_size}, {bottleneck_in_features}], " +
              f"压缩={bottleneck_dim}, 压缩率={bottleneck_dim/bottleneck_in_size:.4f}")
        
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_in_size, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_in_size)
        )

        # 解码器层
        self.decoder_layers = nn.ModuleList()
        depths_decoder = depths_decoder[::-1]  # 反转解码器深度列表
        for i_layer in range(self.num_layers):
            input_resolution = (patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                              patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)))
            
            dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            
            layer = BasicLayer(
                dim=dim,
                input_resolution=input_resolution,
                depth=depths_decoder[i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:self.num_layers - 1 - i_layer]):sum(depths[:self.num_layers - i_layer])],
                norm_layer=norm_layer,
                downsample=None,
                upsample=PatchExpand if i_layer < self.num_layers - 1 else None)
            self.decoder_layers.append(layer)

        # 输出层
        self.final_resolution = (patches_resolution[0], patches_resolution[1])
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        self.apply(self._init_weights)

        print(f"输出分辨率: {self.final_resolution}, 嵌入维度: {embed_dim} -> {patch_size * patch_size * in_chans}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)  # B, L, C
        B, L, C = x.shape
        print(f"Patch Embed 输出: {B}x{L}x{C}")
        
        # 编码器前向传播
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"编码器层 {i+1} 输出: {x.shape}")
        
        # 瓶颈层
        B, L, C = x.shape
        x = x.reshape(B, -1)  # B, L*C
        print(f"瓶颈层 输入: {x.shape}")
        x = self.bottleneck(x)  # 通过瓶颈压缩和恢复
        print(f"瓶颈层 输出: {x.shape}")
        x = x.reshape(B, L, C)  # B, L, C
        
        return x

    def forward_decoder(self, x):
        # 解码器前向传播
        for i, layer in enumerate(self.decoder_layers):
            B, L, C = x.shape
            print(f"解码器层 {i+1} 输入: {B}x{L}x{C}")
            x = layer(x)
            print(f"解码器层 {i+1} 输出: {x.shape}")
        
        # 输出投影
        B, L, C = x.shape
        H, W = self.final_resolution
        
        # 确保特征图尺寸正确
        assert L == H * W, f"解码器输出特征尺寸错误，应为{H*W}，实际为{L}"
        
        print(f"输出投影 输入: {x.shape}")
        x = self.output_proj(x)  # B, L, patch_size*patch_size*in_chans
        print(f"输出投影 输出: {x.shape}")
        
        # 重塑为图像格式
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, H, patch_size, W, patch_size
        x = x.reshape(B, self.in_chans, H * self.patch_size, W * self.patch_size)  # B, C, H*patch_size, W*patch_size
        print(f"最终输出: {x.shape}")
        
        return x

    def forward(self, x):
        print(f"\n--- 前向传播 ---")
        print(f"输入: {x.shape}")
        
        # 编码
        latent = self.forward_features(x)
        
        # 解码
        output = self.forward_decoder(latent)
        
        return output


# MNIST数据集尺寸调整为32x32的自定义变换
class MNISTResize(object):
    def __call__(self, img):
        # 将MNIST图像(28x28)填充到32x32
        padding = (2, 2, 2, 2)  # 左、上、右、下各填充2个像素
        return F.pad(img, padding, "constant", 0)


# 训练函数
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


# 可视化重建结果
def visualize_reconstruction(model, data_loader, device, num_images=5):
    model.eval()
    
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
    
    # 创建图表
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*3, 6))
    
    for i in range(num_images):
        # 显示原始图像 (裁剪回28x28以便直观比较)
        axes[0, i].imshow(images[i][0, 2:30, 2:30].numpy(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # 显示重建图像 (同样裁剪回28x28)
        axes[1, i].imshow(reconstructed[i][0, 2:30, 2:30].numpy(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


# 主程序
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
    
    print(f"模型总参数量: {total_params}")              # 810884
    print(f"原始图像大小: {image_size}")                # 1024
    print(f"瓶颈维度: {model.bottleneck_dim}")         # 64
    print(f"压缩率: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)")    # 6.25%
    
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
    visualize_reconstruction(trained_model, test_loader, device)


if __name__ == "__main__":
    main()