# 包含完整的自编码器模型
#       ---- Class SwinTransformerAutoEncoder
import torch
import torch.nn as nn
from patch_operations import PatchEmbed, PatchMerging, PatchExpand
from basic_blocks import BasicLayer

if_print = False
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
        self.bottleneck_dim = bottleneck_dim

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
        if if_print: print(f"Patch Embed 输出: {B}x{L}x{C}")
        
        # 编码器前向传播
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if if_print: print(f"编码器层 {i+1} 输出: {x.shape}")
        
        # 瓶颈层
        B, L, C = x.shape
        x = x.reshape(B, -1)  # B, L*C
        if if_print:(print(f"瓶颈层 输入: {x.shape}"))
        x = self.bottleneck(x)  # 通过瓶颈压缩和恢复
        if if_print:(print(f"瓶颈层 输出: {x.shape}"))
        x = x.reshape(B, L, C)  # B, L, C
        
        return x

    def forward_decoder(self, x):
        # 解码器前向传播
        for i, layer in enumerate(self.decoder_layers):
            B, L, C = x.shape
            if if_print:(print(f"解码器层 {i+1} 输入: {B}x{L}x{C}"))
            x = layer(x)
            if if_print:(print(f"解码器层 {i+1} 输出: {x.shape}"))
        
        # 输出投影
        B, L, C = x.shape
        H, W = self.final_resolution
        
        # 确保特征图尺寸正确
        assert L == H * W, f"解码器输出特征尺寸错误，应为{H*W}，实际为{L}"
        
        if if_print:(print(f"输出投影 输入: {x.shape}"))
        x = self.output_proj(x)  # B, L, patch_size*patch_size*in_chans
        if if_print:(print(f"输出投影 输出: {x.shape}"))
        
        # 重塑为图像格式
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, H, patch_size, W, patch_size
        x = x.reshape(B, self.in_chans, H * self.patch_size, W * self.patch_size)  # B, C, H*patch_size, W*patch_size
        if if_print:(print(f"最终输出: {x.shape}"))
        
        return x

    def forward(self, x):
        if if_print:(print(f"\n--- 前向传播 ---"))
        if if_print:(print(f"输入: {x.shape}"))
        
        # 编码
        latent = self.forward_features(x)
        
        # 解码
        output = self.forward_decoder(latent)
        
        return output