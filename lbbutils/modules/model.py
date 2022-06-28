import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    将2D图像转成一维的多个块
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({self.img_size[0]}*{self.img_size[1]}) doesn't match model."
        x = self.proj(x)  # (2,768,14,14)
        x = x.flatten(2)  # (2,768,196)
        x = x.transpose(1, 2)  # (2,196,768)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        '''

        Args:
            dim: 768
            num_heads: 12
            qkv_bias:
            qk_scale:
            attn_drop_ratio:
            proj_drop_ratio:

        Returns:

        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 整除
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape  # [B, 196, 768]
        qkv_1 = self.qkv(x)  # (2,196,2304)
        qkv_2 = qkv_1.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (2,196,3,12,64)
        qkv = qkv_2.permute(2, 0, 3, 1, 4)  # (3,2,12,196,64)
        # .reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (B,197,12,768/12) = (B,197,12,64)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (2,12,196,64)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # (2,12,196,196)
        x = (attn @ v).transpose(1, 2)  # (2,196,12,64)
        x = x.reshape(B, N, C)  # (2,196,768)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        pass


if __name__ == '__main__':
    # 测试PatchEmbed
    img = torch.ones((2, 3, 224, 224))
    model = PatchEmbed()
    y = model(img) #（2,196,768）
    # 测试 Attention
    att = Attention()
    att_out = att(y)
    a = 1
