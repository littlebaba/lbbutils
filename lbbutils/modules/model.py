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
        x = self.proj(x)  # .flatten(2).traspose(1, 2)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    img = torch.ones((2, 3, 224, 224))
    model = PatchEmbed()
    y = model(img)
    a = 1
