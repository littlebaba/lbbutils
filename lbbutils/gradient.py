# @Author  : liheng
# @Time    : 2020/8/5 21:45


import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import ToTensor,Compose


def gradient(img):
    weight = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    return F.conv2d(img, weight, padding=1)


if __name__ == '__main__':
    # TODO 测试gradient
    img = Image.open("./test/res/left.png")
    img=ToTensor()(img)
    img=torch.squeeze(img,0)
    new_img=gradient(img)
    print(new_img.shape)
