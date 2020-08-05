# @Author  : liheng
# @Time    : 2020/8/5 21:45


import torch
from torch.nn import functional as F


def gradient(img):
    weight = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    return F.conv2d(img,weight, padding=1)


if __name__ == '__main__':
    pass
    #TODO 测试gradient
