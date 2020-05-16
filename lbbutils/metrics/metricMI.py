import numpy as np
import torch
from lbbutils.metrics.common import normalize


def q_mi(im1: np.ndarray, im2: np.ndarray, fim: np.ndarray):
    # TODO 计算联合直方图
    im1.astype(np.float64)
    im2.astype(np.float64)
    row, col = im1.shape
    h = np.zeros((256, 256))
    for i in range(row):
        for j in range(col):
            h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
    # TODO 边缘直方图
    a = 1
    pass


if __name__ == '__main__':
    img = torch.rand((1, 1, 256, 256))
    lam = lambda x: x.data.squeeze(0).permute((1, 2, 0)).numpy()
    x_ = lam(img)
    ret = normalize(x_)
    q_mi(ret, ret, ret)
