import numpy as np


def _mi(im1, im2):
    im1.astype(np.float64)
    im2.astype(np.float64)
    row, col = im1.shape
    h = np.zeros((256, 256))
    for i in range(row):
        for j in range(col):
            h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
    # TODO 边缘直方图
    h = h / np.sum(h)
    im1_marg = np.sum(h, 0)  # 求列和
    im2_marg = np.sum(h, 1)  # 求行和

    lam = lambda x: (x == 0).astype(np.float64)

    H_x = -np.sum(im1_marg * np.log2(im1_marg + lam(im1_marg)))
    H_y = -np.sum(im2_marg * np.log2(im2_marg + lam(im2_marg)))
    H_xy = -np.sum(h * np.log2(h + lam(h)))
    mi = H_x + H_y - H_xy
    return mi, H_xy, H_x, H_y
