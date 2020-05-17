import numpy as np
from lbbutils.metrics.MI import _MI


def _q_mi(im1: np.ndarray, im2: np.ndarray, fim: np.ndarray):
    [I_fx, H_xf, H_x, H_f1] = _MI(im1, fim)
    [I_fy, H_yf, H_y, H_f2] = _MI(im2, fim)

    res = 2 * (I_fx / (H_f1 + H_x) + I_fy / (H_f2 + H_y))
    return res


if __name__ == '__main__':
    # img1 = torch.rand((1, 1, 256, 256))
    # img2 = torch.rand((1, 1, 256, 256))
    # lam = lambda x: x.data.squeeze(0).permute((1, 2, 0)).numpy()
    # x_ = lam(img1)
    # y_ = lam(img2)
    # ret1 = normalize(x_)
    # ret2 = normalize(y_)
    ret1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ret2 = np.array([[2, 3, 5], [3, 4, 5], [5, 7, 4]])
    fim = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
    _q_mi(ret1, ret2, fim)
