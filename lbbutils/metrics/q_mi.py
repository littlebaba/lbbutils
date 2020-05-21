import numpy as np
from lbbutils.metrics.mi import _mi
from PIL import Image


def _q_mi(im1: np.ndarray, im2: np.ndarray, fim: np.ndarray):
    """
        Ref: Comments on "Information measure for performance of image fusion"
        By M. Hossny et al.
        Electronics Letters Vol. 44, No.18, 2008
    :param im1:
    :param im2:
    :param fim:
    :return:
    """
    [I_fx, H_xf, H_x, H_f1] = _mi(im1, fim)
    [I_fy, H_yf, H_y, H_f2] = _mi(im2, fim)

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
    m1 = np.array(Image.open('../test/res/left.png'), dtype=np.float64)
    m2 = np.array(Image.open('../test/res/right.png'), dtype=np.float64)
    fim = np.array(Image.open('../test/res/fim.png'), dtype=np.float64)
    ret = _q_mi(m1, m2, fim)
    a = 1
