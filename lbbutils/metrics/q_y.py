from lbbutils.metrics.common import ssim_yang
import numpy as np


def _q_y(im1, im2, fim):
    """
        Ref: A novel similarity based quality metric for image fusion, Information Fusion, Vol.9, pp156-160, 2008
        by Cui Yang et al.
    :param im1:
    :param im2:
    :param fim:
    :return:
    """
    ssim_map1, sigma1_sq1, sigma2_sq1 = ssim_yang(im1, im2)
    ssim_map2, sigma1_sq2, sigma2_sq2 = ssim_yang(im1, fim)
    ssim_map3, sigma1_sq3, sigma2_sq3 = ssim_yang(im2, fim)
    boo_map = ssim_map1 >= 0.75
    ramda = sigma1_sq1 / (sigma1_sq1 + sigma2_sq1)
    Q1 = (ramda * ssim_map2 + (1 - ramda) * ssim_map3) * boo_map.astype(np.int32)
    Q2 = (np.maximum(ssim_map2, ssim_map3)) * ((~boo_map).astype(np.int32))
    Q = np.mean(Q1 + Q2)
    return Q


if __name__ == '__main__':
    from PIL import Image

    m1 = np.array(Image.open('../test/res/left.png'), dtype=np.float64)
    m2 = np.array(Image.open('../test/res/right.png'), dtype=np.float64)
    fim = np.array(Image.open('../test/res/fim.png'), dtype=np.float64)
    ret = _q_y(m1, m2, fim)
    a = 1
