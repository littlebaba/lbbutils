from lbbutils.metrics.common import ssim_yang
import numpy as np


def _q_y(im1, im2, fim):
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

    m1 = np.array(Image.open('../test/fused1_ours.png'), dtype=np.float64)[:, :, 0]
    m2 = np.array(Image.open('../test/fused2_ours.png'), dtype=np.float64)[:, :, 0]
    fim = np.array(Image.open('../test/fused3_ours.png'), dtype=np.float64)[:, :, 0]
    ret=_q_y(m1,m2,fim)