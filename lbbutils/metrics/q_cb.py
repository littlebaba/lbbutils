from lbbutils.metrics.common import normalize, freqspace, conv
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, ifft2


def _q_cb(im1, im2, fim):
    """
        Ref: A new automated quality assessment algorithm for image fusion, Image and Vision Computing, 27 (2009) 1421-1432
        By Yin Chen et al.
    :param im1:
    :param im2:
    :param fim:
    :return:
    """
    im1 = im1 / 255
    im2 = im2 / 255
    fim = fim / 255

    im1 = normalize(im1)
    im2 = normalize(im2)
    fim = normalize(fim)

    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622

    row, col = im1.shape
    HH, LL = row / 30, col / 30
    u, v = freqspace([row, col])
    u = LL * u
    v = HH * v
    r = np.sqrt(u * u + v * v)
    Sd = np.exp(-(r / f0) ** 2) - a * np.exp(-(r / f1) ** 2)

    fim1 = ifft2(ifftshift(fftshift(fft2(im1)) * Sd))
    fim2 = ifft2(ifftshift(fftshift(fft2(im2)) * Sd))
    ffim = ifft2(ifftshift(fftshift(fft2(fim)) * Sd))

    def gaussian2d(sigma):
        x, y = np.meshgrid(range(-15, 16), range(-15, 16))
        win = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
        return win

    def contrast(win1, win2, im):
        buff = conv(im, win1, padding=15)
        buff1 = conv(im, win2, padding=15)
        return buff / buff1 - 1

    G1 = gaussian2d(2)
    G2 = gaussian2d(4)

    k, p, q, h, Z = 1, 3, 2, 1, 0.0001
    C1 = contrast(G1, G2, np.real(fim1))
    C1 = abs(C1)
    C1P = (k * (np.power(C1, p))) / (h * (np.power(C1, q)) + Z)

    C2 = contrast(G1, G2, np.real(fim2))
    C2 = abs(C2)
    C2P = (k * (np.power(C2, p))) / (h * (np.power(C2, q)) + Z)

    Cf = contrast(G1, G2, np.real(ffim))
    Cf = abs(Cf)
    CfP = (k * (np.power(Cf, p))) / (h * (np.power(Cf, q)) + Z)

    mask = (C1P < CfP)
    mask = mask.astype(np.float64)
    Q1F = (C1P / CfP) * mask + (CfP / C1P) * (1 - mask)

    mask = (C2P < CfP)
    mask = mask.astype(np.float64)
    Q2F = (C2P / CfP) * mask + (CfP / C2P) * (1 - mask)

    ramda1 = (C1P * C1P) / (C1P * C1P + C2P * C2P)
    ramda2 = (C2P * C2P) / (C1P * C1P + C2P * C2P)
    Q = ramda1 * Q1F + ramda2 * Q2F
    ret = np.mean(Q)
    return ret


if __name__ == '__main__':
    from PIL import Image

    m1 = np.array(Image.open('../test/res/left.png'), dtype=np.float64)
    m2 = np.array(Image.open('../test/res/right.png'), dtype=np.float64)
    fim = np.array(Image.open('../test/res/fim.png'), dtype=np.float64)
    ret = _q_cb(m1, m2, fim)
    a = 1
