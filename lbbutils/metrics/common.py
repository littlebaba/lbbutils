import numpy as np


def normalize(x: np.ndarray):
    """

    :param x numpy.ndarray: x.shape (w,h,1)
    :return res numpy.ndarray: res.shape (w,h) astype=int16
    """
    if len(x.shape) == 2:
        data = x
        fz = data - np.min(data)
        fm = np.max(data) - np.min(data)
        res = fz / fm * 255
        return res.astype(np.int16)


def conv(im: np.ndarray, ker: np.ndarray, stride=1, padding=0):
    """

    :param im numpy.ndarray: (w,h) 2-D
    :param ker numpy.ndarray: (w,h) 2-D
    :param stride:
    :param padding: 1 is 'same',0 is 'valid'.
    :return:
    """
    r, c = im.shape
    pad_im = np.zeros((r + 2 * padding, c + 2 * padding))
    r_p, c_p = pad_im.shape
    pad_im[padding:r_p - padding, padding:c_p - padding] = im
    im = pad_im
    out_size = (r_p - ker.shape[0]) // stride + 1
    ret = np.zeros((out_size, out_size))
    for ri in range(0, out_size * stride, stride):
        for ci in range(0, out_size * stride, stride):
            region = im[ri:ri + ker.shape[0], ci:ci + ker.shape[0]]
            ret[ri // stride, ci // stride] = np.sum(region * ker)
    return ret


def create_window(win_size):
    def gaussian(win_size, sigma):
        gauss = np.array([np.exp(-(x - win_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(win_size)])
        return gauss / np.sum(gauss)

    _1D_window = np.expand_dims(gaussian(win_size, 1.5), 1)
    _2D_window = np.dot(_1D_window, _1D_window.T)
    return _2D_window


def ssim_yang(im1, im2):
    C1 = 2e-16;
    C2 = 2e-16;
    window = create_window(7)
    window = window / np.sum(window)
    mu1 = conv(im1, window)
    mu2 = conv(im2, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = conv(im1 * im1, window) - mu1_sq
    sigma2_sq = conv(im2 * im2, window) - mu2_sq
    sigma12 = conv(im1 * im2, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map, sigma1_sq, sigma2_sq


def freqspace(num):
    lam = lambda x: np.arange(-x + 1, x, 2) / x if x // 2 == 1 else np.arange(-x, x - 1, 2) / x
    f1 = lam(num[0])
    f2 = lam(num[1])

    m, n = np.meshgrid(f1, f2)
    return n.T, m.T


if __name__ == '__main__':
    import torch

    # # (1)normalize
    # x = torch.rand((1, 1, 5, 5))
    # lam = lambda x: x.data.squeeze(0).permute((1, 2, 0)).numpy()
    # x_ = lam(x)
    # ret = normalize(x_)
    # a = 1

    # (2) conv
    # im = np.array([[1, 1, 1, 1, 1],
    #                [2, 2, 2, 2, 2],
    #                [3, 3, 3, 3, 3],
    #                [4, 3, 4, 3, 2],
    #                [3, 1, 5, 7, 3]])  # 1,1,1,1,1;2,2,2,2,2;3,3,3,3,3
    # ker = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # -1 0 1 ; -2 0 2 ; -1 0 1
    # ret = conv(im, ker, padding=0)

    # (3) create_window
    # create_window(7)

    # (4) ssim_yang
    # from PIL import Image
    #
    # m1 = np.array(Image.open('../test/fused1_ours.png'), dtype=np.float64)[:, :, 0]
    # m2 = np.array(Image.open('../test/fused2_ours.png'), dtype=np.float64)[:, :, 0]
    # fim = np.array(Image.open('../test/fused3_ours.png'), dtype=np.float64)[:, :, 0]
    # ssim_yang(m1, m2)

    # (5) freqspace()
    u, v = freqspace([3, 4])
    a = 1
