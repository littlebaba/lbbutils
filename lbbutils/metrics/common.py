import numpy as np


def normalize(x: np.ndarray):
    """

    :param x numpy.ndarray: x.shape (w,h,1)
    :return res numpy.ndarray: res.shape (w,h) astype=int16
    """
    if x.shape[2] == 1:
        data = x.squeeze()
        fz = data - np.min(data)
        fm = np.max(data) - np.min(data)
        res = fz / fm * 255
        return res.astype(np.int16)


def conv(im: np.ndarray, ker: np.ndarray, stride=1, padding=0):
    """

    :param im numpy.ndarray: (w,h) 2-D
    :param ker numpy.ndarray: (w,h) 2-D
    :param stride:
    :param padding:
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
    # ret = conv(im, ker, padding=1)

    # (3) create_window
    create_window(7)

    a = 1
