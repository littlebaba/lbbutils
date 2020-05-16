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


if __name__ == '__main__':
    import torch

    x = torch.rand((1, 1, 5, 5))
    lam = lambda x: x.data.squeeze(0).permute((1, 2, 0)).numpy()
    x_ = lam(x)
    ret = normalize(x_)
    a = 1
