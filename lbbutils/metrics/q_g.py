import numpy as np
from lbbutils.metrics.common import conv
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')


def _q_g(im1: np.ndarray, im2: np.ndarray, fim: np.ndarray):
    """
        Ref: Objective Pixel-level Image Fusion Performance Measure, Proc. SPIE 4051, 89 (2000)
        by C. Xydeas
    :param im1:
    :param im2:
    :param fim:
    :return:
    """
    ker1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ker2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # get the map
    fuseG, fuseA = get_map(fim, ker1, ker2)
    im1G, im1A = get_map(im1, ker1, ker2)
    im2G, im2A = get_map(im2, ker1, ker2)
    # edge preservation estimation
    Gaf, Aaf = edge_pres_est(fuseG, fuseA, im1G, im1A)
    Gbf, Abf = edge_pres_est(fuseG, fuseA, im2G, im2A)

    Qaf = f(Gaf, Aaf)
    Qbf = f(Gbf, Abf)

    Wa = im1G
    Wb = im2G
    tmp = (Qaf * Wa + Qbf * Wb) / (Wa + Wb)
    res=np.nanmean(tmp)
    return res


def f(G_f, A_f):
    gama1 = 1
    gama2 = 1
    k1 = -10
    k2 = -20
    delta1 = 0.5
    delta2 = 0.75
    Qg_AF = gama1 / (1 + np.exp(k1 * (G_f - delta1)))
    Qalpha_AF = gama2 / (1 + np.exp(k2 * (A_f - delta2)))
    Q_f = Qg_AF * Qalpha_AF
    return Q_f


def edge_pres_est(fuseG, fuseA, imG, imA):
    bimap = (imG > fuseG).astype(np.float64)
    buffer = (imG == 0).astype(np.float64)
    buffer = buffer * 0.00001
    imG = imG + buffer
    buffer1 = fuseG / imG
    buffer = (fuseG == 0).astype(np.float64)
    buffer = buffer * 0.00001
    fuseG = fuseG + buffer
    buffer2 = imG / fuseG
    Gaf = bimap * buffer1 + (1 - bimap) * buffer2
    Aaf = abs(abs(imA - fuseA) - np.pi / 2) * 2 / np.pi
    return Gaf, Aaf


def get_map(matrix, ker1, ker2):
    mat_X = conv(matrix, ker1, padding=1)
    mat_Y = conv(matrix, ker2, padding=1)
    mat_G = np.sqrt(mat_X * mat_X + mat_Y * mat_Y)
    buffer = (mat_X == 0).astype(np.float64)
    buffer = buffer * 0.00001
    mat_X = mat_X + buffer
    mat_A = np.arctan(mat_Y / mat_X)
    return mat_G, mat_A


if __name__ == '__main__':
    m1 = np.array(Image.open('../test/res/left.png'), dtype=np.float64)
    m2 = np.array(Image.open('../test/res/right.png'), dtype=np.float64)
    fim = np.array(Image.open('../test/res/fim.png'), dtype=np.float64)
    res = _q_g(m1, m2, fim)
    a = 1
