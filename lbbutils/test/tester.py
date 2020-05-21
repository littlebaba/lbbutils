from unittest import TestCase
import numpy as np
from PIL import Image
import os.path as osp


class Tester(TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.res_dir = osp.join(osp.dirname(__file__), 'res')
        self.img = {'im1': 'left.png', 'im2': 'right.png', 'fim': 'fim.png'}

    def read(self, key):
        return np.array(Image.open(osp.join(self.res_dir, self.img[key])), dtype=np.float64)

    def path(self, key):
        return osp.join(self.res_dir, self.img[key])

    def trans(self, key):
        img = Image.open(osp.join(self.res_dir, self.img[key])).resize((256, 256))
        img.convert('L')
        img = np.array(img)[:, :, 0]
        img = Image.fromarray(img)
        img.save(osp.join(self.res_dir, f'{key}.png'))


if __name__ == '__main__':
    test = Tester()
    x = test.read('fim')
    a = 1
