import lbbutils.metrics as mt
from .tester import Tester


class Test4Metrics(Tester):
    def test_Q_MI(self):
        ret = mt.q_im(self.read('im1'), self.read('im2'), self.read('fim'))
        print(ret)
        self.assertTrue(ret != 0)

    def test_Q_G(self):
        ret = mt.q_g(self.read('im1'), self.read('im2'), self.read('fim'))
        print(ret)
        self.assertTrue(ret != 0)

    def test_Q_Y(self):
        ret = mt.q_y(self.read('im1'), self.read('im2'), self.read('fim'))
        print(ret)
        self.assertTrue(ret != 0)

    def test_Q_CB(self):
        ret = mt.q_cb(self.read('im1'), self.read('im2'), self.read('fim'))
        print(ret)
        self.assertTrue(ret != 0)
