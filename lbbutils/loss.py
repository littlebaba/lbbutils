from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # self.channel = 1
        # self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        def create_window(window_size, channel):
            def gaussian(window_size, sigma):
                gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
                return gauss / gauss.sum()

            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
            return window

        if channel == 1 and self.window.data.type() == img1.data.type():
            window = create_window(self.window_size, 1)
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        def _ssim(img1, img2, window, window_size, channel, size_average=True):
            mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(1).mean(1).mean(1)

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class TVLoss(torch.nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class ColorLoss(torch.nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, img1, img2):
        n, c, h, w = img1.shape
        vec1 = img1.view((n, -1, 3))
        vec2 = img2.view((n, -1, 3))
        norm_vec1 = vec1 / torch.norm(vec1, dim=2).unsqueeze(2).repeat(1, 1, 3)
        norm_vec2 = vec2 / torch.norm(vec2, dim=2).unsqueeze(2).repeat(1, 1, 3)
        tmp = torch.sum(torch.abs(norm_vec1 - norm_vec2), dim=2)
        tmp2 = torch.clamp(tmp, -0.999999, 0.999999)
        tmp3 = torch.acos(tmp2)
        return torch.mean(tmp3)


if __name__ == '__main__':
    x1 = torch.rand((1, 3, 3, 3), requires_grad=True)
    x2 = torch.rand((1, 3, 3, 3))
    # print(x1)
    loss = SSIM()
    res = loss(x1, x2)
    print(res)
    res.backward()
    # print(x1)
