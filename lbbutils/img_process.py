from fnmatch import fnmatchcase as match
from os import listdir
from os.path import join
from typing import Tuple, List, Any

import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageFilter
from torch import Tensor
from torchvision.transforms import ToPILImage


class BlurImage(object):
    def __init__(self, num_div, style='C'):
        self.num_div = num_div
        self.style = style

    def __call__(self, image):
        l = Image.new('L', image.size)
        r = Image.new('L', image.size)
        blocks = []
        block_size = image.size[0] // self.num_div
        for row in range(1, self.num_div + 1):
            for col in range(1, self.num_div + 1):
                box = (block_size * (col - 1), block_size * (row - 1), block_size * (col - 1) + block_size + 1,
                       block_size * (row - 1) + block_size + 1)
                block = image.crop(box)
                blocks.append(block)
                blur_block = block.filter(ImageFilter.GaussianBlur(radius=3))
                if self.style == 'C':
                    if row % 2 == 1:
                        if (self.num_div * (row - 1) + (col - 1)) % 2 == 0:
                            l.paste(blur_block, box)
                            r.paste(block, box)
                        else:
                            l.paste(block, box)
                            r.paste(blur_block, box)
                    else:
                        if (self.num_div * (row - 1) + (col - 1)) % 2 == 1:
                            l.paste(blur_block, box)
                            r.paste(block, box)
                        else:
                            l.paste(block, box)
                            r.paste(blur_block, box)
                elif self.style == 'P':
                    if (self.num_div * (row - 1) + (col - 1)) % 2 == 0:
                        l.paste(blur_block, box)
                        r.paste(block, box)
                    else:
                        l.paste(block, box)
                        r.paste(blur_block, box)
        return l, r, image


class BlurImageRandomly:
    def __init__(self, radius=3, sigma=1):
        self.radius = radius
        self.sigma = sigma

    def __call__(self, image):
        im = np.array(image)
        left = im.copy()
        right = im.copy()
        row, col = im.shape
        filter = get_blur_filter(self.radius, self.sigma)
        mask = np.random.randint(0, 2, (row, col))

        for i in range(row):
            for j in range(col):
                if mask[j, i] == 1:
                    get_one_blurpoint(left, (j, i), filter)
                if mask[j, i] == 0:
                    get_one_blurpoint(right, (j, i), filter)
        l, r, image = Image.fromarray(left), Image.fromarray(right), Image.fromarray(im)
        return l, r, image


class ToTensors(object):
    def __call__(self, pic):
        return F.to_tensor(pic[0]), F.to_tensor(pic[1]), F.to_tensor(pic[2])


def save_img(imgs: Tuple[Tensor]) -> List[Any]:
    ret = list()
    for i, img in enumerate(imgs):
        if img.is_cuda:
            img = img.cpu().clone()
        else:
            img = img.clone()
        img.squeeze_(0)
        img_PIL = ToPILImage()(img)
        ret.append(img_PIL)
    return ret


def path_match(root, word):
    items = listdir(root)
    tmp = [item for item in items if match(item, '*-' + word + '-*')]
    return join(root, tmp[-1])


def gaussian2D(x, y, sigma=1):
    return 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / 2 * sigma ** 2)


def get_blur_filter(r, sigma=1):
    size = 2 * r + 1
    filter = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            filter[row, col] = gaussian2D(row - r, col - r, sigma=sigma)
    sum = np.sum(filter)
    filter = filter / sum
    return filter


def get_one_blurpoint(img, point, filter):
    r = filter.shape[0] // 2
    height, width = img.shape
    c_x, c_y = point
    min_x, min_y = c_x - r, c_y - r
    max_x, max_y = c_x + r, c_y + r
    tmp = 0
    for i in range(min_y, max_y + 1):
        for j in range(min_x, max_x + 1):
            now_x, now_y = j, i
            if j < 0:
                now_x = -j
            if i < 0:
                now_y = -i
            if j > width - 1:
                now_x = width - 1 - j
            if i > height - 1:
                now_y = height - 1 - i
            tmp += filter[i - min_y, j - min_x] * img[now_x, now_y]
    img[c_x, c_y] = tmp
    return img


if __name__ == '__main__':
    img = Image.open('test/res/fim.png')
    # im = np.array(img)
    # ret = np.zeros_like(im)
    # row, col = im.shape
    # filter = get_blur_filter(3)
    # for i in range(row):
    #     for j in range(col):
    #         get_one_blurpoint(im, (j, i), filter)
    # res = Image.fromarray(im)
    # res.save('test/res/fim_blured3_sigma1.png')
    bir = BlurImageRandomly()
    l, r, image = bir(img)
    l.save('test/res/bir_l.png')
    r.save('test/res/bir_r.png')
    image.save('test/res/bir_img.png')
