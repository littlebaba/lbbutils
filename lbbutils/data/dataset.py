from torch.utils.data import Dataset
from PIL import Image

class FlowerDataset(Dataset):
    """
    花朵数据集：有5种花，每种图像（600~800张），大小222M
    https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    或者https://pan.baidu.com/s/1QLCTA4sXnQAw_yvxPj9szg 提取码:58p0
    """

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        if img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[index]} isn't RGB mode.")
        label = self.images_class[index]

        if self.transform is not None:
            img = self.transform(img)

        return img,label

