import random
import os
import json
import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), f"dataset root: {root} dose not exist."

    flower_class = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    flower_class.sort()

    class_indices = dict((cls, i) for i, cls in enumerate(flower_class))
    json_str = json.dumps(dict((i, cls) for cls, i in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应的标签信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应的标签信息
    every_class_num = []  # 存储每个类别的样本数量
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cls in flower_class:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(root, cls, i) for i in os.listdir(cls_path) if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cls]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))  # 按比例随机采样验证样本

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    print(f"{sum(every_class_num)} images were found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation")

    plot_image = True
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel("image class")
        plt.ylabel("number of images")
        plt.title("flower class distribution")
        plt.show()
    return train_images_path, train_images_label, val_images_path, val_images_label


if __name__ == '__main__':
    read_split_data(root=r"D:\迅雷下载\flower_photos")
