import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms

dataset_path = './dataset/'

def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''

    print("BEGIN TO LOAD DATA!")
    dataset_csv = pd.read_csv(dir_path + 'label.csv')
    data_classes = ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']
    images_list = dataset_csv['file_name']                        # images_name list
    labels_list = dataset_csv['label'].apply(data_classes.index)  # labels list

    return images_list, labels_list

class ImageDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path    # 数据集根目录
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # Convert the 3-channels RGB image to 1-channel Gray-scale image
            transforms.ToTensor() # Normalization, convert the values of X from (0,255) to (0,1)
        ])
        self.images, self.labels = get_images_and_labels(self.dir_path)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        # 支持根据给定的key来获取数据样本
        img_path = self.images[index]
        label = self.labels[index]
        img = Image.open(self.dir_path + 'image/' + img_path) # Image(512, 512)
        img = self.transform(img)
        sample = {'image': img, 'label': label}
        return sample

if __name__ == '__main__':
    Imgdataset = ImageDataset(dataset_path)
    train_size = int(len(Imgdataset)*0.8)
    valid_size = int(len(Imgdataset)*0.1)
    test_size = int(len(Imgdataset)*0.1)
    train_data, valid_data, test_data = random_split(Imgdataset, [train_size, valid_size, test_size])
    print(train_data)
    dataloader = DataLoader(train_data, batch_size = 64, shuffle=True)
    print(dataloader)
    for index, batch_data in enumerate(dataloader):
        print(index, batch_data['image'].shape, batch_data['label'].shape)