import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

###Defining the hyperparameters####

dataset_path = './dataset/'
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)


def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''

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
    train_dataset = ImageDataset(dataset_path)
    dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
    for index, batch_data in enumerate(dataloader):
        print(index, batch_data['image'].shape, batch_data['label'].shape)