import os
import pandas as pd
from collections import Counter

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

    train_images_list = images_list[:2400]
    valid_images_list = images_list[2400:2700]
    test_images_list = images_list[2700:]

    train_labels_list = labels_list[:2400]
    valid_labels_list = labels_list[2400:2700]
    test_labels_list = labels_list[2700:]

    train_label = Counter(train_labels_list)
    valid_label = Counter(valid_labels_list)
    test_label = Counter(test_labels_list)

    print(train_label, '\n', valid_label, '\n', test_label, '\n')

    return images_list, labels_list

dataset_path = './dataset/'

if __name__ == '__main__':
    get_images_and_labels(dataset_path)