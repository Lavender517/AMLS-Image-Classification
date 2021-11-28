import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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

dataset_path = './dataset/'

images, labels = get_images_and_labels(dataset_path)

images_trans = []
for index in tqdm(range(len(images))): 
    img_path = images[index]
    org_img = Image.open(dataset_path + 'image/' + img_path).convert('L') # Image(512, 512)
    img = np.array(org_img)
    images_trans.append(img)

X = np.array(images_trans)
print('!')
X = X.reshape((X.shape[0], -1))
X = X / 255
print("X.shape is", X.shape)

Y = np.array(labels)
print("Y.shape is", Y.shape)