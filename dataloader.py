import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

label = pd.read_csv('./dataset/label.csv')

ImageData = []
for i in tqdm(range(label.shape[0])):  
    org_img = Image.open('./dataset/image/' + label['file_name'][i]).convert('L') #Image(512, 512), gray-scale
    img = np.array(org_img) # numpy(512, 512)
    ImageData.append(img)

X = np.array(ImageData)
Y = np.array(label['label'])
print(Y.shape)