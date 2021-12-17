# AMLS-Image-Classification
> The mini-project implementation of course ELEC0134: Applied Machine Learning Systems (21/22) in UCL MSc IMLS.
```
SN: 21056542
```

Brain tumor has become one of the major dangerous diseases in clinical diagnosis and treatment. This GitHub repository provides the codes supplementation of the course report, in which we apply various methods for brain tumor classification including **machine learning (ML)** approaches and **deep learning** models. 

The problem is divided into **binary classification task (task A)** and **multi-classification task (task B)**. We adopt Random Forests (RF) Classifier on task A to achieve the *96.67\%* prediction accuracy and pre-trained VGG-16 network on task B to realize *97.83\%* accuracy score. The experiment proves the effectiveness of Convolutional Neural Network (CNN) in the field of image classification, as well as high performance of the pre-trained model in specific tasks.

## Environments

You can build an environment and then install the modules from following command:
```
pip install -r requirements.txt 
```


## File Directory
```

AMLS-Image-Classification
├── README.md
├── requirements.txt
├── /dataset/
│  ├── /images/
│  ├── label.csv
├── /output/
│  ├── xxxx-xx-xx_xx:xx:xx.log
├── /pictures/
├── /runs/
├── dataloader_preprocessing.py
├── dataloader_random.py
├── ml-taskA.py
├── ml-taskB.py
├── model_mlp_softmax.py
├── model_mlp.py
├── model_cnn.py
├── model_vgg.py
├── train.py
├── run.sh

```

### File Description

```
requirements.txt
```
Provide specific list of external libraries and packages, the virtual environment builds on Anaconda.
```
dataset
```
we use a brain tumor image classification dataset[1], which contains 3000 gray-scale MRI images with 512*512 pixels, and they are organised into 4 tumor types.
```
output
```
This folder records the individual output of each training process, including train_acc, valid_acc and other criterion results produced by each epoch. They are named by the date and time of the beginning of the training.
```
pictures
```
This folder saves some image outputs in the ML training procedure, such as the ROC curve and P-R curve of different ML algorithms implemented for binary classification.
```
runs
```
This folder saves the learing curves of each neural network training process, they can be viewed by *TensorBoard* and the results are corresponding to the *output* folder.
```
dataloader_preprocessing.py
dataloader_random.py
```
These two python scripts construct dataloaders to load the dataset systematically, while *dataloader_preprocessing.py* do the pre-processing to data before passing into VGG-16 network, *dataloader_random.py* is the generalized processing before passing into other neural networks.
```
ml-taskA.ipynb
ml-taskB.ipynb
```
These two jupyter notebooks implement the ML part of each task respectively, the corresponding normalization step and cross validation step are presented as well. Each ML model can be tested separately.
```
model_mlp_softmax.py
model_mlp.py
model_cnn.py
model_vgg.py
```
The above scipts construct four models to implement the deep learning part of task B. While *model_mlp_softmax.py* is another realization of Softmax Regression, *model_mlp.py* builds a MLP model, *model_cnn* constructs simple CNN model, and *model_vgg.py* builds VGG-16 network and its pre-trained version.
```
run.sh
```
This shell script enables the *train.py* file to be proceeded without hanging up on the backstage of the system.
```
train.py
```
This file implements the training command to choose specific model from deep learning approaches and produces the test results on the dataset.


### Usage example

When starting to use, select which task to solve and choose a specifci model to implement it.

For example, if I want to test the simple *CNN model* for task B, just use the following command:
```
python train.py --model 2
```
Then it will output like this:
```
---Using model Simple CNN---
BEGIN TO LOAD DATA!
```
Which means you have begun to train successfully.
For deep learning models, the correspondance for each networks are:
* 0: Softmax Regression
* 1: MLP
* 2: Simple CNN
* 3: VGG-16
* 4: Pre-trained VGG-16

## References

[1]Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, and Swati Kanchan, “Brain Tumor 
Classification (MRI).” Kaggle, 2020, doi: 10.34740/KAGGLE/DSV/1183165. 