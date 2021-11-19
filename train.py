from numpy import imag
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from dataloader_pytorch import ImageDataset
from model_mlp import MLP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default='./dataset/', help='The dataset saved path')
parser.add_argument('--batch_size', type=int, default=64, help='The data to be included in each epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='Training epochs = samples_num / batch_size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--device', type=int, default=6, help='The specified GPU number to be used')
args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP()
model.to(args.device)

Imgdataset = ImageDataset(args.dir_path)
train_size = int(len(Imgdataset)*0.8)
valid_size = int(len(Imgdataset)*0.1)
test_size = int(len(Imgdataset)*0.1)
train_data, valid_data, test_data = random_split(Imgdataset, [train_size, valid_size, test_size])
train_loader = DataLoader(train_data, batch_size = args.batch_size, num_workers=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size = args.batch_size, num_workers=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size = args.batch_size, num_workers=16, shuffle=True)

def train():
    '''
    Train the Model
    '''
    #Define the loss function and optimizer
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = args.lr)

    # Begin to train
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        for data in train_loader:
            images, labels = data['image'].to(args.device), data['label'].to(args.device)
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(images)    # 得到预测值
            loss = lossfunc(output, labels)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()*images.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        test()


def test():
    '''
    Test the Model
    '''
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data['image'].to(args.device), data['label'].to(args.device)
            # images, labels = data['image'], data['label']
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total

if __name__ == '__main__':
    train()