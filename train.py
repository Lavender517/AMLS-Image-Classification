from numpy import imag
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter   
from dataloader_pytorch import ImageDataset
from model_mlp import MLP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default='./dataset/', help='The dataset saved path')
parser.add_argument('--batch_size', type=int, default=64, help='The data to be included in each epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='Training epochs = samples_num / batch_size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Regularization coefficient, usually use 5 times, for example: 1e-4/5e-4/1e-5/5e-5')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout coefficient')
parser.add_argument('--device', type=int, default=6, help='The specified GPU number to be used')
parser.add_argument('--early_stop_TH', type=int, default=10, help='The theshold value of the valid_loss continue_bigger_num in early stopping criterion')
args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP(args.dropout)
model.initialize()
model.to(args.device)

Imgdataset = ImageDataset(args.dir_path)
train_size = int(len(Imgdataset)*0.8)
valid_size = int(len(Imgdataset)*0.1)
test_size = int(len(Imgdataset)*0.1)
train_data, valid_data, test_data = random_split(Imgdataset, [train_size, valid_size, test_size])
train_loader = DataLoader(train_data, batch_size = args.batch_size, num_workers=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size = args.batch_size, num_workers=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size = args.batch_size, num_workers=16, shuffle=True)

log_writer = SummaryWriter() # Write log file

#Define the loss function and optimizer
lossfunc = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params = model.parameters(), lr = args.lr)
optimizer = torch.optim.AdamW(params = model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

def train():
    '''
    Train the Model
    '''

    model.train()
    best_val_loss, continue_bigger_num, batch_num = float('inf'), 0, 0

    # Begin to train
    for epoch in range(args.n_epochs):
        train_loss, correct = 0.0, 0.0
        for data in train_loader:
            batch_num += 1
            images, labels = data['image'].to(args.device), data['label'].to(args.device)
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            outputs = model(images)    # 得到预测值
            loss = lossfunc(outputs, labels)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()*images.size(0) # item() transform the tensor value of float number
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            log_writer.add_scalar('Loss/Train', float(loss), batch_num) # Draw in Tensorboard
        total = len(train_data)
        train_acc = 100 * correct / total
        print('Epoch:  {}  \tTraining Loss: {:.6f}    \tTraining Acc:  {:.3f} %'.format(epoch + 1, train_loss / total, train_acc))

        # Run through the data set each time to test for accuracy
        test_acc = test()
        
        valid_loss, _ = valid()
        log_writer.add_scalar('Loss/Validation', float(valid_loss), epoch) # Write in Tensorboard
        # Early Stop, if valid_acc continuely increases from last 'early_stop_TH'(5 in default) epoch, stop training
        if valid_loss <= best_val_loss:
            best_val_loss = valid_loss
            best_test_acc = test_acc
            continue_bigger_num = 0
        else:
            continue_bigger_num += 1
            if continue_bigger_num == args.early_stop_TH:
                print("EARLY STOP SATISFIES, STOP TRAINING")
                break
    
    return best_test_acc
        

def valid():
    '''
    Valid the Model
    '''
    model.eval()
    correct, valid_loss = 0, 0.0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data['image'].to(args.device), data['label'].to(args.device)
            outputs = model(images)
            loss = lossfunc(outputs, labels)
            _, predicted = torch.max(outputs.data, 1) # Return the maximum values of each row
            valid_loss += loss.item()*images.size(0)
            correct += (predicted == labels).sum().item()
    total = len(test_data)
    valid_acc = 100 * correct / total
    valid_loss = valid_loss / total
    print('Loss and Accuracy on Validation images: Validation Loss: {:.6f}    \tValidation Acc:  {:.3f} %'.format(valid_loss, valid_acc))
    return valid_loss, valid_acc


def test():
    '''
    Test the Model
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data['image'].to(args.device), data['label'].to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Return the maximum values of each row
            correct += (predicted == labels).sum().item()
    total = len(test_data)
    test_acc = 100 * correct / total
    print('Accuracy on Testing images: Testing Acc:  {:.3f} %'.format(test_acc))
    return test_acc

if __name__ == '__main__':
    best_test_acc = train()
    print("---Traing Process End---")
    print("Best Test Accuracy is: {:.3f} %".format(best_test_acc))