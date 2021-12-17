from numpy import imag
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter   
from dataloader_random import ImageDataset
from dataloader_preprocessing import PreProcSet
from model_mlp_softmax import MLP_softmax
from model_mlp import MLP
from model_cnn import CNN
from model_vgg import VGG16, pre_trained_VGG16
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default='./dataset/', help='The dataset saved path')
parser.add_argument('--batch_size', type=int, default=64, help='The data to be included in each epoch')
parser.add_argument('--num_workers', type=int, default=16, help='How many subprocesses to use for data loading')
parser.add_argument('--n_epochs', type=int, default=60, help='Training epochs = samples_num / batch_size')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Regularization coefficient, usually use 5 times, for example: 1e-4/5e-4/1e-5/5e-5')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout coefficient')
parser.add_argument('--device', type=int, default=2, help='The specified GPU number to be used')
parser.add_argument('--early_stop_TH', type=int, default=10, help='The theshold value of the valid_loss continue_bigger_num in early stopping criterion')
parser.add_argument('--model', type=int, default=4, help='The specifc deep learning model to be chosed')
args = parser.parse_args()

if args.model == 0:
    print("---Using model Softmax Regression---")
    model = MLP_softmax().to(args.device)
    model.initialize()
    Imgdataset = ImageDataset(args.dir_path)
elif args.model == 1:
    print("---Using model MLP---")
    model = MLP(args.dropout).to(args.device)
    model.initialize()
    Imgdataset = ImageDataset(args.dir_path)
elif args.model == 2:
    print("---Using model Simple CNN---")
    model = CNN(args.dropout).to(args.device)
    model.initialize()
    Imgdataset = ImageDataset(args.dir_path)
elif args.model == 3:
    print("---Using model VGG-16---")
    model = VGG16().to(args.device)
    Imgdataset = PreProcSet(args.dir_path)
else:
    model = pre_trained_VGG16().to(args.device)
    Imgdataset = PreProcSet(args.dir_path)

# # model = CNN(args.dropout).to(args.device)
# model = VGG16().to(args.device)
# model.initialize()

# # Imgdataset = ImageDataset(args.dir_path)
# Imgdataset = PreProcSet(args.dir_path)
train_data = Subset(Imgdataset, list(range(2400)))
valid_data = Subset(Imgdataset, list(range(2400, 2700)))
test_data = Subset(Imgdataset, list(range(2400, 3000)))
train_loader = DataLoader(train_data, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=False)
test_loader = DataLoader(test_data, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=False)

#Define the loss function and optimizer
lossfunc = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params = model.parameters(), lr = args.lr)
optimizer = torch.optim.AdamW(params = model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

def train():
    '''
    Train the Model
    '''

    best_val_acc, continue_bigger_num, batch_num = 0, 0, 0
    log_writer = SummaryWriter() # Write log file

    # Begin to train
    for epoch in range(args.n_epochs):
        model.train() # Dropout added in Model Architecture
        train_loss, correct = 0.0, 0.0
        for data in train_loader:
            batch_num += 1
            images, labels = data['image'].to(args.device), data['label'].to(args.device)
            optimizer.zero_grad()
            outputs = model(images)    # get predicted values
            loss = lossfunc(outputs, labels)  # calculate the loss function
            loss.backward()         # Loss back propagation, calculate parameter update value
            optimizer.step()        # Parameters updating
            train_loss += loss.item()*images.size(0) # item() transform the tensor value of float number
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            log_writer.add_scalar('Loss/Train', float(loss), batch_num) # Draw in Tensorboard
        total = len(train_data)
        train_acc = 100 * correct / total
        print(f'----------------EPOCH {epoch + 1}-------------------')
        print(f'Train Loss: {(train_loss / total):8.2f} | Training Acc: {train_acc:8.2f}')

        # Run through the data set each time to test for accuracy
        test_acc = test()
        
        valid_loss, valid_acc = valid()
        log_writer.add_scalar('Loss/Validation', float(valid_loss), epoch) # Write in Tensorboard
        # Early Stop, if valid_acc continuely increases from last 'early_stop_TH'(10 in default) epoch, stop training
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
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
    total = len(valid_data)
    valid_acc = 100 * correct / total
    valid_loss = valid_loss / total
    print(f'Validation Loss: {valid_loss:8.2f} | Val Acc: {valid_acc:8.2f}')
    return valid_loss, valid_acc


def test():
    '''
    Test the Model
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # NO back propagation in test set
        for data in test_loader:
            images, labels = data['image'].to(args.device), data['label'].to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Return the maximum values of each row
            correct += (predicted == labels).sum().item()
    total = len(test_data)
    test_acc = 100 * correct / total
    print(f'Testing Acc:  {test_acc:8.2f}')
    return test_acc

if __name__ == '__main__':
    best_test_acc = train()
    print("---Traing Process End---")
    print("Best Test Accuracy is: {:.3f} %".format(best_test_acc))