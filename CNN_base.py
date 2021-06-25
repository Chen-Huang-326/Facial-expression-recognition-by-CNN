'''
This file implement a baseline CNN (convolutional neural network) to classify the facial emotions
The CNN contains 3 convolutional layers with max pooling in each layer, and 3 FC layers for classification
'''

# import libraries
from IMG_preprocessor import IMG_Preprocessor
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from plotTool import plot_confusion
from models import CNN_Net


# hyper parameters/setting
parser = argparse.ArgumentParser(description='Facial Recognition Classification by CNN')
parser.add_argument('--data-partition', type=float, default=0.8, metavar='P',
                    help='train data proportion (default: 0.8)')
parser.add_argument('--batch-size', type=int, default=15, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--test-batch-size', type=int, default=15, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args(['--epochs','30'])
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# import and preprocess data
path = './data/Raw_SFEW'
label_info = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

preprocessor = IMG_Preprocessor(path, label_info)

# partition the data into train dataset and test dataset with PPI
msk = np.random.rand(preprocessor.data_num) < args.data_partition
train_data, train_target, test_data, test_target = preprocessor.PPI_divide(msk)
print(train_target.shape)
print(test_target.shape)

# convert train/test data to tensor
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_target))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_target))

# construct the data loader for train and test dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                           shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                          shuffle=True, **kwargs)

# Debug
for X, Y in test_loader:
    print(X.shape)


model = CNN_Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

loss_func = torch.nn.NLLLoss()

def train(epoch):
    model.train()
    correct = 0
    train_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if args.cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        # clear gradient
        optimizer.zero_grad()
        output = model(batch_x)
        # calculate the loss
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, batch_y)
        # add up losses of all batches
        train_loss += loss.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        # calculate the correct predict labels (sum all True)
        correct += pred.eq(batch_y.data.view_as(pred)).long().cpu().sum()
        # store loss
        loss.backward()
        optimizer.step()
        # print training result
        if step % args.log_interval == 0:
            print('Train Epoch: %d [%d/%d] Loss: %.6f}' % (
                epoch, step * len(batch_x), len(train_loader.dataset), loss.item()))

    train_loss /= len(train_loader)
    train_acc = 100. * correct / len(train_loader.dataset)
    print('\nTrain set: Average loss: %.4f, Accuracy: %d/%d (%.2f%%)\n' % (
        train_loss, correct, len(train_loader.dataset), train_acc))
    return train_loss, train_acc

def test():
    model.eval()
    test_loss = 0
    correct = 0
    all_pred = []
    all_target = []
    for batch_x, batch_y in test_loader:
        if args.cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        output = model(batch_x)
        # sum up batch loss to get the loss of the whole test set
        test_loss += loss_func(output, batch_y).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        all_pred.append(pred.cpu())
        all_target.append(batch_y.data.view_as(pred).cpu())
        correct += pred.eq(batch_y.data.view_as(pred)).long().cpu().sum()

    all_pred = torch.cat(all_pred).reshape(-1,)
    all_target = torch.cat(all_target).reshape(-1,)
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: %.4f, Accuracy: %d/%d (%.2f%%)\n' % (
        test_loss, correct, len(test_loader.dataset), test_acc))
    return all_pred, all_target, test_acc, test_loss

# Train model
train_losses = []
test_accs = []
train_accs = []
test_losses = []
for epoch in range(args.epochs):
    train_loss, train_acc = train(epoch+1)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    _,_,test_acc, test_loss = test()
    test_accs.append(test_acc)
    test_losses.append(test_loss)


# calculate the confusion matrix and precision, recall and specificity
test_pred, test_target,_,_ = test()
test_pred = test_pred.reshape(-1,)
test_target = test_target.reshape(-1,)
evaluation_test, confusion_test = plot_confusion(len(test_target), 7, test_pred, test_target)
print('Confusion matrix for testing:')
print(confusion_test)
print('evaluation of test data')
for label in range(len(evaluation_test)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_test[label][0], evaluation_test[label][1], evaluation_test[label][2]))

# visualisation
# plot the loss curve
plt.figure(figsize=(15,10))

ax1 = plt.subplot(211)
ax1.margins(x= 0.1, y =0.5)
ax1.plot(train_losses, c='g', label='train loss')
ax1.plot(test_losses, c='b', label='test loss')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(212, sharex=ax1)
ax2.margins(x= 0.1, y =0.5)
ax2.plot(train_accs, c='g', label='train accuracy')
ax2.plot(test_accs, c='b', label ='test accuracy')
ax2.set_title('Accuracy Curve')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()
plt.show()

# print useful info
print(train_losses)
test_accs = [float(i) for i in test_accs]
train_accs = [float(i) for i in train_accs]
print(test_accs)
print(train_accs)