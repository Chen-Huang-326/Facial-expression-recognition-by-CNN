'''
This file implement an Autoencoder [by CNN (convolutional neural network)]
to get the learnt features
Autoencoder_CNN aims to extract the important features from images
Then use the extracted features for classification in a Full Connected classifier
Use the reconstructed images to compare with the raw images as the loss
'''

# import libraries
import os
from IMG_preprocessor import IMG_Preprocessor
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from models import Auto_Net

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

args = parser.parse_args(['--epochs','100'])
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# import and preprocess image data
path = './data/Raw_SFEW'
label_info = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

preprocessor = IMG_Preprocessor(path, label_info)


# get the entire data
msk = np.random.rand(preprocessor.data_num) < 1
data, target,_,_ = preprocessor.PPI_divide(msk)
print(data.shape)
print(target.shape)


# convert train/test data to tensor
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float(),torch.from_numpy(target))


# construct the data loader for train and test dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                           shuffle=True, **kwargs)

# model
model = Auto_Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

loss_func = torch.nn.MSELoss()

# mini-batch training
def train(epoch):
    model.train()
    train_loss = 0
    features = []
    targets = []
    outputs = []
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if args.cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        # clear gradient
        optimizer.zero_grad()
        feature, output = model(batch_x)
        # calculate the loss
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, batch_x)
        # add up losses of all batches
        train_loss += loss.item()
        # store loss
        loss.backward()
        optimizer.step()
        features.append(feature)
        targets.append(batch_y)
        outputs.append(output)

    train_loss /= len(train_loader)
    features = torch.cat(features).reshape(-1,256,5,3)
    targets = torch.cat(targets).reshape(-1,)
    outputs = torch.cat(outputs).reshape(-1, 3, 720, 576)
    print('Autoencoder epoch %d, loss: %.4f' % (epoch, train_loss))

    return train_loss, features, targets, outputs

Autoencoder_losses = []
dataset = 0
data_target = 0
reconstrutions = 0
for epoch in range(args.epochs):
    loss, features, targets, outputs = train(epoch+1)
    Autoencoder_losses.append(loss)
    dataset = features
    data_target = targets
    reconstructions = outputs

# # visualize the reconstruction image
# rec_im = reconstructions[0]
# rec_im = rec_im.cpu().detach()
# rec_im = (rec_im - torch.mean(rec_im))
# plt.imshow(rec_im.T)
# plt.axis('off')
# plt.show()

# # plot loss curve of autoencoder
# plt.plot(Autoencoder_losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
#
# plt.title('loss of VAE')
# plt.show()

# store extracted features
save_data = torch.cat((data_target.reshape(-1,1)+1,dataset.reshape(-1,3840)),dim=1).cpu().detach().numpy()
save_data = pd.DataFrame(save_data)
print(save_data.shape)
if not os.path.exists(os.path.join('data', 'processed_data', 'cnn_extracted_features.csv')):
    save_data.to_csv(os.path.join('data', 'processed_data', 'cnn_extracted_features.csv'))

# store the trained model
if not os.path.exists(os.path.join('data','pretrained_model')):
    os.makedirs(os.path.join('data','pretrained_model'))
torch.save(model, os.path.join('data','pretrained_model', 'autoencoder.pth'))
