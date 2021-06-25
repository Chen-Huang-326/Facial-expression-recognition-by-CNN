'''
This file implements the Fine Tuning of ResNet 18/34/50/101/152
We keep the entire ResNet, only use it to get features of each image (a (100,) tensor)
Use the features gotten from ResNet to feed into a FC neural network with 1 hidden layer
Only train the 1-hidden-layer-FC-nn
'''

import torch
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn
from trainingMethod import norm_train
import numpy as np
from plotTool import plot_confusion
import torch.nn.functional as F
import argparse
from models import FC_Net_1H


# Load RestNet 18/34/50/101/152 model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

path = './data/Raw_SFEW'
label_info = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# resize images into (3, 244, 244)
target = []
files = []
inputs = []
for label in os.listdir(path):
    if label != ".DS_Store":
        path_f = os.path.join(path, label)
        label = label_info.index(label)
        for file in os.listdir(path_f):
            target.append(label)
            files.append(file)
            image = Image.open(os.path.join(path_f, file))
            input_image = preprocess(image)
            inputs.append(input_image)
            # print(input_image.shape)

# print(len(inputs))
inputs = torch.stack(inputs)
print(inputs.shape)

if torch.cuda.is_available():
    inputs = inputs.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(inputs)

print(output.shape)

target = torch.tensor(target).long()
output = output.cpu()


net = FC_Net_1H(1000, 64, 7)

# Loss function and Optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

msk = np.random.rand(len(target)) < 0.8
train_X = output[msk]
train_Y = target[msk]
test_X = output[~msk]
test_Y = target[~msk]

# train a neural network
all_losses = norm_train(train_X, train_Y, net, optimizer, loss_fun, 2000)

# plot the loss curve
# plt.figure()
# plt.plot(all_losses)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.title('Loss Curve')
# plt.show()

# print confusion matrix

out_train = net(train_X)
_, predicted_train = torch.max(out_train, 1)

evaluation_train, confusion_train = plot_confusion(train_X.shape[0], 7, predicted_train.long().data, train_Y.data)

print('Confusion matrix for training:')
print(confusion_train)
print('evaluation of train data')
for label in range(len(evaluation_train)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_train[label][0], evaluation_train[label][1], evaluation_train[label][2]))

'''
Test the neural network
'''

out_test = net(test_X)

_, predicted_test = torch.max(out_test, 1)

total_test = predicted_test.size(0)


correct_test = sum(predicted_test.data.numpy() == test_Y.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))


evaluation_test, confusion_test = plot_confusion(test_X.shape[0], 7, predicted_test.long().data, test_Y)
print('Confusion matrix for testing:')
print(confusion_test)
print('evaluation of test data')
for label in range(len(evaluation_test)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_test[label][0], evaluation_test[label][1], evaluation_test[label][2]))