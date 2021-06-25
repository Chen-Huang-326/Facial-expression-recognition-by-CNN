'''
This file implement a simple neural network with K-fold cross validation method to compare with the baseline
reported in progressive compression network (19%)
The neural network contains only one hidden layer
when comparing with the baseline, the activation function of the hidden layer is a sigmoid function.
when comparing with the distinctiveness pruning and dropout, the model should be modified with:
    the activation function of the hidden layer should be changed to tanh
    the optimizer should add a re_lambda to implement L2 norm weight decay
'''

# import libraries
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data
from preprocessor import Preprocessor
from trainingMethod import K_fold
from plotTool import plot_confusion
from trainingMethod import get_K_fold
import numpy as np
from models import FC_Net_1H
import os


# Hyper parameters
input_neurons = 10
hidden_neurons = 160
output_neurons = 7
num_epochs = 500
# batch_size = 10
learning_rate = 0.001
K = 10
re_lambda = 0.001 # apply weight decay
training_p = 0.8
# re_lambda = 0 # no weight decay


def LP_features():
    # data preprocessing
    p = Preprocessor()
    # Apply PPI protocol
    msk = np.random.rand(p.data_num) < training_p
    train_data, test_data = p.PPI_partition(p.data, msk)
    # Apply SPI protocol
    # train_data, test_data = p.SPI_partition(p.data)
    # Apply SPS protocol
    # train_data, test_data = p.SPS_partition(p.data)
    print(train_data.shape)
    print(test_data.shape)

    # obtain the validation sets to be applied in the K_fold training
    validation_datasets, validation_targets = get_K_fold(train_data, K)
    print(validation_datasets[0].shape)
    print(validation_targets[0].shape)
    return train_data, test_data, validation_datasets, validation_targets


def Autoencoder_features():
    # import data
    extracted_data = pd.read_csv(os.path.join('data', 'processed_data', 'cnn_extracted_features.csv'))
    extracted_data = extracted_data.iloc[:,1:]

    # Data normalization
    # extracted_data = (extracted_data - pd.DataFrame.mean(extracted_data))/pd.DataFrame.std(extracted_data)


    # partition into train dataset and test dataset
    msk2 = np.random.rand(len(extracted_data)) < training_p
    train_data = extracted_data[msk2]
    test_data = extracted_data[~msk2]


    # obtain the validation sets to be applied in the K_fold training

    validation_datasets, validation_targets = get_K_fold(train_data, K)
    print(validation_datasets[0].shape)
    print(validation_targets[0].shape)
    return train_data,test_data,validation_datasets,validation_targets

# Using the features of LPQ and PHOG
# train_data,test_data,validation_datasets,validation_targets =LP_features()
# Using the features extracted from autoencoder
train_data,test_data,validation_datasets,validation_targets = Autoencoder_features()

''' 
Define a simple neural network with 1 hidden layer
where:
    input layer: 10 neurons, representing the 10 dimensions of inputs
    hidden layer: 10 neurons, using Sigmoid/Tanh as activation function
    output layer: 7 neurons, representing the class of emotion, using softmax as the activation function
'''
# net = FC_Net_1H(input_neurons, hidden_neurons, output_neurons)
net = FC_Net_1H(3840, 64, 7)

# Loss function and Optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                             #, weight_decay=re_lambda)

# Apply K fold training
all_losses = K_fold(validation_datasets, validation_targets, net, loss_fun, optimizer, num_epochs, K)

# plot the loss curve
# plt.figure()
# plt.plot(all_losses)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.title('Loss Curve')
# plt.show()

# print confusion matrix
train_input = train_data.iloc[:,1:]
train_target = train_data.iloc[:,0]

X_train = torch.tensor(train_input.values).float()
Y_train = torch.tensor(train_target.values-1).long()

out_train = net(X_train)
_, predicted_train = torch.max(out_train, 1)

evaluation_train, confusion_train = plot_confusion(train_input.shape[0], output_neurons, predicted_train.long().data, Y_train.data)

print('Confusion matrix for training:')
print(confusion_train)
print('evaluation of train data')
for label in range(len(evaluation_train)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_train[label][0], evaluation_train[label][1], evaluation_train[label][2]))


'''
Test the neural network
'''
test_input = test_data.iloc[:,1:]
test_target = test_data.iloc[:,0]

X_test = torch.tensor(test_input.values).float()
Y_test = torch.tensor(test_target.values-1).long()

out_test = net(X_test)

_, predicted_test = torch.max(out_test, 1)

total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
evaluation_test, confusion_test = plot_confusion(test_input.shape[0], output_neurons, predicted_test.long().data, Y_test)
print('Confusion matrix for testing:')
print(confusion_test)
print('evaluation of test data')
for label in range(len(evaluation_test)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_test[label][0], evaluation_test[label][1], evaluation_test[label][2]))



