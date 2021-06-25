'''
[Apply K-fold cross validation]
This file implement a pruning method by detecting the distinctiveness between hidden neuron pair
Distinctiveness detecting approach:
  The activation vector of the ith hidden neuron is the ith column of the activation output of the hidden layer.
  It reflects the functionality of the hidden neuron.
  If the activation matrix (the matrix of the activation output of the hidden layer) is a "N x h_d" matrix,
  where N is the number of data points, h_d is the dimension of the hidden layer (= the number of hidden neurons),
  we can get the distinctiveness matrix D by norm(x).transpose x norm(x), where x is the activation output
  and norm means normalize all the column vector by the corresponding length
  The upper triangle (expect the diagonal elements) elements reflect all the pair distinctiveness.
  i.e. d[1][2] is the element in D at the 2nd row and 3rd column,
  which is the distinctiveness between the 2nd hidden neuron and the 3rd hidden neuron
'''


# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessor import Preprocessor
import math
import numpy as np
from plotTool import plot_confusion
import matplotlib.pyplot as plt
from trainingMethod import Pruning_train
from trainingMethod import RemoveAllPruning
from trainingMethod import KeepOnePruning
from trainingMethod import get_K_fold
from models import Prune_Net
import pandas as pd
import os


# Hyper parameters
input_neurons = 10
hidden_neurons = 10
output_neurons = 7
num_epochs = 500
learning_rate = 0.001
floor_angle = math.pi/6
ceiling_angle = math.pi*5/6
training_p = 0.8
K = 10

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


# A one hidden layer feed forward Neural Network
# net = Prune_Net(input_neurons, hidden_neurons, output_neurons)
net = Prune_Net(3840, 64, 7)

# Loss function and Optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# store all losses for visualisation
all_losses = []

# train a neural network with K-fold
for epoch in range(num_epochs):
    K_loss = []
    train_acc = []
    valid_acc = []
    h_mat_norm = None
    for k in range(K):
        # validation set and training set
        validation_data = validation_datasets[k]
        validation_target = validation_targets[k]
        training_list = validation_datasets[:k] + validation_datasets[(k + 1):]
        training_targets = validation_targets[:k] + validation_targets[(k + 1):]
        training_data = torch.cat(training_list, dim=0)
        training_target = torch.cat(training_targets, dim=0)

        _, Y_pre_train, loss_val = Pruning_train(training_data, training_target, net, optimizer, loss_fun)
        K_loss.append(loss_val)

        h_mat, Y_pre_valid,_ = Pruning_train(validation_data, validation_target, net, optimizer, loss_fun)

        h_mat_norm = h_mat

        if epoch % 50 == 0:
            # training accuracy
            predicted_train = np.argmax(Y_pre_train, axis=1)
            total_train = len(predicted_train)
            correct_train = np.sum(predicted_train == training_target.data.numpy())

            t_acc = 100 * correct_train / total_train
            train_acc.append(t_acc)

            # validation accuracy
            predicted_valid = np.argmax(Y_pre_valid, axis=1)
            total_valid = len(predicted_valid)
            correct_valid = np.sum(predicted_valid == validation_target.data.numpy())

            v_acc = 100 * correct_valid / total_valid
            valid_acc.append(v_acc)

    # calculate the averaged training and validation accuracy every 50 epochs
    loss_ave = sum(K_loss) / K
    all_losses.append(loss_ave)
    if epoch % 50 == 0:
        t_acc_ave = sum(train_acc) / K
        v_acc_ave = sum(valid_acc) / K
        print('Epoch [%d/%d] Loss: %.4f  Training Accuracy: %.2f %%  Validation Accuracy: %.2f %%'
                % (epoch + 1, num_epochs, loss_ave, t_acc_ave, v_acc_ave))

    if epoch % 200 == 0 and epoch != 0:
        # construct the distinctiveness matrix D
        D = h_mat_norm.T @ h_mat_norm
        D = np.clip(D,-1,1)
        # Covert the elements in D to angle, math.pi is 180 degree
        D_angles = np.arccos(D)
        D_upper = D_angles - np.tril(D_angles)
        # Get the index matrix of elements which are less than the floor_angle
        diff_mat = np.array(np.where((D_upper < floor_angle)&(D_upper > 0)))
        # Get the index matrix of elements which are larger than the ceiling_angle
        comp_mat = np.array(np.where(D_upper > ceiling_angle))
        # pruning the neural network
        '''
        case 1: angle < floor_angle
        that is the length of diff_mat is larger than 0
        remove one neuron in the hidden neuron pair and add the weight to the remaining one
        case 2: angle > ceiling_angle
        that is the length of comp_mat is larger than 0
        remove the pair of hidden neurons
        '''
        if diff_mat.size > 0:
            KeepOnePruning(net, diff_mat)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        if comp_mat.size > 0:
            RemoveAllPruning(net, comp_mat)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)




# plot the loss curve
plt.figure()
plt.plot(all_losses)
plt.show()

# print confusion matrix
train_input = train_data.iloc[:,1:]
train_target = train_data.iloc[:,0]

X_train = torch.tensor(train_input.values).float()
Y_train = torch.tensor(train_target.values-1).long()

_, out_train = net(X_train)
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

_, out_test = net(X_test)

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
