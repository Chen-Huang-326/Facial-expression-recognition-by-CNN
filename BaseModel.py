'''
This file implement a simple neural network with normal training method to establish a base model
to deal with the facial expression classification/recognition challenge.
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
from plotTool import plot_confusion
from trainingMethod import norm_train
import numpy as np
from models import FC_Net_1H

# Hyper parameters
input_neurons = 10
hidden_neurons = 160
output_neurons = 7
num_epochs = 8000
learning_rate = 0.001
re_lambda = 0.001 # apply weight decay
training_p = 0.8
# re_lambda = 0 # no weight decay



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

#print(p.data)
# Split training data into input and target
train_input = train_data.iloc[:,1:]
train_target = train_data.iloc[:,0]
#print(train_input)
#print(train_target)

# Visualization the distribution of the data
# class_freq = train_target.value_counts()
# class_freq = list(class_freq.sort_index())
#
# x_axis = list(range(0, 7))
#
# graph = plt.bar(x_axis, class_freq)
# plt.xticks(x_axis)
# plt.ylabel('Frequency')
# plt.xlabel('Emotion Label')
# plt.title('Data Distribution along with Emotion Class')
#
# plt.show()

X = torch.tensor(train_input.values).float()
Y = torch.tensor(train_target.values-1).long()


'''
Define a simple neural network with 1 hidden layer
where:
    input layer: 10 neurons, representing the 10 dimensions of inputs
    hidden layer: 10 neurons, using Sigmoid/Tanh as activation function
    output layer: 7 neurons, representing the class of emotion, using softmax as the activation function
'''
net = FC_Net_1H(input_neurons, hidden_neurons, output_neurons)

# Loss function and Optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=re_lambda)


# train a neural network
all_losses = norm_train(X, Y, net, optimizer, loss_fun, num_epochs)

# plot the loss curve
# plt.figure()
# plt.plot(all_losses)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.title('Loss Curve')
# plt.show()

# print confusion matrix

out_train = net(X)
_, predicted_train = torch.max(out_train, 1)

evaluation_train, confusion_train = plot_confusion(train_input.shape[0], output_neurons, predicted_train.long().data, Y.data)

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

print(predicted_test)
print(Y_test)

evaluation_test, confusion_test = plot_confusion(test_input.shape[0], output_neurons, predicted_test.long().data, Y_test)
print('Confusion matrix for testing:')
print(confusion_test)
print('evaluation of test data')
for label in range(len(evaluation_test)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_test[label][0], evaluation_test[label][1], evaluation_test[label][2]))

