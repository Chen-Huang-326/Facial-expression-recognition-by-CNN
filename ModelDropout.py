'''
This file implements a model with dropout process.
The model is trained with normal process
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
from trainingMethod import DropoutProcess
from models import Prune_Net


# Hyper parameters
input_neurons = 10
hidden_neurons = 10
output_neurons = 7
num_epochs = 8000
# batch_size = 10
learning_rate = 0.001
# floor_angle = math.pi/12
# ceiling_angle = math.pi*11/12
training_p = 0.8
dropout = 0.5


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

X = torch.tensor(train_input.values).float()
Y = torch.tensor(train_target.values-1).long()


# A one hidden layer feed forward Neural Network
net = Prune_Net(input_neurons, hidden_neurons, output_neurons)

# Loss function and Optimizer
loss_fun = torch.nn.CrossEntropyLoss()
# Adam optimizer
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# Adam optimizer with regularizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# SDG optimizer
# optimizer = torch.optim.SGD(net.parameters(), lr= learning_rate)


# store all losses for visualisation
all_losses = []
# store the removed hidden weights, bias and output weights
rhW = None
rhB = None
roW = None

# train a neural network
for epoch in range(num_epochs):
    _, Y_pre_mat, loss_val = Pruning_train(X, Y, net, optimizer, loss_fun)
    all_losses.append(loss_val)

    # dropout processing
    rem_h_weight, rem_h_bias, rem_o_weight = DropoutProcess(net, rhW, rhB, roW, input_neurons, hidden_neurons, output_neurons)
    # update the removed hidden weights, bias and output weights
    rhW = rem_h_weight
    rhB = rem_h_bias
    roW = rem_o_weight
    # update the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    if epoch % 50 == 0:
        predicted = np.argmax(Y_pre_mat, axis=1)
        total_num = len(predicted)
        correct_num = np.sum(predicted == Y.data.numpy())

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss_val, 100 * correct_num / total_num))



# plot the loss curve
plt.figure()
plt.plot(all_losses)
plt.show()

# print confusion matrix

_, out_train = net(X)
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
