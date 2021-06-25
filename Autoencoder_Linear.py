'''
This file implement a FC neural network which uses the extracted features (from Autoencoder) as input
and do the classification of facial expression recognition
'''

# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from plotTool import plot_confusion
from trainingMethod import get_K_fold
from trainingMethod import K_fold
import pandas as pd
from models import FC_Net_1H


# hyper parameters
K = 10
num_epochs = 500
train_partition = 0.8
learning_rate = 0.0001

# import data from stored feature file
extracted_data = pd.read_csv(os.path.join('data', 'processed_data', 'cnn_extracted_features.csv'))
extracted_data = extracted_data.iloc[:,1:]


# partition into train dataset and test dataset
msk2 = np.random.rand(len(extracted_data)) < train_partition
train_set = extracted_data[msk2]
test_set = extracted_data[~msk2]


# obtain the validation sets to be applied in the K_fold training
validation_datasets, validation_targets = get_K_fold(train_set, K)
print(validation_datasets[0].shape)
print(validation_targets[0].shape)

# build the classifier
classifier = FC_Net_1H(3840, 64, 7)
# Loss function and Optimizer
loss_fun2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

# Apply K fold training
all_losses = K_fold(validation_datasets, validation_targets, classifier, loss_fun2, optimizer2, num_epochs, K)

# print confusion matrix
train_input = train_set.iloc[:,1:]
train_target = train_set.iloc[:,0]

X_train = torch.tensor(train_input.values).float()
Y_train = torch.tensor(train_target.values-1).long()

out_train = classifier(X_train)
_, predicted_train = torch.max(out_train, 1)

evaluation_train, confusion_train = plot_confusion(train_input.shape[0], 7, predicted_train.long().data, Y_train.data)

print('Confusion matrix for training:')
print(confusion_train)
print('evaluation of train data')
for label in range(len(evaluation_train)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_train[label][0], evaluation_train[label][1], evaluation_train[label][2]))


'''
Test the neural network
'''
test_input = test_set.iloc[:,1:]
test_target = test_set.iloc[:,0]

X_test = torch.tensor(test_input.values).float()
Y_test = torch.tensor(test_target.values-1).long()

out_test = classifier(X_test)

_, predicted_test = torch.max(out_test, 1)

total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
evaluation_test, confusion_test = plot_confusion(test_input.shape[0], 7, predicted_test.long().data, Y_test)
print('Confusion matrix for testing:')
print(confusion_test)
print('evaluation of test data')
for label in range(len(evaluation_test)):
    print('class %d: recall [%.2f %%]; precision [%.2f %%]; specificity [%.2f %%]' %
          (label, evaluation_test[label][0], evaluation_test[label][1], evaluation_test[label][2]))