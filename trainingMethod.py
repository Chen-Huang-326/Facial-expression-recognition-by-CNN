'''
The file implements all training methods:
  1. normal training
  2. K-fold cross validation
  3. distinctiveness pruning training and corresponding methods
  4. dropout process
'''
# import libraries
import torch
from preprocessor import DataFrameDataset
import torch.nn.functional as F


# obtain the K_fold validation datasets
def get_K_fold(train_data, K):
    # Get dataset for K-fold cross-validation
    batch_size = int(train_data.shape[0] / K) + 1
    train_dataset = DataFrameDataset(df=train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_datasets = []
    validation_targets = []
    for step, (batch_x, batch_y) in enumerate(train_loader):
        X = batch_x
        Y = batch_y.long()
        validation_datasets.append(X)
        validation_targets.append(Y)

    return validation_datasets, validation_targets


# K-fold cross validation
def K_fold (validation_datasets, validation_targets, net, loss_fun, optimizer, num_epochs, K):
    all_losses = []
    for epoch in range(num_epochs):
        K_loss = []
        train_acc = []
        valid_acc = []
        for k in range(K):
            # validation set and training set
            validation_data = validation_datasets[k]
            validation_target = validation_targets[k]
            training_list = validation_datasets[:k]+validation_datasets[(k+1):]
            training_targets = validation_targets[:k]+validation_targets[(k+1):]
            training_data = torch.cat(training_list, dim=0)
            training_target = torch.cat(training_targets, dim=0)

            training_pred = net(training_data)
            loss = loss_fun(training_pred, training_target)
            K_loss.append(loss.item())

            # Clear the gradients before running the backward pass.
            net.zero_grad()

            # Perform backward pass
            loss.backward()

            # Calling the step function on an Optimiser makes an update to its parameters
            optimizer.step()

            # validate the model to get the validation accuracy,every
            valid_pred = net(validation_data)

            # Print information, every 50 epochs print the training accuracy and validation accuracy
            if epoch % 50 == 0:
                # training accuracy
                _, predicted_train = torch.max(training_pred, 1)
                total_train = predicted_train.size(0)
                correct_train = sum(predicted_train.data.numpy() == training_target.data.numpy())
                t_acc = 100 * correct_train / total_train
                train_acc.append(t_acc)


                # validation accuracy
                _, predicted_valid = torch.max(valid_pred, 1)
                total_valid = predicted_valid.size(0)
                correct_valid = sum(predicted_valid.data.numpy() == validation_target.data.numpy())

                v_acc = 100 * correct_valid / total_valid
                valid_acc.append(v_acc)

        # calculate the averaged training and validation accuracy every 50 epochs
        loss_ave = sum(K_loss) / K
        all_losses.append(loss_ave)
        if epoch % 50 == 0:
            t_acc_ave = sum(train_acc)/K
            v_acc_ave = sum(valid_acc)/K
            print('Epoch [%d/%d] Loss: %.4f  Training Accuracy: %.2f %%  Validation Accuracy: %.2f %%'
                  % (epoch + 1, num_epochs, loss_ave, t_acc_ave, v_acc_ave))

    return all_losses



# define a normal training method
def norm_train (X, Y, net, optimizer, loss_fun, num_epochs,):
    # store all losses for visualisation
    all_losses = []

    # train a neural network
    for epoch in range(num_epochs):
        # Clear the gradients before running the backward pass.
        optimizer.zero_grad()

        # Feed forward to get the predicted result
        Y_pre = net(X)

        # Compute loss
        loss = loss_fun(Y_pre, Y)
        all_losses.append(loss.item())

        # print progress
        if epoch % 50 == 0:
            # convert three-column predicted Y values to one column for comparison
            _, predicted = torch.max(Y_pre, 1)

            # calculate and print accuracy
            total_num = predicted.size(0)
            correct_num = predicted.data.numpy() == Y.data.numpy()

            print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
                  % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct_num) / total_num))

        # Perform backward pass
        loss.backward()

        # Calling the step function on an Optimiser makes an update to its parameters
        optimizer.step()

    return all_losses

'''
The activation vector of the ith hidden neuron is the ith column of the activation output of the hidden layer.
If the activation matrix (the matrix of the activation output of the hidden layer) is a "N x h_d" matrix,
where N is the number of data points, h_d is the dimension of the hidden layer (= the number of hidden neurons),
we can get the distinctiveness matrix D by norm(x).transpose x norm(x), where x is the activation output
and norm means normalize all the column vector by the corresponding length
The upper triangle (expect the diagonal elements) elements reflect all the pair distinctiveness.
i.e. d[1][2] is the element in D at the 2nd row and 3rd column, 
which is the distinctiveness between the 2nd hidden neuron and the 3rd hidden neuron
'''
def KeepOnePruning (net, diff_mat):
    hidden_layer = net.hidden
    output_layer = net.output
    num_hidden = len(hidden_layer.bias.data)
    print(num_hidden)
    # restrict once only remove one hidden neuron, the rest keep
    to_keep = [i for i in range(num_hidden)]
    pruned = diff_mat[0, 0]
    remain = diff_mat[1, 0]
    to_keep.remove(pruned)

    hidden_layer.weight.data[remain] += hidden_layer.weight.data[pruned]
    hidden_layer.weight.data = hidden_layer.weight.data[to_keep]
    hidden_layer.bias.data[remain] += hidden_layer.bias.data[pruned]
    hidden_layer.bias.data = hidden_layer.bias.data[to_keep]
    output_layer.weight.data = output_layer.weight.data[:, to_keep]


def RemoveAllPruning(net, comp_mat):
    hidden_layer = net.hidden
    output_layer = net.output
    num_hidden = len(hidden_layer.bias.data)
    print(num_hidden)
    # restrict once only remove one pair hidden neuron, the rest keep
    to_keep = [i for i in range(num_hidden)]
    pruned = list(comp_mat[:,0])
    to_keep = list(set(to_keep)-set(pruned))

    hidden_layer.weight.data = hidden_layer.weight.data[to_keep]
    hidden_layer.bias.data = hidden_layer.bias.data[to_keep]
    output_layer.weight.data = output_layer.weight.data[:, to_keep]

def Pruning_train(X, Y, net, optimizer, loss_fun):
    h_mat, Y_pre = net(X)
    loss = loss_fun(Y_pre, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    h_mat_norm = F.normalize(h_mat, p=2, dim=0).detach().numpy()
    Y_pre_mat = Y_pre.detach().numpy()
    return h_mat_norm, Y_pre_mat, loss.item()


'''
This method aims at implementing the dropout process 
i.e., if the dropout mask is 0.5
Step 1. before each epoch's training phase, randomly remove 50% hidden neurons
  So, in the training phase, the network only use 50% hidden neurons to do the feed forward process,
  and only updates the wights of the 50% neurons
Step 2. recover the hidden layer
  So the 50% neurons will be recovered with the corresponding weights which is trained before
Step 3. randomly remove 50% hidden neurons for next training process
(repeat step 2 and step 3)
As the designate neural network only contains fully connected layers, the sequence of the hidden neurons does not matter
Parameters:
    removed: store the removed weights
'''
def DropoutProcess (net, rem_h_weight, rem_h_bias, rem_o_weight, input_neurons, hidden_neurons, output_neurons):
    hidden_layer = net.hidden
    output_layer = net.output
    # recover the weight matrix
    if rem_h_weight is not None:
        rem_h_weight = torch.tensor(rem_h_weight)
        rem_h_bias = torch.tensor(rem_h_bias)
        rem_o_weight = torch.tensor(rem_o_weight)
        hidden_layer.weight.data = torch.cat((hidden_layer.weight.data, rem_h_weight),dim=0)
        hidden_layer.bias.data = torch.cat((hidden_layer.bias.data, rem_h_bias), dim=0)
        output_layer.weight.data = torch.cat((output_layer.weight.data, rem_o_weight), dim=1)

    # randomly remove 50% hidden neurons
    mask = torch.rand(hidden_neurons) < 0.5
    # remove the weights in hidden layer
    hw_t = torch.transpose(hidden_layer.weight.data, 0, 1)
    rem_h_weight_t = torch.reshape(torch.masked_select(hw_t, ~mask), (input_neurons, -1))
    hw_t_s = torch.reshape(torch.masked_select(hw_t, mask), (input_neurons, -1))
    rem_h_weight = torch.transpose(rem_h_weight_t, 0, 1)
    hidden_layer.weight.data = torch.transpose(hw_t_s, 0, 1)
    # remove the corresponding bias in hidden layer
    rem_h_bias = torch.masked_select(hidden_layer.bias.data, ~mask)
    hidden_layer.bias.data = torch.masked_select(hidden_layer.bias.data, mask)
    # remove the corresponding weights in output layer
    rem_o_weight = torch.reshape(torch.masked_select(output_layer.weight.data, ~mask), (output_neurons, -1))
    output_layer.weight.data = torch.reshape(torch.masked_select(output_layer.weight.data, mask), (output_neurons, -1))

    # for name, para in net.named_parameters():
    #     print(name)
    #     print(para.data)
    return rem_h_weight.numpy(), rem_h_bias.numpy(), rem_o_weight.numpy()


