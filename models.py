'''
This file stroes all the models used in the experiments
'''

# import libraries
import torch.nn as nn
import torch
import torch.nn.functional as F

'''
A simple full-connected neural network with 1 hidden layer
'''
class FC_Net_1H(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(FC_Net_1H, self).__init__()
        self.hidden = nn.Linear(input_neurons,hidden_neurons)
        self.output = nn.Linear(hidden_neurons,output_neurons)

    def forward(self, x):
        hid = self.hidden(x)
        hid = torch.tanh(hid)
        # hid = torch.sigmoid(hid)
        out = self.output(hid)
        out = torch.softmax(out, dim=1)
        return out


'''
A simple full-connected neural network with 1 hidden layer
Aims to pune
'''
class Prune_Net(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(Prune_Net, self).__init__()
        self.hidden = nn.Linear(input_neurons,hidden_neurons)
        self.output = nn.Linear(hidden_neurons,output_neurons)


    def forward(self, x):
        hidden = self.hidden(x)
        hidden = torch.tanh(hidden)
        out = self.output(hidden)
        return hidden, out



'''
Define the baseline CNN model
'''
ndf = 64
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.main = nn.Sequential(
            # 3 * 720 * 576
            nn.BatchNorm2d(3),
            nn.Conv2d(3, ndf, kernel_size=20,stride=2),
            # nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 * 175 * 139
            nn.Conv2d(ndf, ndf*2, kernel_size=20, stride=2),
            # nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.1, inplace=True),
            # 128 *39 * 30
            nn.Conv2d(ndf*2, ndf*4, kernel_size=20,stride=2),
            # nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc1 = nn.Linear(3840, 64)
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 7)

    def forward(self, x):
        x = self.main(x)
        # print(x.shape)
        x = x.view(-1, 3840)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


'''
Define the FC_CNN parallel connection model.
There is a CNN neural network processes the images
There is a FC neural network processes the LPQ/PHOG data
Before classification, the output of CNN and output of FC will be concatenated
Use the combined vector for classification
'''
class PC_Net(nn.Module):
    def __init__(self):
        super(PC_Net, self).__init__()
        self.main = nn.Sequential(
            # 3 * 720 * 576
            nn.BatchNorm2d(3),
            nn.Conv2d(3, ndf, kernel_size=20,stride=2),
            # nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 * 175 * 139
            nn.Conv2d(ndf, ndf*2, kernel_size=20, stride=2),
            # nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.1, inplace=True),
            # 128 *39 * 30
            nn.Conv2d(ndf*2, ndf*4, kernel_size=20,stride=2),
            # nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.cnn_fc1 = nn.Linear(3840, 128)
        self.lp_fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 7)

    def forward(self, x, lp):
        x = self.main(x)
        # print(x.shape)
        x = x.view(-1, 3840)
        x = F.relu(self.cnn_fc1(x))
        x = F.dropout(x, training=self.training)
        lp = F.relu(self.lp_fc1(lp))
        x = torch.cat((x,lp),dim=1)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x



'''
Define the Autoencoder model
'''
class Auto_Net(nn.Module):
    def __init__(self):
        super(Auto_Net, self).__init__()
        self.encoder1 = nn.Sequential(
            # 3 * 720 * 576
            nn.Conv2d(3, ndf, kernel_size=20, stride=2),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, return_indices=True)
        )
        self.encoder2 = nn.Sequential(
            # 64 * 178 * 142
            nn.Conv2d(ndf, ndf * 2, kernel_size=20, stride=2),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, return_indices=True)
        )
        self.encoder3 = nn.Sequential(
            # 34 * 43 * 128
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=20, stride=2),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, return_indices=True)
        )
        self.unpool1 = nn.MaxUnpool2d(2)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(4 * ndf, 2 * ndf, kernel_size=20, stride=2, output_padding=(1,0)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.unpool2 = nn.MaxUnpool2d(2)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=20, stride=2,output_padding=(1,1)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.unpool3 = nn.MaxUnpool2d(3,2)
        self.output = nn.ConvTranspose2d(ndf, 3, kernel_size=20, stride=2)

    def forward(self, x):
        x, indices1 = self.encoder1(x)
        x, indices2 = self.encoder2(x)
        f, indices3 = self.encoder3(x)

        output = self.unpool1(f, indices3)
        output = self.decoder1(output)
        output = self.unpool2(output, indices2)
        output = self.decoder2(output)
        # print(output.shape)
        output = self.unpool3(output, indices1)
        # print(output.shape)
        output = self.output(output)
        # print(output.shape)
        output = F.relu(output)

        return f, output