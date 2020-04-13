import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class ConvNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(ConvNet, self).__init__()
        
        # define all layers, here
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.sig(x)
        return x
