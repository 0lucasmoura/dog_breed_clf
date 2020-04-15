import torch
import torch.nn as nn
import torchvision.models as models
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
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        new_fc = nn.Sequential([Linear(num_ftrs, hidden_dim),
                               ])
        
        
        self.resnet.fc = new_fc # replaces resnet FC to new one / Transfer Learning
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        x = F.relu(self.resnet(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(self.hidden3(x))
        x = self.softmax(x)
        return x
