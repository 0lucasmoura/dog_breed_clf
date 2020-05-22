import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ConvNet(nn.Module):
    
    def __init__(self, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(ConvNet, self).__init__()

        # resnet definition
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(num_ftrs, output_dim)

    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        x = self.resnet(x)
        return x
