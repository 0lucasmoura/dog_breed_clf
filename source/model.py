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
        resnet_num_ftrs = self.resnet.fc.in_features
        new_resnet_fc = nn.Linear(resnet_num_ftrs, hidden_dim)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = new_resnet_fc # replaces resnet FC to new one / Transfer Learning
        
        # squeezenet definition
        self.sqznet = models.squeezenet1_1(pretrained=True)
        new_sqznet_conv = nn.Conv2d(512, hidden_dim, kernel_size=(1,1), stride=(1,1))
        for param in self.sqznet.parameters():
            param.requires_grad = False
        self.sqznet.classifier[1] = new_sqznet_conv
        self.sqznet.num_classes = hidden_dim
        
        # ensemble definition
        self.ensemble = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        x1 = self.resnet(x)
        x2 = self.sqznet(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.ensemble(F.relu(x))
        return x
