import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn

config = qn.load('hyperparams.yml')

class DisNet(nn.Module):
    def __init__(self,dim_in):
        super(DisNet, self).__init__()
        self.dim_in = dim_in

        self.fc1 = nn.Linear(dim_in,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GenNet(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(GenNet, self).__init__()
        self.dim_in = dim_in

        self.fc1 = nn.Linear(dim_in,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,dim_out)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
