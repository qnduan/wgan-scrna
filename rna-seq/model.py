import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import qn

class DisNet(nn.Module):
    def __init__(self,dim_in):
        super(DisNet, self).__init__()
        self.dim_in = dim_in

        self.fc1 = nn.Linear(dim_in,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,1)

    def forward(self,x):
        x = F.leaky_relu(self.fc1(x),negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x),negative_slope=0.2)
        x = self.fc3(x)
        return x


class GenNet(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(GenNet, self).__init__()
        self.dim_in = dim_in

        self.fc1 = nn.Linear(dim_in,600)
        self.fc2 = nn.Linear(600,600)
        self.fc3 = nn.Linear(600,dim_out)

    def forward(self,x):
        x = F.leaky_relu(self.fc1(x),negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x),negative_slope=0.2)
        x = F.relu(self.fc3(x))
        return x
