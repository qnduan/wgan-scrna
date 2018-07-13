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
#         self.dp1 = nn.Dropout()
#         self.fc2 = nn.Linear(200,200)
#         self.dp2 = nn.Dropout()
        self.fc3 = nn.Linear(200,1)

    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
#         x = self.dp1(x)
#         x = F.leaky_relu(self.fc2(x))
#         x = self.dp2(x)
        x = self.fc3(x)
        return x


class GenNet(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(GenNet, self).__init__()
        self.dim_in = dim_in

        self.fc1 = nn.Linear(dim_in,600)
#         self.dp1 = nn.Dropout()
#         self.fc2 = nn.Linear(600,600)
#         self.dp2 = nn.Dropout()
        self.fc3 = nn.Linear(600,dim_out)

    def forward(self,x):
        x = F.relu(self.fc1(x))
#         x = self.dp1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dp2(x)
        # change to leaky?
        x = F.relu(self.fc3(x))
        return x
