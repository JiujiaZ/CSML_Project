import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp_classifier(nn.Module):
    def __init__(self, input_dim=784, h_dim=500, y_dim=10, if_bn=True):
        super().__init__()
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.y_dim)

        if if_bn:
            self.bn1 = nn.BatchNorm1d(self.h_dim)
        else:
            self.bn1 = lambda x: x

    def forward(self, x, y=None):
        if y is not None:
            x = torch.flatten(x, start_dim=1)
            y = torch.flatten(y, start_dim=1)
            x = torch.cat([x, y], dim=1)
        x = x.view(-1, self.input_dim)
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.softmax(self.fc2(h), dim = -1)
        return h

        
class densenet_encoder(nn.Module):
    def __init__(self,  input_dim=784, h_dim=500, z_dim=50, if_bn=True):
        super().__init__()
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.input_dim=input_dim
        
        self.fc1 = nn.Linear(input_dim, self.h_dim)
        self.fc21 = nn.Linear(self.h_dim, self.z_dim)
        self.fc22 = nn.Linear(self.h_dim, self.z_dim)

        if if_bn:
            self.bn1 = nn.BatchNorm1d(self.h_dim)
        else:
            self.bn1 = lambda x:x

    def forward(self, x, y=None):
        if y is not None:
            x = torch.flatten(x, start_dim=1)
            y = torch.flatten(y, start_dim=1)
            x=torch.cat([x, y], dim=1)
        x=x.view(-1,self.input_dim)
        h=F.relu(self.bn1(self.fc1(x)))
        mu=self.fc21(h)
        std=torch.nn.functional.softplus(self.fc22(h))
        return mu, std
        

class densenet_decoder(nn.Module):
    def __init__(self,o_dim=1,h_dim=500, z_dim=50, if_bn=True):
        super().__init__()
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.o_dim=o_dim

        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.o_dim*784)
        if if_bn:
            self.bn1 = nn.BatchNorm1d(self.h_dim)
        else:
            self.bn1 = lambda x:x
        
    def forward(self,z, y=None):
        if y is not None:
            y = torch.flatten(y, start_dim=1)
            z = torch.cat([y, z], dim=1)
        h=F.relu(self.bn1(self.fc1(z)))
        h=torch.sigmoid(self.fc2(h))
        return h.view(-1,self.o_dim,28,28)


