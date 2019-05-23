from torch import nn
import torch.nn.functional as F

from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx,x,l):
        ctx.l = l
        return x.view_as(x)

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.neg()*ctx.l,None


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 48, 5)
        self.fc1_c = nn.Linear(4*4*48, 100)
        self.fc2_c = nn.Linear(100, 100)
        self.fc3_c = nn.Linear(100, 10)

        self.fc1_d = nn.Linear(4*4*48, 100)
        self.fc2_d = nn.Linear(100, 1)
    def forward(self, x,l):#l==lambda
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4*4*48)
        
        x_grl = GRL.apply(x,l)
        x_grl = F.relu(self.fc1_d(x_grl))
        x_grl = self.fc2_d(x_grl)
        
        x = F.relu(self.fc1_c(x))
        x = F.relu(self.fc2_c(x))
        x = self.fc3_c(x)

        return x,x_grl