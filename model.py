from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 48, 5)
        self.fc1_c = nn.Linear(4*4*48, 100)
        self.fc2_c = nn.Linear(100, 100)
        self.fc3_c = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4*4*48)
        x = F.relu(self.fc1_c(x))
        x = F.relu(self.fc2_c(x))
        x = self.fc3_c(x)

        return x