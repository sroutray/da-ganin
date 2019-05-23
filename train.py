import torch
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from model import Net

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset_m = datasets.MNIST("./MNIST/data", train=True, download=True, transform=transform)
trainloader_m = torch.utils.data.DataLoader(trainset_m, batch_size=128, shuffle=True)
# dataiter_m = iter(trainloader_m)
# imgs, lbls = dataiter_m.next()
#plt.imshow(imgs[1].numpy().squeeze(), cmap="Greys_r")
#print(type(imgs), imgs.shape)
device = "cuda:0" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

epochs = 10
for e in range(epochs):
    running_loss = 0
    for imgs, lbls in trainloader_m:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    else:
        print(f"training loss: {running_loss/len(trainloader_m)}")