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

testset_m = datasets.MNIST("./MNIST/data", train=False, download=True, transform=transform)
testloader_m = torch.utils.data.DataLoader(testset_m, batch_size=128, shuffle=True)
# dataiter_m = iter(trainloader_m)
# imgs, lbls = dataiter_m.next()
#plt.imshow(imgs[1].numpy().squeeze(), cmap="Greys_r")
#print(type(imgs), imgs.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

epochs = 5
train_losses, test_losses = [], []
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
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            net.eval()
            for imgs, lbls in testloader_m:
                imgs,lbls = imgs.to(device),lbls.to(device)
                logits = net(imgs)
                test_loss += criterion(logits, lbls)
                
                ps = torch.exp(logits)/(torch.sum(torch.exp(logits)))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == lbls.view(*top_class.shape)
                #print(top_class,lbls.view(*top_class.shape))
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        
        net.train()
        
        train_losses.append(running_loss/len(trainloader_m))
        test_losses.append(test_loss/len(testloader_m))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader_m)))
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()