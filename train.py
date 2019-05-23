import torch
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from model import Net
from dataset import GetLoader
import os

batch_size = 64
image_size = 28

#MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset_m = datasets.MNIST("./MNIST/data", train=True, download=True, transform=transform)
trainloader_m = torch.utils.data.DataLoader(trainset_m, batch_size=batch_size, shuffle=True)

testset_m = datasets.MNIST("./MNIST/data", train=False, download=True, transform=transform)
testloader_m = torch.utils.data.DataLoader(testset_m, batch_size=batch_size, shuffle=True)

#MNIST_M
transform_mm = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_list = os.path.join('.','mnist_m', 'mnist_m_train_labels.txt')
test_list = os.path.join('.','mnist_m','mnist_m_test_labels.txt')
trainset_mm = GetLoader(
    data_root=os.path.join('.','mnist_m', 'mnist_m_train'),
    data_list=train_list,
    transform=transform_mm
)
trainloader_mm = torch.utils.data.DataLoader(
    dataset=trainset_mm,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)
testset_mm = GetLoader(
    data_root=os.path.join('.','mnist_m', 'mnist_m_test'),
    data_list=test_list,
    transform=transform_mm
)
testloader_mm = torch.utils.data.DataLoader(
    dataset=testset_mm,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)


#target
#dataiter_mm = iter(trainloader_mm)

#source
#dataiter_m = iter(trainloader_m)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)

criterion_l = nn.CrossEntropyLoss()
criterion_d = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 100

train_losses, test_losses = [], []

count_batches = min(len(trainloader_m),len(trainloader_mm))

#lambda1 = lambda epoch: 1/((1+10*((epoch+0.0)/epochs))**(0.75))
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
for epoch in range(epochs):
    running_loss_total=0
    running_loss_d=0
    running_loss_l=0
    dataiter_mm = iter(trainloader_mm)
    dataiter_m = iter(trainloader_m)
    l = (2/(1+np.exp(-10*((epoch+0.0)/epochs))))-1
    for c in range(count_batches):
        loss_total = 0
        loss_d=0
        loss_l=0

        optimizer.zero_grad()
        #for source domain
        imgs,lbls = dataiter_m.next()
        imgs,lbls = imgs.to(device),lbls.to(device)
        imgs = torch.cat((imgs,imgs,imgs),1)
        out_l,out_d = net(imgs,l)#l==lambda
        loss_l = criterion_l(out_l,lbls)
        actual_d = torch.zeros(out_d.shape)
        actual_d = actual_d.to(device) 
        loss_d = criterion_d(out_d,actual_d)

        #for target domain
        imgs,lbls = dataiter_mm.next()
        imgs = imgs.to(device)
        _,out_d = net(imgs,l)
        actual_d = torch.ones(out_d.shape)
        actual_d = actual_d.to(device)
        loss_d += criterion_d(out_d,actual_d)

        loss_total = loss_d + loss_l
        loss_total.backward()
        optimizer.step()
        #scheduler.step()
        running_loss_total+=loss_total
        running_loss_d+=loss_d
        running_loss_l+=loss_l

    else:
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            net.eval()
            for imgs,lbls in testloader_mm:
                imgs,lbls = imgs.to(device),lbls.to(device)
                #print(logits.shape,lbls.shape)
                logits,_ = net(imgs,l)
                #lbls = lbls.view(*logits.shape)
                #print(logits.shape,lbls.shape)
                test_loss +=criterion_l(logits,lbls)

                ps = torch.exp(logits)/(torch.sum(torch.exp(logits)))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == lbls.view(*top_class.shape)
                #print(top_class,lbls.view(*top_class.shape))
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        net.train()

    
    #print("Learning Rate: {}".format(optimizer.param_groups[0]['lr']))
    print("Epoch: {}/{} ...".format(epoch+1,epochs))
    print("Lambda: {}".format(l))
    print("Total running_loss: {}".format(running_loss_total/count_batches))
    print("Domain running_loss: {}".format(running_loss_d/count_batches))
    print("Label running_loss: {}".format(running_loss_l/count_batches))
    print("Test Loss: {}".format(test_loss/len(testloader_mm)))
    print("Test accuracy: {}".format(accuracy/len(testloader_mm)))



# for e in range(epochs):
#     running_loss = 0

#     for imgs, lbls in trainloader_m:
#         imgs, lbls = imgs.to(device), lbls.to(device)
#         optimizer.zero_grad()
#         out = net(imgs)
#         loss = criterion(out, lbls)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     else:
#         test_loss = 0
#         accuracy = 0
        
#         # Turn off gradients for validation, saves memory and computations
#         with torch.no_grad():
#             net.eval()
#             for imgs, lbls in testloader_m:
#                 imgs,lbls = imgs.to(device),lbls.to(device)
#                 logits = net(imgs)
#                 test_loss += criterion(logits, lbls)
                
                # ps = torch.exp(logits)/(torch.sum(torch.exp(logits)))
                # top_p, top_class = ps.topk(1, dim=1)
                # equals = top_class == lbls.view(*top_class.shape)
                # #print(top_class,lbls.view(*top_class.shape))
                # accuracy += torch.mean(equals.type(torch.FloatTensor))

        
#         net.train()
        
#         train_losses.append(running_loss/len(trainloader_m))
#         test_losses.append(test_loss/len(testloader_m))

#         print("Epoch: {}/{}.. ".format(e+1, epochs),
#               "Training Loss: {:.3f}.. ".format(train_losses[-1]),
#               "Test Loss: {:.3f}.. ".format(test_losses[-1]),
#               "Test Accuracy: {:.3f}".format(accuracy/len(testloader_m)))
# # plt.plot(train_losses, label='Training loss')
# # plt.plot(test_losses, label='Validation loss')
# # plt.legend(frameon=False)
# # plt.show()