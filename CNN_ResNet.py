#!/usr/bin/env python
# coding: utf-8

# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/CS229
# ! ls # verify that you are in the right directory


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
ufrom torchvision import *
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import time
import copy
import os

batch_size = 32
learning_rate = 1e-3

transforms = transforms.Compose(
[
    transforms.Resize((480,640)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root='train_classes', transform=transforms)
valid_dataset = datasets.ImageFolder(root='valid_classes', transform=transforms)
test_dataset = datasets.ImageFolder(root='test_classes', transform=transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def imshow(inp, title=None):
    
#     inp = inp.cpu() if device else inp
#     inp = inp.numpy().transpose((1, 2, 0))
    
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
    
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
    
images, labels = next(iter(train_dataloader)) 
print("images-size:", images.shape)

out = torchvision.utils.make_grid(images)
print("out-size:", out.shape)

# imshow(out, title=[train_dataset.classes[x] for x in labels])


# In[4]:


net = models.resnet50(weights='DEFAULT')
net = net.cuda() if device else net
net


# In[5]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 128)
#try out this for dropout
#net.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.2, training=m.training))
###
net.fc = net.fc.cuda() if device else net.fc


# In[ ]:


n_epochs = 10
print_every = 10
valid_loss_min = np.Inf
test_loss_min = np.Inf
val_loss = []
val_acc = []
test_loss = []
test_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
#net.load_state_dict(torch.load('resnet.pt'))
net.train()
for epoch in range(n_epochs):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = net(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (valid_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(valid_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    net.train()


batch_loss = 0
total_t=0
correct_t=0
net.load_state_dict(torch.load('resnet.pt'))
with torch.no_grad():
    net.eval()
    for data_t, target_t in (test_dataloader):
        data_t, target_t = data_t.to(device), target_t.to(device)
        outputs_t = net(data_t)
        loss_t = criterion(outputs_t, target_t)
        batch_loss += loss_t.item()
        _,pred_t = torch.max(outputs_t, dim=1)
        correct_t += torch.sum(pred_t==target_t).item()
        total_t += target_t.size(0)
    test_acc.append(100 * correct_t/total_t)
    test_loss.append(batch_loss/len(test_dataloader))
    network_learned = batch_loss < test_loss_min
    print(f'test loss: {np.mean(test_loss):.4f}, test acc: {(100 * correct_t/total_t):.4f}\n')


# In[ ]:


fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')
plt.savefig("train-validation-accuracy")


# In[ ]:


def imshow(inp, title=None):
    
    inp = inp.cpu() if device else inp
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
def visualize_model(net, num_images=4):
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        if device:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy() if device else preds.numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(2, num_images//2, images_so_far)
            ax.axis('off')
            ax.set_title('predictes: {}'.format(test_dataset.classes[preds[j]]))
            imshow(inputs[j])
            
            if images_so_far == num_images:
                return 

plt.ion()
visualize_model(net)
plt.ioff()

