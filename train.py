import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import re
from collections import Counter
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from wang_model_CNN import Wang_CNN
from zhang_model_CNN import Zhang_CNN


#### adapted from https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844
#### and from https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1
model = Zhang_CNN() #can change to wang-model-CNN

datapath = r'C:\Users\mkara\OneDrive\Desktop\exampe3' 
batch_size = 30 #raise to improve
percent_train = 0.80

dataset = ImageFolder(datapath,transform = transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor(),
transforms.Grayscale(num_output_channels=1) ]))
#resize images then transform to tensor

train_size = int(percent_train*len(dataset))
test_size = int((len(dataset) - train_size)//2)
valid_size = int(len(dataset)-train_size-test_size)

#split into train and test
train_data, test_data = random_split(dataset, [train_size, (test_size+valid_size)])
valid_data = random_split(test_data, [test_size, valid_size])

#obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(test_size *num_train)) #finds index to stop at to get the validation set from dataset.
train_idx, valid_idx = indices[split:], indices[:split]

#define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#prepare data loader(combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
    sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

classes = ['MS', 'other']

#specify loss function
criterion = nn.CrossEntropyLoss()

#specify optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

n_epochs = 5 #30
train_losslist = []
valid_loss_min = np.Inf 

for epoch in range(1, n_epochs +1):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss +=loss.item()*data.size(0)

    model.eval()
    for data, target in valid_loader:
        output = model(data)
        loss = criterion(output, target)
        valid_loss +=loss.item()*data.size(0)
    
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    train_losslist.append(train_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    if valid_loss <=valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'good_model.pt')
        valid_loss_min = valid_loss
plt.plot(np.arange(n_epochs), train_losslist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Performance of Model 1")
plt.show()


model.load_state_dict(torch.load('good_model.pt'))
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()


# iterate over test data
for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) 
    # calculate test accuracy for each object class
    b = batch_size
    if len(target.data) < batch_size:
        b = len(target.data)
    for i in range(b):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

