import os
import pandas as pd
import numpy as np
import torch

import numpy.linalg as la
from torch.autograd import Variable

from PIL import Image
import glob,os
from imageio import imread
import numpy.linalg as la
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision import transforms
from torch.autograd import Variable
from torchvision.transforms import Compose
import cv2

from PIL import Image
import glob,os
from imageio import imread
from efficientnet_pytorch import EfficientNet
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
import ConvNets
from datetime import datetime

def cv(df, r=0.8):
    index = np.random.choice(len(df), size = int(len(df)*r), replace = False)
    train_df = df.iloc[index].reset_index()
    files = np.array(train_df['path'].sort_index())
    labels = np.array(train_df['label'].sort_index())
    
    val_df = df.drop(index).reset_index()
    val_files = np.array(val_df['path'].sort_index())
    val_labels = np.array(val_df['label'].sort_index())
    return files, labels, val_files, val_labels


def get_time_str():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y(%H:%M:%S)")
    return timestampStr

def get_img(filename, rpath):
    img = Image.open(rpath + filename)
    img = np.asarray(img)
    
    img = (img-np.min(img))
    img = img/(np.max(img) +1)
    return img  

def cv(df, r=0.8):
    index = np.random.choice(len(df), size = int(len(df)*r), replace = False)
    train_df = df.iloc[index].reset_index()
    files = np.array(train_df['path'].sort_index())
    labels = np.array(train_df['label'].sort_index())
    
    val_df = df.drop(index).reset_index()
    val_files = np.array(val_df['path'].sort_index())
    val_labels = np.array(val_df['label'].sort_index())
    return files, labels, val_files, val_labels

def augmentation(npimg, rnum=0):
    r2 = np.random.rand(3)
    if rnum == 0:
        return npimg
    if rnum == 1:
        return cv2.rotate(npimg, cv2.ROTATE_90_CLOCKWISE)
    if rnum == 2:
        return cv2.rotate(npimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rnum == 3:
        return cv2.rotate(npimg, cv2.ROTATE_180)
    if rnum == 4:
        return cv2.flip(npimg, 0)
    if rnum == 5:
        return cv2.flip(npimg, 1)
    if rnum == 6:
        return cv2.flip(npimg, -1)
#     if rnum == 7:
#         return cv2.flip(cv2.rotate(npimg, cv2.ROTATE_90_CLOCKWISE),0)
#     if rnum == 8:
#         return cv2.flip(cv2.rotate(npimg, cv2.ROTATE_90_COUNTERCLOCKWISE),0)
#     if rnum == 9:
#         return cv2.flip(cv2.rotate(npimg, cv2.ROTATE_180),0

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        
def prepare_model(name='efn', num_classes=2, learning_rate = 0.005, weight=[], in_channels=1, drop_connect_rate=0.5):
    if name == 'efn':
        model = EfficientNet.from_pretrained('efficientnet-b0',  in_channels=in_channels, 
                                             num_classes=num_classes, drop_connect_rate=drop_connect_rate)
    
    elif name == 'res':

        model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)   
    elif name =='ConvNet2':
        model = ConvNets.ConvNet2(num_classes)
    else:
        model = ConvNets.ConvNet(num_classes)
    model.eval()
    
    
    # Loss and optimizer
    if any(weight):
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

def training(model, criterion, optimizer, files, labels, dic, device, num_epochs = 5,total_step = 5000, batch_size = 64):
    loss_list = []
    acc_list = []
    va = []
    for epoch in range(num_epochs): 
        for i in range(total_step):
            # Run the forward pass
            batch_ind = np.random.randint(len(labels), size = batch_size)
            b_imgs = torch.from_numpy(np.array([augmentation(dic[files[ind]], np.random.randint(7)) for ind in batch_ind])).unsqueeze(1).float().to(device)
            b_labels = torch.from_numpy(labels[batch_ind]).to(device)

            outputs = model(b_imgs)

            loss = criterion(outputs, b_labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = b_labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == b_labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
    
    return loss_list, acc_list, model