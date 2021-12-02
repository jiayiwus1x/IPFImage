import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
import torch
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import seaborn as sns
import cv2
from torchvision.transforms import Compose
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

import seaborn as sns
from PIL import Image
import glob,os
from imageio import imread
from efficientnet_pytorch import EfficientNet
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
import ConvNets
    
from sklearn.model_selection import KFold
import argparse

torch.manual_seed(700)
from datetime import datetime
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
        criterion = nn.CrossEntropyLoss(weight=weight.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

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
    
def training(model, files, labels, val_files, val_labels, 
             j=0, num_epochs = 5,total_step = 5000, batch_size = 64, checkpoint_name = 'conv_network_model_ConvNet_aug'):
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


        b_imgs = torch.from_numpy(np.array([dic[i] for i in val_files])).unsqueeze(1).float().to(device)
        outputs = model(b_imgs)

        b_labels = torch.from_numpy(val_labels).to(device)                       
        total = b_labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == b_labels).sum().item()
        print('val accuracy', correct/total)
        va += [correct/total]  
        timestampStr = get_time_str()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': correct/total,
            }, 'checkpoint/%s_%04d_b0'%(timestampStr,epoch))
    
    y_pred_labels = predicted.cpu().numpy() 
    y_true_labels = val_labels
    confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels, normalize= 'true')
    
    name = "checkpoint/"+checkpoint_name+"_cv%04d"%j
    if os.path.exists(name):
        print('file exist, please rename')
    else:
        torch.save(model, name)
        print('saved as ', name)
    print(np.max(acc_list))  
    if np.max(acc_list)<0.8:
        print('rerun')
        torch.manual_seed(np.random.randint(100000))
        weights_init(model)
        training(model, files, labels, val_files, val_labels, 
             j, num_epochs,total_step, batch_size, checkpoint_name +'rerun')
    else:   
        return loss_list, acc_list, va, confusion_matrix

def assign_test_val(test_index, train_index):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    test_df = all_data.iloc[test_index]
    train_df = all_data.iloc[train_index]
    files, labels = np.array(train_df['path']), np.array(train_df['label'])
    val_files, val_labels = np.array(test_df['path']), np.array(test_df['label'])
    return files, labels, val_files, val_labels

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
print('using ', device.type)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-images", dest="imgs", default='Data/CNN/ht-img.npy', help = "image dictionary path, npy file")
    parser.add_argument("-df", dest="df", default='Data/CNN/tr_data_1206.csv', help = "dataframe path for target labels, pandas file")
    parser.add_argument("-weight", dest= "weight", default=[], type=str, help = "training weight for each class(delimited list with "")")
    parser.add_argument("-balance_weight", dest="balance_weight", type=bool, default=False, help = "if doing balaced training weight for each class")
    parser.add_argument("-tot_steps", dest="total_steps",default=3000, type=int, help = "total step for each epoch(5 in total)")
    args = parser.parse_args()
    
    dic = np.load(args.imgs, allow_pickle = True).item()
    all_data = pd.read_csv(args.df, index_col = 0)
    if any(args.weight):
        weight = [int(item) for item in args.weight.split(',')]
    elif args.balance_weight == True:
        tot = sum(all_data['type'].value_counts())
        weight = tot/(all_data[['label']]).value_counts().sort_index()
        weight = weight/len(weight)
    else:
        weight = []
    print('weight each index like below')
    print(weight)
    weight = torch.tensor(weight, dtype=torch.float32)
    print('convert weight into tensor:', weight)

    kf = KFold(n_splits=5, shuffle=True)
    X = all_data['path']
    kf.get_n_splits(X)
    cv5 = []
    j =0
    cms = []



    for train_index, test_index in kf.split(X):   
        j+=1
        print('cv: ', j)

        files, labels, val_files, val_labels = assign_test_val(test_index, train_index)
        model, criterion, optimizer = prepare_model(name='efn', num_classes=5, learning_rate = 0.0005, weight = weight)
        model.to(device)

        loss_list, acc_list, va, confusion_matrix = training(model, files, labels, val_files, val_labels, j=j, num_epochs = 5,
                                           total_step = args.total_steps, batch_size = 64, checkpoint_name = 'b0_aug')

        cv5 += [va[-1]]
        cms += [confusion_matrix]


    np.save('result_%s.npy'%get_time_str(), [cv5,cms])

    size=18
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20,8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size*0.75,
              'ytick.labelsize': size*0.75,
              'axes.titlepad': 2}
    plt.rcParams.update(params)
    z = np.round(np.mean(cms, axis = 0),2)
    zstd = np.round(np.std(cms, axis = 0),2).flatten()
    plt.figure(figsize=(8,8))
    labels = [f'{v1} ± \n {v2}' for v1, v2 in
              zip(z.flatten(), zstd)]
    labels = np.asarray(labels).reshape(5,5)
    ax = sns.heatmap(z, annot=labels,  fmt='',cmap='Blues')
    # xlabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = ['emphysema', 'fibrosis', 'ground glass', 'healthy', 'micronodules']
    ax.set_xticklabels(xlabels, rotation =30)
    ax.set_yticklabels(xlabels, rotation = 360)
    plt.xlabel('Real value')
    plt.ylabel('Perdict value')
    plt.savefig('confusion_matrix_%s.png'%get_time_str())
    plt.close()


