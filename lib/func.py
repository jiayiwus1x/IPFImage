import numpy.linalg as la
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.metrics import r2_score
from torch.autograd import Variable
import os, random, warnings
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label # for labeling regions
from skimage.measure import regionprops # for shape analysis
from skimage.color import label2rgb # for making overlay plots
from skimage.io import imread # for reading images
from skimage.feature import greycomatrix, greycoprops
warnings.simplefilter(action='ignore', category=FutureWarning)
'''Basic function library for GLCM process and some helpers for image reformat and normalization.'''

def unpack(fea_dic, slop):
    labels = []
    arr = []
    fib_imgs = []
    pids = []
    for key, value in fea_dic.items():
        labels += [52*slop[key][0]/slop[key][1]]*len(value['images'])
        pids += [key]*len(value['images'])
        arr += [value['CNN_features']]
        if not np.any(fib_imgs):
            fib_imgs = value['images']
        else:
            fib_imgs = np.concatenate([value['images'],fib_imgs])
    return labels, arr, fib_imgs, pids

def get_img(filename, rpath):
    img = Image.open(rpath + filename)
    img = np.asarray(img)
    img = (img-np.min(img))
    img = img/(np.max(img) +1)
    return img  

def img_to_torch(img):
    img = (img-np.min(img))
    img = img/(np.max(img) +1)

    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

def imgs_to_torch(imgs):
    re_imgs = []
    for img in imgs:
        img = (img-np.min(img))
        img = img/(np.max(img) +1)
        re_imgs += [img]
    re_imgs = np.array(re_imgs)
    return torch.from_numpy(re_imgs).unsqueeze(1).float()

def pred_cl(image, model):
    cls = model(img_to_torch(image))            
    return torch.max(cls.data,1)[1].detach().numpy()[0]

def max_ind(img_arr):
    return np.argmax([np.count_nonzero(img) for img in img_arr])
 
def sec_order_stats(pos, **kwargs):
    d = kwargs['d'] if 'd' in kwargs.keys() else [15, 20]
    theta = kwargs['theta'] if 'theta' in kwargs.keys() else [0, np.pi/3]
    greyco_prop_list = kwargs['greyco_prop_list'] if 'greyco_prop_list' in kwargs.keys() else ['contrast', 'homogeneity', 'correlation', 'ASM']
 
    grey_levels = np.max(pos)+1
    glcm = greycomatrix(pos,d, theta, grey_levels, symmetric=True, normed=True)

    out_row = dict(
        intensity_mean=np.mean(pos),
        intensity_std=np.std(pos))

    for c_prop in greyco_prop_list:
        for i, di in enumerate(d):
            for j, thi in enumerate(theta):
                out_row[c_prop + '_d_' +str(di)+'_th_'+str(np.round(thi,2))] = greycoprops(glcm, c_prop)[i, j]
                
    for i, di in enumerate(d):
        for j, thi in enumerate(theta):
            m = glcm[:,:,i, j]
            out_row['entropy' + '_d_' +str(di)+'_th_'+str(np.round(thi,2))] = np.sum(m*np.log2(m, out=np.zeros_like(m), where=(m!=0)))
    
    return out_row

def corp(image):
    maxx,maxy =  np.max(np.where(image!=0), axis = 1)
    minx,miny =  np.min(np.where(image!=0), axis = 1)   
    corp_img = image[minx:maxx, miny:maxy]
    return corp_img

def grey_scale(image, grey_level=20):   
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * grey_level).astype(np.uint8)
    
    return image

def get_patches(corp_img, grey_img=[], dp = 32):
    if not np.any(grey_img):
        grey_img = corp_img
    patches = []
    for x in range(0,len(corp_img[0]), dp//2):
        for y in range(0,len(corp_img[1]), dp//2):
            tup = (x, y)
            pos = corp_img[tup[0]:tup[0]+dp, tup[1]:tup[1]+dp]
            if len(np.where(pos == 0)[0])<10 and pos.size == dp**2:
                patches += [grey_img[tup[0]:tup[0]+dp, tup[1]:tup[1]+dp]]

    return patches

def get_12order_stat(image, save_img=False, dp=32, grey_level = 20, **kw2ndstats):
    
    corp_img = corp(image)
    grey_img = grey_scale(corp_img)
    dx, dy = np.shape(corp_img)

    if dx>dp and dy>dp:
        mean_score = np.sum(corp_img)/len(corp_img[corp_img != 0])
        std_score = np.std(corp_img[np.where(corp_img!=0)])
        patches = get_patches(corp_img, grey_img, dp)
        data = []
        for patch in patches:
            row = sec_order_stats(patch, **kw2ndstats)
            row['sample_mean_HU'] = mean_score
            row['sample_std_HU'] = std_score
            data.append(row)
            
        if save_img:
            return pd.DataFrame(data), patches
        return pd.DataFrame(data)

    return [],[]

def get_2nd_fea(images, dp=32, grey_level = 20, d = [15], theta=[0]):
    df1 = pd.DataFrame()
    for image in images:
        df = get_12order_stat(np.array(image),save_img=False, dp=dp, grey_level=grey_level, d=d, theta=theta)
        if np.any(df):
            df1 = df1.append(df)
            
    return df1

def get_modes(df, pid):
    
    d_des = df.describe(percentiles=[]).drop(['count','50%', 'min', 'max'])
    sk = df.skew().to_frame().T.rename(index={0:'skew'})
    kur = df.kurtosis().to_frame().T.rename(index={0:'kurtosis'})
    d_des = d_des.append(sk).append(kur)
    d_des['pid'] = pid
    d_des['idx'] = d_des.index
    c_df = d_des.pivot(index='pid',columns='idx')
    return coll_index(c_df)

def coll_index(df):
    df = df.sort_index(axis=1, level=1)
    df.columns = [f'{x}_{y}' for x,y in df.columns]
    df = df.reset_index()
    
    return df


def get_ave(x, y, db=20):
    bins = np.linspace(min(x), max(x), db)
    nums, _ = np.histogram(x, bins)
    j0 = 0
    p = []
    ps= []
    seq = np.argsort(x)
    y = np.array(y)[seq]
    for j in nums:
        
        p += [np.mean(y[j0:j0+j])]
        ps += [np.std(y[j0:j0+j])]
        j0 += j
        
    return np.array(p), np.array(ps) ,np.array(bins) + (bins[1]-bins[0])/2