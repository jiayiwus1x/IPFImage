import pandas as pd
from zipfile import ZipFile
import os, random, warnings
import cv2, pydicom
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import get_custom_objects
#warnings.filterwarnings('ignore')
import skimage.measure as measure
print('Tensorflow version : {}'.format(tf.__version__))
folder = '/home/dfeng/project/IPFimage/train/'
print('default folder:', folder)

def unzip(folder, filename):
    with ZipFile(folder + '/' + filename, 'r') as zf:
        zf.extractall(folder)
        
def window_image(img, img_min=-1200, img_max=200):
    
    # img_min: is the minimum windowing limit
    # img_max: is the maximum windowing limit
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    # you can then optionally rescale the image to be within [0,255] or [0,1]
    return img

def transform_to_hu(dcm):
    image = dcm.pixel_array
    image = image.astype(np.int16)

    image = set_outside_scanner_to_air(image)
  
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
        
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def set_outside_scanner_to_air(raw_pixelarrays):
    # in OSIC we find outside-scanner-regions with raw-values of -2000. 
    # Let's threshold between air (0) and this default (-2000) using -1000
    raw_pixelarrays[raw_pixelarrays <= -2000] = 0
    return raw_pixelarrays


def Load_pateint_ct_scan_HU(patient_ID, CT_PATH = folder, DIM = 512):
   
    subfolder = os.path.join(CT_PATH, patient_ID)
    filenames = os.listdir(subfolder)    
    images = np.zeros((len(filenames), DIM, DIM), dtype=float)
    fns = os.listdir(os.path.join(CT_PATH, patient_ID))
    labels = []
    for idx, fn_ID in enumerate(filenames):
        
        dcm = pydicom.dcmread(os.path.join(subfolder, fn_ID))

        image = transform_to_hu(dcm)
        if patient_ID == 'ID0013263720222217876132':
            image = image + 2000
        if patient_ID == 'ID00128637202219474716089' or patient_ID == 'ID00026637202179561894768':
            image = image + 1000
            
        image = window_image(image)
        image = creshape(image, size = 512)

        images[idx] = image  
        labels.append(int(fn_ID.split('.')[0]))
    return images[np.argsort(labels)]

def creshape(image, size = 512):    
    if image.shape[0] != size or image.shape[1] != size:
        old_x, old_y = image.shape[0], image.shape[1]
        # centering 
        x = (image.shape[0] - size) // 2
        y = (image.shape[1] - size) // 2
        image = image[x : old_x-x, y : old_y-y]
        image = image[:size, :size]
    return image

def Load_pateint_ct_scan_nf(patient_ID, CT_PATH = folder, DIM = 256):
    subfolder = os.path.join(CT_PATH, patient_ID)
    filenames = os.listdir(subfolder)   
    images = np.zeros((len(filenames), DIM, DIM, 3), dtype=np.uint8)
    fns = os.listdir(os.path.join(CT_PATH, patient_ID))
    labels = []
    for idx, fn_ID in enumerate(filenames):
        
        dcm = pydicom.dcmread(os.path.join(subfolder, fn_ID))

        image = dcm.pixel_array
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        image = creshape(image, size = 512)

        image = cv2.resize(image, (DIM,DIM), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        images[idx] = image 
        labels.append(int(fn_ID.split('.')[0]))
    return images[np.argsort(labels)] / 255.0

def get_segmentation_model(model_path = '/home/dfeng/project/IPFimage/osic_segmentation_model.h5'):
    
    class FixedDropout(tf.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = tf.keras.backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    def DiceCoef(y_trues, y_preds, smooth=1e-5, axis=None):
        intersection = tf.reduce_sum(y_trues * y_preds, axis=axis)
        union = tf.reduce_sum(y_trues, axis=axis) + tf.reduce_sum(y_preds, axis=axis)
        return tf.reduce_mean((2*intersection+smooth) / (union + smooth))

    def DiceLoss(y_trues, y_preds):
        return 1.0 - DiceCoef(y_trues, y_preds)

    get_custom_objects().update({'swish': tf.keras.layers.Activation(tf.nn.swish)})
    get_custom_objects().update({'FixedDropout':FixedDropout})
    get_custom_objects().update({'DiceCoef' : DiceCoef})
    get_custom_objects().update({'DiceLoss' : DiceLoss})
    
    print('Load segmentation model...')
    model = tf.keras.models.load_model(model_path)
    return model