
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
import xgboost as xgb
get_ipython().magic('matplotlib inline')


# In[1]:

from __future__ import print_function
from keras.utils import np_utils
#from mnist import MNIST
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow
import bcolz
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras import regularizers

# import packages and modules
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


# import packages and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# import packages and modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from random import randint

from sklearn.metrics import f1_score
from __future__ import division
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Merge, merge
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
#from multi_gpu import make_parallel #Available here https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os
from sklearn.utils import shuffle
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import fbeta_score
from keras.optimizers import Adam, SGD


# In[3]:

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


# In[4]:

def load_array(fname):
    return bcolz.open(fname)[:]


# In[5]:

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


# In[6]:

### https://www.kaggle.com/sashakorekov/end-to-end-resnet50-with-tta-lb-0-93

def rotate(img):
    rows = img.shape[0]
    cols = img.shape[1]
    angle = np.random.choice((10, 20, 30))#, 40, 50, 60, 70, 80, 90))
    rotation_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_M, (cols, rows))
    return img

def rotate_bound(image, size):
    #credits http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    angle = np.random.randint(10,180)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    output = cv2.resize(cv2.warpAffine(image, M, (nW, nH)), (size, size))
    return output

def perspective(img):
    rows = img.shape[0]
    cols = img.shape[1]

    shrink_ratio1 = np.random.randint(low=85, high=110, dtype=int) / 100
    shrink_ratio2 = np.random.randint(low=85, high=110, dtype=int) / 100

    zero_point = rows - np.round(rows * shrink_ratio1, 0)
    max_point_row = np.round(rows * shrink_ratio1, 0)
    max_point_col = np.round(cols * shrink_ratio2, 0)

    src = np.float32([[zero_point, zero_point], [max_point_row-1, zero_point], [zero_point, max_point_col+1], [max_point_row-1, max_point_col+1]])
    dst = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])

    perspective_M = cv2.getPerspectiveTransform(src, dst)

    img = cv2.warpPerspective(img, perspective_M, (cols,rows))#, borderValue=mean_pix)
    return img

def shift(img):
    rows = img.shape[0]
    cols = img.shape[1]

    shift_ratio1 = (random.random() * 2 - 1) * np.random.randint(low=3, high=15, dtype=int)
    shift_ratio2 = (random.random() * 2 - 1) * np.random.randint(low=3, high=15, dtype=int)

    shift_M = np.float32([[1,0,shift_ratio1], [0,1,shift_ratio2]])
    img = cv2.warpAffine(img, shift_M, (cols, rows))#, borderValue=mean_pix)
    return img

def batch_generator_train(zip_list, img_size, batch_size, is_train=True, shuffle=True):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle == True:
        random.shuffle(zip_list)
    counter = 0
    while True:
        if shuffle == True:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:

            #image = cv2.imread(file) #
            image = cv2.resize(cv2.imread(file), (img_size,img_size)) / 255.
            #image = image[:, :, [2, 1, 0]] - mean_pix

            rnd_flip = np.random.randint(2, dtype=int)
            rnd_rotate = np.random.randint(2, dtype=int)
            rnd_zoom = np.random.randint(2, dtype=int)
            rnd_shift = np.random.randint(2, dtype=int)

            if (rnd_flip == 1) & (is_train == True):
                rnd_flip = np.random.randint(3, dtype=int) - 1
                image = cv2.flip(image, rnd_flip)

            if (rnd_rotate == 1) & (is_train == True):
                image = rotate_bound(image, img_size)

            if (rnd_zoom == 1) & (is_train == True):
                image = perspective(image)

            if (rnd_shift == 1) & (is_train == True):
                image = shift(image)
                
            #cv2.resize(image, (img_size,img_size))
            
            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        yield (image_list, mask_list)

        if counter == number_of_batches:
            if shuffle == True:
                random.shuffle(zip_list)
            counter = 0

def batch_generator_test(zip_list, img_size, batch_size, shuffle=True):
    number_of_batches = np.ceil(len(zip_list)/batch_size)
    print(len(zip_list), number_of_batches)
    counter = 0
    if shuffle:
        random.shuffle(zip_list)
    while True:
        batch_files = zip_list[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:

            image = cv2.resize(cv2.imread(file), (img_size, img_size))
            #image = image[:, :, [2, 1, 0]] - mean_pix
            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield (image_list, mask_list)

        if counter == number_of_batches:
            random.shuffle(zip_list)
            counter = 0

def predict_generator(files, img_size, batch_size):
    number_of_batches = np.ceil(len(files) / batch_size)
    print(len(files), number_of_batches)
    counter = 0
    int_counter = 0

    while True:
            beg = batch_size * counter
            end = batch_size * (counter + 1)
            batch_files = files[beg:end]
            image_list = []

            for file in batch_files:
                int_counter += 1
                image = cv2.resize(cv2.imread(file), (img_size, img_size))
                #image = image[:, :, [2, 1, 0]] - mean_pix

                rnd_flip = np.random.randint(2, dtype=int)
                rnd_rotate = np.random.randint(2, dtype=int)
                rnd_zoom = np.random.randint(2, dtype=int)
                rnd_shift = np.random.randint(2, dtype=int)

                if rnd_flip == 1:
                    rnd_flip = np.random.randint(3, dtype=int) - 1
                    image = cv2.flip(image, rnd_flip)

                if rnd_rotate == 1:
                    image = rotate_bound(image, img_size)

                if rnd_zoom == 1:
                    image = perspective(image)

                if rnd_shift == 1:
                    image = shift(image)

                image_list.append(image)

            counter += 1

            image_list = np.array(image_list)

            yield (image_list)


# In[7]:

labels = ['rice', 'candy', 'jam', 'coffee', 'vinegar', 'chocolate', 'sugar',
       'water', 'juice', 'milk', 'soda', 'nuts', 'chips', 'spices',
       'cereal', 'beans', 'cake', 'honey', 'flour', 'pasta', 'tomatosauce',
       'tea', 'corn', 'oil', 'fish']
label_map = {'rice':1, 'candy':2, 'jam':3, 'coffee':4, 'vinegar':5, 'chocolate':6, 'sugar':0,
       'water':7, 'juice':8, 'milk':9, 'soda':10, 'nuts':11, 'chips':12, 'spices':13,
       'cereal':14, 'beans':15, 'cake':16, 'honey':17, 'flour':18, 'pasta':19, 'tomatosauce':20,
       'tea':21, 'corn':22, 'oil':23, 'fish':24}


# In[8]:

inv_label_map = {i: l for l, i in label_map.items()}
inv_label_map


# In[9]:

### load vgg16 arrays

feat_path = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/vgg16_features/'
vgg16_tr_feat = load_array(feat_path+'vgg16_tr_feat.dat')
vgg16_val_feat = load_array(feat_path+'vgg16_val_feat.dat')
vgg16_test_feat = load_array(feat_path+'vgg16_test_feat.dat')
train_labels = load_array(feat_path+'vgg16_train_labels.dat')
valid_labels = load_array(feat_path+'vgg16_val_labels.dat')


# In[10]:

def reshape_df(df):
    return(df.reshape(-1,df.shape[1]*df.shape[2]*df.shape[3]))


# In[11]:

vgg16_tr_feat = reshape_df(vgg16_tr_feat)
vgg16_val_feat = reshape_df(vgg16_val_feat)
vgg16_test_feat = reshape_df(vgg16_test_feat)


# In[17]:

train_labels_encoded = train_labels.argmax(axis=1)
val_labels_encoded = valid_labels.argmax(axis=1)


# In[18]:

tr = xgb.DMatrix(vgg16_tr_feat, label=train_labels_encoded)
val = xgb.DMatrix(vgg16_val_feat, label=val_labels_encoded)
#test = xgb.DMatrix(vgg16_test_feat, label=None)


# In[20]:

model = xgb.XGBClassifier()
model.fit(vgg16_tr_feat, train_labels_encoded)
print(model)


# In[22]:

from sklearn.metrics import accuracy_score
y_pred = model.predict(vgg16_val_feat)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(val_labels_encoded, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[23]:

pred = model.predict(vgg16_test_feat)


# In[24]:

pred = pd.DataFrame(pred)
pred.head()


# In[25]:

GLOBAL_PATH = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/'
df_test = pd.read_csv(GLOBAL_PATH + 'sample_submission.csv')


# In[32]:

df_test['label'] = pred
df_test['label']= df_test['label'].map(inv_label_map)


# In[33]:

df_test.head()


# In[35]:

df_test.to_csv('submission_vgg_feat_xgb_1.csv', index=False)  ### 0.53

