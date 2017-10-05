
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm

get_ipython().magic('matplotlib inline')


# In[ ]:

from __future__ import print_function
import numpy as np
#from mnist import MNIST
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow

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


# In[ ]:

### import data labels csv


# In[ ]:

path = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/'


# In[ ]:

## load files
train_labels = pd.read_csv(path+'train.csv')
test_labels = pd.read_csv(path+'test.csv')

print ("total images in train are {}".format(train_labels.shape[0]))
print ("total images in test are {}".format(test_labels.shape[0]))


# #### check distribution of train 

# In[ ]:

plt.figure(figsize=(18,7))
ax=sns.countplot(train_labels['label'])
sns.plt.title('Train labels distribution')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()


# In[ ]:

train_labels['label'].unique()


# In[ ]:

del train_labels, test_labels


# ##### looks like pretty even stevens across categories although some categories such as rice, sugar,beans,flour,corn, and fish have sub 100 n sizes

# In[ ]:

#### let import the images now --- I can import all as such rather than flowfromdirectory; 

#### TODO: Write a code for flowfromdir in case of single folder


# In[ ]:

# # function to read images as arrays
# def read_image(img_path,img_size=256):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (img_size,img_size)) # you can resize to  (128,128) or (256,256)
#     return img


# In[ ]:

# ## set path for images
# TRAIN_PATH = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/train_img/'
# TEST_PATH = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/test_img/'


# In[ ]:

# # load data
# train_img, test_img = [],[]
# for img_path in tqdm(train_labels['image_id'].values):
#     train_img.append(read_image(TRAIN_PATH + img_path + '.png',256))

# print("-------------- Training images loaded ---------------")    
    
# for img_path in tqdm(test_labels['image_id'].values):
#     test_img.append(read_image(TEST_PATH + img_path + '.png',256))
    
# print("-------------- Test images loaded ---------------")    


# In[ ]:

# print("Dimensions of train is: {} and test is: {}".format(len(train_img),len(test_img)))


# In[ ]:

###### lets define all the generic functions


# In[ ]:

from sklearn.metrics import f1_score
from __future__ import division
import cv2
import numpy as np
import pandas as pd
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


# In[ ]:

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


# In[ ]:

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


# In[ ]:

GLOBAL_PATH = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/'
TRAIN_FOLDER = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/train_img/' #All train files resized to 224*224
TEST_FOLDER = 'F:/Hackerearth competition/data/a0409a00-8-dataset_dp/test_img/' #All test files in one folder
F_CLASSES = GLOBAL_PATH + 'train.csv'


# In[ ]:

df_train = pd.read_csv(F_CLASSES)
df_test = pd.read_csv(GLOBAL_PATH + 'sample_submission.csv')


# In[ ]:

labels = ['rice', 'candy', 'jam', 'coffee', 'vinegar', 'chocolate', 'sugar',
       'water', 'juice', 'milk', 'soda', 'nuts', 'chips', 'spices',
       'cereal', 'beans', 'cake', 'honey', 'flour', 'pasta', 'tomatosauce',
       'tea', 'corn', 'oil', 'fish']
label_map = {'rice':1, 'candy':2, 'jam':3, 'coffee':4, 'vinegar':5, 'chocolate':6, 'sugar':0,
       'water':7, 'juice':8, 'milk':9, 'soda':10, 'nuts':11, 'chips':12, 'spices':13,
       'cereal':14, 'beans':15, 'cake':16, 'honey':17, 'flour':18, 'pasta':19, 'tomatosauce':20,
       'tea':21, 'corn':22, 'oil':23, 'fish':24}


# In[ ]:

flatten = lambda l: [item for sublist in l for item in sublist]

x_train = []
x_test = []
y_train = []


# In[ ]:

for f, tags in tqdm(df_train.values, miniters=1000):
    img = TRAIN_FOLDER + '{}.png'.format(f)
    targets = np.zeros(25)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(img)
    y_train.append(targets)


# In[ ]:

assert len(y_train[0]) == len(labels)

print ("sizes of labels match ----- good to proceed")


# In[ ]:

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=42)


# In[ ]:

img_rows, img_cols,img_size = 224, 224, 224 # Resolution of inputs
channel = 3
num_classes = len(labels) 
batch_size = 128
nb_epoch = 10
mean_pix = np.array([102.9801, 115.9465, 122.7717])


# In[ ]:

### callbacks
callbacks = [ModelCheckpoint('Retail_V2.hdf5', monitor='val_loss', save_best_only=True, verbose=2, save_weights_only=False),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0000001),
             EarlyStopping(monitor='val_loss', patience=5, verbose=0)]


# In[ ]:

from keras.utils import np_utils


# In[ ]:

model = Sequential()
model.add(BatchNormalization(axis=1, input_shape=(224,224,3)))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(128,3,3, activation='relu'))##added
model.add(BatchNormalization(axis=1))##added
model.add(MaxPooling2D((3,3)))##added
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(len(labels) , activation='softmax'))

model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy']) ## for 1st try lr ws 1e-4
model.summary()


# In[ ]:

model.fit_generator(generator=batch_generator_train(list(zip(X_train, Y_train)), img_size, batch_size),
                      steps_per_epoch=np.ceil(len(x_train)/batch_size),
                      epochs=15,
                      verbose=1,
                      validation_data=batch_generator_train(list(zip(X_valid, Y_valid)), img_size, 16),
                      validation_steps=np.ceil(len(X_valid)/16),
                      callbacks=callbacks,
                      initial_epoch=0)


# In[ ]:

for f, tags in tqdm(df_test.values, miniters=1000):
    img = TEST_FOLDER + '{}.png'.format(f)
    x_test.append(img)

batch_size_test = 128
len_test = len(x_test)
x_tst = []
yfull_test = []


# In[ ]:

probs = model.predict_generator(predict_generator(x_test,IMG_SIZE=img_size,batch_size_test), steps=np.ceil(len(x_test)/batch_size_test),verbose=1)


# In[ ]:

TTA_steps = 30

for k in range(0, TTA_steps):
    print(k)
    probs = model.predict_generator(predict_generator(x_test,IMG_SIZE=img_size,batch_size_test), steps=np.ceil(len(x_test)/batch_size_test),verbose=1)
    yfull_test.append(probs)
    k += 1

result = np.array(yfull_test[0])

for i in range(1, TTA_steps):
    result += np.array(yfull_test[i])
result /= TTA_steps

res = pd.DataFrame(result, columns=labels)
preds = []


# In[ ]:

res1 = res.copy()


# In[ ]:

preds = res1.idxmax(axis=1)


# In[ ]:

df_test['label'] = preds


# In[ ]:

df_test.to_csv('submission_conv1_tta.csv', index=False)


# In[ ]:

### gives 0.54 something on public LB

