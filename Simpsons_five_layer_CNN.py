# -*- coding: utf-8 -*-
# import packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import h5py
import glob
import time
from random import shuffle
from collections import Counter
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

# character label-encoding
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

img_width = 42
img_height = 42


num_classes = len(map_characters)

pictures_per_class = 1000
test_size = 0.15

imgsPath = "/home/hank0327/.keras/datasets/train"

start_time=time.time()
with tf.device('/gpu:0'):

        def load_pictures():
                pics = []
                labels = []

                for k, v in map_characters.items(): # k : number v:characters labels
                        pictures = [k for k in glob.glob(imgsPath + "/" + v + "/*")]
                        print v + " : " + str(len(pictures))
                        for i, pic in enumerate(pictures):
                                tmp_img = cv2.imread(pic)


                                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                                tmp_img = cv2.resize(tmp_img, (img_height, img_width))
                                pics.append(tmp_img)
                                labels.append(k)
                return np.array(pics), np.array(labels)

        def get_dataset(save=False, load=False):
                X, y = load_pictures()
                y = pd.get_dummies(y)
                y=y.values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                X_train = X_train.astype('float32') / 255.
                X_test = X_test.astype('float32') / 255.
                print "Train", X_train.shape, y_train.shape
                print "Test", X_test.shape, y_test.shape

                return X_train, X_test, y_train, y_test

# Obtain training and validation dataset
X_train, X_test, y_train, y_test = get_dataset(save=False, load=False)

print "X_train original shape", X_train.shape

end=time.time()

print "Elapsed Time" + ":"+str(round(start_time - end, 2))+"seconds"
# Convolution Neural Network: we use 6-layers CNN fully connected hidden layers,including Dropout layers to avoid overfiting issue

def create_model_four_conv(input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax'))

        return model;


# Image Shape (42,42,3)
model = create_model_four_conv((img_height, img_width, 3))
model.summary()  # check the structure of model

# Before training the network, we need to define loss function and training function, we use SGD
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

batch_size = 32
epochs = 30

with tf.device('/gpu:0'):

        start_time=time.time()
        history = model.fit(X_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_test, y_test),
                 shuffle=True,
                 callbacks=[LearningRateScheduler(lr_schedule)])
        end_time=time.time()
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print "Elapsed Time" + ":" + str(round((start_time - end_time) / 60, 2)) + "minutes"


import matplotlib.pyplot as plt


def plot_train_history(history, train_metrics, val_metrics):
        plt.plot(history.history.get(train_metrics), '-o')
        plt.plot(history.history.get(val_metrics), '-o')
        plt.ylabel(train_metrics)
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_train_history(history, 'loss', 'val_loss')

plt.subplot(1, 2, 2)
plot_train_history(history, 'acc', 'val_acc')

plt.show()