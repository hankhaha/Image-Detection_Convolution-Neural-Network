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

with tf.device('/gpu:0'):

        def load_pictures():
                pics = []
                labels = []

                for k, v in map_characters.items(): # k : number v:characters labels
                        pictures = [k for k in glob.glob(imgsPath + "/" + v + "/*")]
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
print "X_test original shape", X_test.shape
print "y_train original shape", y_train.shape
print "y_test original shape", y_test.shape


# Convert each image (42*42*3) to an array 5292
dimData=np.prod(X_train.shape[1:])
X_train=X_train.reshape(X_train.shape[0],dimData)
X_test=X_test.reshape(X_test.shape[0],dimData)

# MLP for multi-class softmax classification
with tf.device('/gpu:0'):
    start_time=time.time()
    model_MLP= Sequential()

    model_MLP.add(Dense(512,activation='relu', input_shape=(5292,)))
    model_MLP.add(Dropout(0.2))
    model_MLP.add(Dense(512, activation='relu'))
    model_MLP.add(Dropout(0.2))
    #drop-out layer for avoid over-fitting
    model_MLP.add(Dense(512, activation='relu'))
    model_MLP.add(Dropout(0.2))
    model_MLP.add(Dense(512, activation='relu'))
    model_MLP.add(Dropout(0.2))
    model_MLP.add(Dropout(0.2))
    model_MLP.add(Dense(512, activation='relu'))
    model_MLP.add(Dropout(0.2))
    model_MLP.add(Dropout(0.2))
    model_MLP.add(Dense(512, activation='relu'))
    model_MLP.add(Dropout(0.2))
    model_MLP.add(Dense(18,activation='softmax'))

    sgd=SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
    model_MLP.compile(loss='categorical_crossentropy',
                     optimizer=sgd,
                     metrics=['accuracy'])

    start=time.time()
    MLP=model_MLP.fit(X_train,y_train,
                 epochs=30,
                 batch_size=64,
                 validation_data=(X_test, y_test))
    score = model_MLP.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    end=time.time()
    print "Elapsed Time" + ":" + str(round((start_time - end) / 60, 2)) + "minutes"


    def plot_train_history(MLP, train_metrics, val_metrics):
        plt.plot(MLP.history.get(train_metrics), '-o')
        plt.plot(MLP.history.get(val_metrics), '-o')
        plt.ylabel(train_metrics)
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_train_history(MLP, 'loss', 'val_loss')

plt.subplot(1, 2, 2)
plot_train_history(MLP, 'acc', 'val_acc')

plt.show()