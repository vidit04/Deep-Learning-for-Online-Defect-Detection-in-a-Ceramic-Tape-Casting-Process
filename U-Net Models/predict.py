# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 00:41:00 2020

@author: guptav
"""

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import segmentation_models as sm
from keras import backend as K
import keras
import tensorflow as tf
import efficientnet.tfkeras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def accuracy5(y_true, y_pred, th=0.5):

    y_pred = tf.argmax(y_pred, axis =3)
    y_pred = tf.one_hot(y_pred , 4)
    
    y_pred   = K.max(y_pred, axis =-2)
    y_pred   = K.max(y_pred, axis =-2)


    y_true  = K.max(y_true, axis= -2)
    y_true   = K.max(y_true, axis= -2)
    equal = K.equal(y_true, y_pred)
    equal_red = K.all(equal, axis =1)
    
    #y_true_r = K.max(y_true, axis= -1)
    #y_true_f = K.greater(y_true_r,0.5)
    #(K.reduce_all(K.equal(y_true, y_pred)),axis = 1)
    
   # y_pred_f = K.greater(y_pred_r,th)
    #intersection = K.sum(y_true_f * y_pred_f)
    return K.mean(equal_red, axis=-1)

def plot_pair(images, gray=False):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10,8))
    i=0
    
    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap='gray')
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i+=1
    
    plt.show()

BACKBONE = 'efficientnetb2'
# define network parameters
n_classes = 4
#if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'softmax'

optim = keras.optimizers.Adam(lr=0.0)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), accuracy5]

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights=None, encoder_freeze=True)

model.load_weights(r"C:\Users\vid90589\Desktop\new 22.11\log_predict\ep119-val_loss0.50.h5")
model.compile(optim, total_loss, metrics)


image_list = sorted(os.listdir('other_small_defect'), key=lambda x: x.split('.')[0])

for imag in image_list:
    os.chdir("C:/Users/vid90589/Desktop/new 22.11/")
    im = cv2.imread(os.path.join('other_small_defect', imag), 1)
    print(imag)
    im = cv2.resize(im, (512,512))
    im_feed = np.reshape(im, (1,512,512,3))
    
    pred = model.predict(im_feed)
    
    #plot = pred[0,:,:,:]
    #plt.imshow(plot[:,:,3])
    #plt.show()
    mask = np.argmax(pred, axis=3)
    mask = np.reshape(mask,(512,512))

    image= np.zeros((512,512,4),dtype =np.uint8)
    #print(mask)

    for i in range(512):
        for j in range(512):
            if mask[i,j]==0:
                image[i,j] = [230,216,173,1]
            if mask[i,j] ==1:
                image[i,j] = [0,0,160,1]
            if mask[i,j]==2:
                image[i,j] = [29,101,181,1]
            if mask[i,j]==3:
                image[i,j] = [0,0,0,0]
      
    #cv2.imshow('image',image)
    cv2.imwrite('final.png', image)
    
    #cv2.waitKey(0)
    image = cv2.imread('final.png')
    result = cv2.addWeighted(im, 1, image, 0.4, 0)
    os.chdir("C:/Users/vid90589/Desktop/new 22.11/other_small_defect_pre/")
    cv2.imwrite( imag +'.png', result)
    #plot_pair([im, ], gray=False)
    #break
