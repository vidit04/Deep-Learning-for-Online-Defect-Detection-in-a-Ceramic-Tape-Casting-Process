
import segmentation_models as sm
from keras import backend as K
import keras
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from data_generator_new import DataGenerator
from data_generator_val_new import DataGenerator_val
#from tensorboard_callbacks import TrainValTensorBoard, TensorBoardMask
from utils_new import generate_missing_json, generate_missing_json_val
from config import model_name, n_classes
from models_new import unet, fcn_8
#from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#import tensorflow as tf

import tensorflow as tf
import efficientnet.tfkeras
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#tf.compat.v1.placeholder()

#from tensorflow.python.platform import build_info as tf_build_info
#print(tf_build_info.cuda_version_number)
#print(tf_build_info.cudnn_version_number)

def lr_schedule(epoch,lr):
  if epoch < 150:   
      return 0.01
  elif epoch < 200:
      return 0.001
  elif epoch < 220:
      return 0.0001
  #elif epoch < 53:
  #    return 0.000005
  #elif epoch < 55:
  #    return 0.000001
  #else:

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


def accuracy6(y_true, y_pred, th=0.6):
    y_true  = K.max(y_true, axis= -1)
    y_true   = K.max(y_true, axis= -1)
    y_true_r = K.max(y_true, axis= -1)
    y_true_f = K.greater(y_true_r,0.5)
    
    y_pred   = K.max(y_pred, axis =-1)
    y_pred   = K.max(y_pred, axis =-1)
    y_pred_r = K.max(y_pred, axis =-1)
    y_pred_f = K.greater(y_pred_r,th)
    #intersection = K.sum(y_true_f * y_pred_f)
    return K.mean(K.equal(y_true_f, y_pred_f), axis=-1)

def accuracy7(y_true, y_pred, th=0.7):
    y_true  = K.max(y_true, axis= -1)
    y_true   = K.max(y_true, axis= -1)
    y_true_r = K.max(y_true, axis= -1)
    y_true_f = K.greater(y_true_r,0.5)
    
    y_pred   = K.max(y_pred, axis =-1)
    y_pred   = K.max(y_pred, axis =-1)
    y_pred_r = K.max(y_pred, axis =-1)
    y_pred_f = K.greater(y_pred_r,th)
    #intersection = K.sum(y_true_f * y_pred_f)
    return K.mean(K.equal(y_true_f, y_pred_f), axis=-1)




def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split('.')[0])

if len(os.listdir('images_train')) != len(os.listdir('annotated_train')):
    generate_missing_json()

if len(os.listdir('images_val')) != len(os.listdir('annotated_val')):
    generate_missing_json_val()
    
image_paths = [os.path.join('images_train', x) for x in sorted_fns('images_train')]
#print(image_paths)
annot_paths = [os.path.join('annotated_train', x) for x in sorted_fns('annotated_train')]
#print(annot_paths)


image_paths_val = [os.path.join('images_val', x) for x in sorted_fns('images_val')]
#print(image_paths)
annot_paths_val = [os.path.join('annotated_val', x) for x in sorted_fns('annotated_val')]
#print(annot_paths)

#if 'unet' in model_name:
#    model = unet(pretrained=False, base=4)
#elif 'fcn_8' in model_name:
#    model = fcn_8(pretrained=False, base=4)

tg = DataGenerator(image_paths=image_paths, annot_paths=annot_paths,
                   batch_size=4, augment=True)

vg = DataGenerator_val(image_paths=image_paths_val, annot_paths=annot_paths_val,
                   batch_size=2, augment=True)

BACKBONE = 'efficientnetb2'
#BATCH_SIZE = 3
CLASSES = ['defect']
LR = 0.001
EPOCHS = 55

#preprocess_input = sm.get_preprocessing(BACKBONE)
print(len(tg))
for x,y in tg:
    print(x.shape)
    print(y.shape)
    break


# define network parameters
n_classes = 4
#if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'softmax'




optim = keras.optimizers.Adam(lr=0.0)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), accuracy5]
#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights=None, encoder_freeze=True)

#json_file = open('model.json', 'r')
#json_string = json_file.read()
#json_file.close()

#model = model_from_json(json_string)
#model.compile(optim, total_loss, metrics)
#model.summary()

#model.summary()
model.load_weights(r"C:\Users\guptav\Desktop\Data1 15.12\logs\ep40-val_loss0.50.h5")


#for layer in model.layers:
#    #if not isinstance(layer, layers.BatchNormalization):
#    layer.trainable = True

model.compile(optim, total_loss, metrics)
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint('C:/Users/guptav/Desktop/Data1 15.12/logs/ep{epoch:02d}-val_loss{val_loss:.2f}.h5', verbose=1, save_weights_only=True, save_best_only=False, monitor='accuracy5',mode='auto' , period= 1 ),keras.callbacks.LearningRateScheduler(lr_schedule)]

hist = model.fit_generator(generator=tg,
                    steps_per_epoch=len(tg),epochs=220, callbacks=callbacks, validation_data=vg, 
    validation_steps=len(vg), initial_epoch=40)


train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc5=hist.history['accuracy5']
val_acc5=hist.history['val_accuracy5']
#train_acc61=hist.history['accuracy6']
#val_acc61=hist.history['val_accuracy6']
#train_acc71=hist.history['accuracy7']
#val_acc71=hist.history['val_accuracy7']


train_f1_score=hist.history['f1-score']
val_f1_score=hist.history['val_f1-score']

train_iou_score=hist.history['iou_score']
val_iou_score=hist.history['val_iou_score']




xc=range(0,180,1)
#print(train_loss)
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs test_loss')
plt.grid(True)
plt.legend(['train','test'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('loss_graph.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc5)
plt.plot(xc,val_acc5)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy5')
plt.title('train_acc vs test_acc')
plt.grid(True)
plt.legend(['train','test'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Accuracy5_graph.png')


plt.figure(5,figsize=(7,5))
plt.plot(xc,train_iou_score)
plt.plot(xc,val_iou_score)
plt.xlabel('num of Epochs')
plt.ylabel('iou_score')
plt.title('train_iou_score vs test_iou_score')
plt.grid(True)
plt.legend(['train','test'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('iou_score_graph.png')

plt.figure(6,figsize=(7,5))
plt.plot(xc,train_f1_score)
plt.plot(xc,val_f1_score)
plt.xlabel('num of Epochs')
plt.ylabel('f1_score')
plt.title('train_f1_score vs test_f1_score')
plt.grid(True)
plt.legend(['train','test'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('f1_score_graph.png')

print(val_acc5[51:55])
print(val_loss[51:55])

