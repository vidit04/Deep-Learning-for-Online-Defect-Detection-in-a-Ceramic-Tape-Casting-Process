import segmentation_models as sm
from keras import backend as K
import keras
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
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
import csv  
import tensorflow as tf
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
  #elif epoch < 280:
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


BACKBONE = 'efficientnetb3'
#BATCH_SIZE = 3
CLASSES = ['defect']
LR = 0.001
EPOCHS = 55
i =0
#preprocess_input = sm.get_preprocessing(BACKBONE)
#print(len(tg))
#for x,y in tg:
#    print(x.shape)
#    print(y.shape)
#    #mask =  y[1,:,:,:]
#    #print(mask.shape)

#    for j in range(4):
#        mask = y[j,:,:,:]
#        for i in range(4):
#            mask1 = mask[:,:,i]
#            print(mask1)
#            plt.imshow(mask1)
#            plt.show()
#    break
#    if i==1:
#        break
#    i=i+1
# define network parameters
n_classes = 4
#if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=True)
#model.load_weights('C:/Users/guptav/Desktop/new 22.11/logs/ep01-val_loss0.72.h5')
# define optomizer
optim = keras.optimizers.Adam(lr=0.0)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), accuracy5]

# compile keras model with defined optimozer, loss and metrics
#for layer in model.layers[:-42]:
#    layer.trainable = False


model.compile(optim, total_loss, metrics)


model.summary()

#callback = keras.callbacks.LearningRateScheduler(lr_schedule)
callbacks = [keras.callbacks.ModelCheckpoint('C:/Users/guptav/Desktop/Data1 15.12/logs/ep{epoch:02d}-val_loss{val_loss:.2f}.h5', verbose=1, save_weights_only=True, save_best_only=False, monitor='accuracy5',mode='auto' , period= 1 ),keras.callbacks.LearningRateScheduler(lr_schedule)]

#    keras.callbacks.ReduceLROnPlateau(),
#]
#callbacks = [keras.callbacks.ModelCheckpoint('C:/Users/guptav/Desktop/new 22.11/logs/ep{epoch:02d}-val_loss{val_loss:.2f}.h5', verbose=1, save_weights_only=True, save_best_only=False, monitor='accuracy5',mode='auto' , period= 1 ),ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.000001)]

#hist = model.fit_generator(generator=tg,
#                    steps_per_epoch=len(tg),epochs=25, callbacks=callbacks, validation_data=vg, 
#    validation_steps=len(vg), initial_epoch=0)

hist = model.fit_generator(generator=tg,
                    steps_per_epoch=len(tg),epochs=220, callbacks=callbacks, validation_data=vg, 
    validation_steps=len(vg), initial_epoch=0)

#json_string = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(json_string)

train_loss=hist.history['loss']
 

val_loss=hist.history['val_loss']
train_acc5=hist.history['accuracy5']
val_acc5=hist.history['val_accuracy5']
#train_acc6=hist.history['accuracy6']
#val_acc6=hist.history['val_accuracy6']
#train_acc7=hist.history['accuracy7']
#val_acc7=hist.history['val_accuracy7']


train_f1_score=hist.history['f1-score']
val_f1_score=hist.history['val_f1-score']

train_iou_score=hist.history['iou_score']
val_iou_score=hist.history['val_iou_score']
#print(train_loss.dtype)
#import csv 
data = train_loss + val_loss + train_acc5 + val_acc5 + train_f1_score + val_f1_score + train_iou_score + val_iou_score
# data to be written row-wise in csv fil 
# opening the csv file in 'a+' mode 
#print(data.dtype)
file = open('g4g.csv', 'w+', newline ='') 
  
# writing the data into the file 
with file:     
    out = csv.writer(file)
    out.writerows(map(lambda x: [x], data))
    #write.writerows(data)
file.close()

#for l in model.layers:    
#    print(l.name, l.trainable)

#print(model.layers[-43].name)

#model.trainable =True
#for layer in model.layers[:-42]:
#    layer.trainable = False
    



#for layer in model.layers:
    #if not isinstance(layer, keras.layers.BatchNormalization):
#    layer.trainable = True

#print(layers.BatchNormalization)
#print(model.layers[-4])
#for l in model.layers:
#    print(l.name, l.trainable)

#model.compile(optim, total_loss, metrics)
#model.summary()

#for l in model.layers:
#    print(l.name, l.trainable)
    
#hist = model.fit_generator(generator=tg,
#                    steps_per_epoch=len(tg),epochs=500, callbacks=callbacks, validation_data=vg, 
#    validation_steps=len(vg), initial_epoch=100)



#train_loss1=hist.history['loss']
#val_loss1=hist.history['val_loss']
#train_acc51=hist.history['accuracy5']
#val_acc51=hist.history['val_accuracy5']
#train_acc61=hist.history['accuracy6']
#val_acc61=hist.history['val_accuracy6']
#train_acc71=hist.history['accuracy7']
#val_acc71=hist.history['val_accuracy7']


#train_f1_score1=hist.history['f1-score']
#val_f1_score1=hist.history['val_f1-score']

#train_iou_score1=hist.history['iou_score']
#val_iou_score1=hist.history['val_iou_score']

#train_loss1=train_loss + train_loss1
#val_loss1=val_loss + val_loss1
#train_acc51=train_acc5 + train_acc51
#val_acc51=val_acc5 + val_acc51
#train_acc61=train_acc6 + train_acc61
#val_acc61=val_acc6 + val_acc61
#train_acc71=train_acc7 + train_acc71
#val_acc71=val_acc7 + val_acc71


#train_f1_score1=train_f1_score + train_f1_score1
#val_f1_score1=val_f1_score + val_f1_score1

#train_iou_score1=train_iou_score + train_iou_score1
#val_iou_score1=val_iou_score + val_iou_score1



# visualizing losses and accuracy
#train_loss=hist.history['loss']
#val_loss=hist.history['val_loss']
#train_acc=hist.history['accuracy']
#val_acc=hist.history['val_accuracy']
xc=range(0,220,1)
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

#plt.figure(3,figsize=(7,5))
#plt.plot(xc,train_acc61)
#plt.plot(xc,val_acc61)
#plt.xlabel('num of Epochs')
#plt.ylabel('accuracy6')
#plt.title('train_acc vs test_acc')
#plt.grid(True)
#plt.legend(['train','test'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])
#plt.savefig('Accuracy6_graph.png')

#plt.figure(4,figsize=(7,5))
#plt.plot(xc,train_acc71)
#plt.plot(xc,val_acc71)
#plt.xlabel('num of Epochs')
#plt.ylabel('accuracy7')
#plt.title('train_acc vs test_acc')
#plt.grid(True)
#plt.legend(['train','test'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])
#plt.savefig('Accuracy7_graph.png')

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

# Plot training & validation iou_score values
#plt.figure(figsize=(90, 20))
#plt.subplot(131)
#plt.figure(1,figsize=(7,5))
#plt.plot(history.history['iou_score'])
#plt.plot(history.history['val_iou_score'])
#plt.title('Model iou_score')
#plt.ylabel('iou_score')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.style.use(['classic'])
#plt.savefig('iou_score.png')

#plt.subplot(132)
#plt.figure(1,figsize=(7,5))
#plt.plot(history.history['f1-score'])
#plt.plot(history.history['val_f1-score'])
#plt.title('Model f1-score')
#plt.ylabel('f1-score')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.style.use(['classic'])
#plt.savefig('f1-score.png')

# Plot training & validation loss values
#plt.subplot(133)
#plt.figure(1,figsize=(7,5))
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.style.use(['classic'])
#plt.savefig('loss.png')
print(val_acc5[115:127])
print(val_loss[115:127])