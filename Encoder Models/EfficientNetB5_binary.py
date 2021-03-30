
import numpy as np
import os
import time
#from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras import layers
import tensorflow as tf
#from imagenet_utils import preprocess_input

from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def lr_schedule(epoch,lr):
  if epoch < 25:   
      return 0.01
  elif epoch < 75:
      return 0.00001
  #else:
  #  return lr * tf.math.exp(-0.1)
 #if epoch < 80:
   # elif epoch < 100:
   #     return 0.0001
   # else:
   #     return 0.00001


# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(456, 456))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:575]=0
labels[576:1151]=1
#labels[404:606]=2
#labels[606:]=3

names = ['defect','no defect']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

training_datagen = ImageDataGenerator(rotation_range=10,brightness_range=(1.1, 0.9),shear_range=10.0,zoom_range=[0.8, 1.2],vertical_flip=True,fill_mode='wrap',
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator()

t_gen = training_datagen.flow(X_train, y_train, batch_size=4)
v_gen = valid_datagen.flow(X_test, y_test, batch_size=4)
#t_gen = training_datagen.flow(X_train, y_train)
#v_gen = valid_datagen.flow(X_test, y_test)

image_input = tf.keras.Input(shape=(456, 456, 3))
#model = ResNet50(weights='imagenet')
model = EfficientNetB5(input_tensor=image_input, include_top=False,weights="imagenet")
model.summary()
model.trainable = False
last_layer = model.output
last_layer = tf.keras.layers.GlobalAveragePooling2D(name= 'avg_pool_last')(last_layer)
#x= tf.keras.layers.Flatten(name='flatten')(last_layer)
last_layer = tf.keras.layers.BatchNormalization(name = 'batch_last')(last_layer)
top_dropout_rate = 0.2
last_layer = tf.keras.layers.Dropout(top_dropout_rate, name = 'top_dropout')(last_layer)
out = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')(last_layer)
#image_input = Input(shape=(500, 500, 3))
custom_resnet_model = tf.keras.Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

#for layer in custom_resnet_model.layers[:-4]:
#	layer.trainable = False





#custom_resnet_model.layers[-6].trainable
#custom_resnet_model.trainable=True

adam = tf.keras.optimizers.Adam()

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer=adam ,metrics=['accuracy'])

t=time.time()
callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
#hist = custom_resnet_model.fit_generator(generator=t_gen, steps_per_epoch=1036//4, epochs=120, verbose = 1, validation_data= v_gen, validation_steps=116//4, shuffle = True, initial_epoch=0 )
hist = custom_resnet_model.fit(t_gen,steps_per_epoch=1036//4, epochs=25, callbacks=[callback],verbose = 1, validation_data= v_gen, validation_steps=116//4, shuffle = True, initial_epoch=0 )

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']

for layer in custom_resnet_model.layers:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True


custom_resnet_model.compile(loss='categorical_crossentropy',optimizer=adam ,metrics=['accuracy'])
hist = custom_resnet_model.fit(t_gen,steps_per_epoch=1036//4, epochs=75, callbacks=[callback],verbose = 1, validation_data= v_gen, validation_steps=116//4, shuffle = True, initial_epoch=25 )

train_loss1=hist.history['loss']
val_loss1=hist.history['val_loss']
train_acc1=hist.history['accuracy']
val_acc1=hist.history['val_accuracy']

train_loss.append(train_loss1)
val_loss.append(val_loss1)
train_acc.append(train_acc1)
val_acc.append(val_acc1)

print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=4, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


data = train_loss + val_loss + train_acc + val_acc 
# opening the csv file in 'a+' mode 
#print(data.dtype)
file = open('data.csv', 'w+', newline ='') 
  
# writing the data into the file 
with file:     
    out = csv.writer(file)
    out.writerows(map(lambda x: [x], data))
    #write.writerows(data)
file.close()

xc=range(0,75,1)
print(train_loss)
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
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs test_acc')
plt.grid(True)
plt.legend(['train','test'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Accuracy_graph.png')