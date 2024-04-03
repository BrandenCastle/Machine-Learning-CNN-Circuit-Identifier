import imghdr
import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import keras
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img
import random 
from shutil import copyfile 
import seaborn as sbn
import pathlib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#data augmentation. takes the image file, number of images desired, directory destination, and file type
def aug(picture,amt,direct,prefix):
    makeimg = ImageDataGenerator(
        rotation_range = 45,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = False,
        fill_mode = 'nearest')
    # image is loaded in and reshaped
    image = load_img(picture)
    x = img_to_array(image) 
    x = x.reshape((1,)+x.shape)

    #multiple ranadomly generated augmentations of the image are made here
    i = 0
    for batch in makeimg.flow(x, batch_size=1,
                              save_to_dir =direct,save_prefix =prefix,save_format = 'jpeg'):
        i+=1
        if i>=amt:
            break;

#splits data into training and validation sets
def splitter(source, train, val,spsize):
    files = []
    for filename in os.listdir(source):#creates a list of all files from a given source
        file = source + '/'+filename
        if os.path.getsize(file) >0:
            files.append(filename)
        else:
            print("no info")
    
    train_size = int(len(files)*spsize)#takes a percentage of the files and puts them into one category
    val_size = int(len(files)-train_size)#takes the remainder of the images for validation
    pick_rand = random.sample(files,len(files))
    train_set = pick_rand[0:train_size]
    val_set = shuffle_file[train_size:]#randomly chooses files from each created set
    
    #creates copies of the chosen files in the given directories (training and validation) 
    for filename in train_set:
        this_file = source +'\\'+ filename
        direct = train +'\\'+ filename
        copyfile(this_file,direct)
        
    for filename in val_set:
        this_file = source +'/'+ filename
        direct = val +'/'+ filename
        copyfile(this_file,direct)

#processes images in training and validation directories by rescaling them to the desired format. variables can be changed to meet specific needs
train_dir = "desired_location_for_training_set"
train_data = ImageDataGenerator(rescale = 1/255.0,
                                rotation_range = 45,
                                zoom_range = 0.2,
                                horizontal_flip = True)
train_gen = train_data.flow_from_directory(train_dir,
                                           batch_size = 16,
                                           class_mode = 'categorical',

val_dir = "desired_location_for_validation_set"

val_datagen = ImageDataGenerator(rescale = 1/255.0)

val_gen = val_datagen.flow_from_directory(val_dir,
                                         batch_size = 16,
                                         class_mode = 'categorical',
                                         target_size = (256,256))    

"""
#early stopping is optional for if you would like the model to stop training when validation loss is not changing properly
es = EarlyStopping(monitor = 'val_loss',patience = 5, verbose = 1, restore_best_weights=True)
#save best model
best_file = "file_directory"
best_model = ModelCheckpoint(best_file,monitor = 'val_acc',verbose = 1, save_best_only = True)

"""

"""
the model is created here and has 5 convolutional layers. This can be changed based on your model. Some models may need more or less layers depending on need
this model also uses softmax activation at the final layer since we are considering multiple classes
"""
act = 'relu'
model = Sequential()

model.add(Conv2D(16,(3,3),activation = act,padding = 'same', input_shape = (256,256,3)))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),padding = 'same'))
    
model.add(Conv2D(32,(3,3),padding = 'same',activation = act))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),padding = 'same'))
    
model.add(Conv2D(64,(3,3),activation = act,padding = 'same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),padding = 'same'))
    
model.add(Conv2D(128,(3,3),padding = 'same',activation = act))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),padding = 'same'))
    
model.add(Conv2D(256,(3,3),activation = act,padding = 'same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),padding = 'same'))
    
model.add(Flatten()),
model.add(Dense(512,activation = act))
model.add(Dense(512,activation = act))
    
model.add(Dense(6,activation = 'softmax'))    

model.summary()

#model is compiled and trained here. You can use earlystopping instead of callbacks if you want to just focus on reducing loss before saving.
model.compile(optimizer = 'Adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

callback = tf.keras.callbacks.ModelCheckpoint(filepath = 'C:/school/EEGR 490/CircuitIdentifier/log//check.ckpt',
                                                         save_weights_only = True,
                                                         verbose = 1)

history = model.fit(train_gen,
                             epochs = 20,
                             verbose = 1,
                             validation_data = val_gen,
                             callbacks = [callback]
                             )

#plots accuracy and loss
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""
#once you are satisfied with your model, use this command to save it
model.save("file_directory/My_Model.h5")
"""
