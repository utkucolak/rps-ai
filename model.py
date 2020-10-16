import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import os
from matplotlib.image import imread
import matplotlib.pyplot as plt

data_path = 'rps'

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(300,200,3), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

image_generator = ImageDataGenerator(validation_split=0.2, rotation_range=0.2, zoom_range=0.1, shear_range=0.1)
train_data_gen = image_generator.flow_from_directory(directory='rps',
                                                     subset='training',
                                                     target_size=(300,200))
val_data_gen = image_generator.flow_from_directory(directory='rps',
                                                   subset='validation',
                                                   target_size=(300,200))
r = model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs=30)

plt.plot(r.history['acc'], label='accuracy')
plt.plot(r.history['val_acc'], label='val_accuracy')
plt.legend()
plt.savefig('graph.png')