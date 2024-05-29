import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import math

def impHistoria(history):
  print(history.history.keys())
  plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy'); plt.ylabel('accuracy'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()
  plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
  plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()

batch_size = 100; num_classes = 10; epochs = 200
nl, nc = 32,32; input_shape = (nl, nc, 3)
(ax, ay), (qx, qy) = cifar10.load_data()
ax = ax.astype('float32'); ax /= 255 #0 a 1
qx = qx.astype('float32'); qx /= 255 #0 a 1
ay = tf.keras.utils.to_categorical(ay, num_classes)
qy = tf.keras.utils.to_categorical(qy, num_classes)

def create_model():
  model = Sequential(
    [
      RandomRotation(0.042,fill_mode="nearest",interpolation="bilinear"), #15 graus: 15*pi/180
      RandomTranslation(0.1, 0.1,fill_mode="nearest",interpolation="bilinear"),
      RandomFlip("horizontal"),

      Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_shape),
      BatchNormalization(), Dropout(0.3),
      Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
      BatchNormalization(),
      MaxPooling2D(pool_size=(2,2)), #20x16x16x3

      Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
      BatchNormalization(), Dropout(0.3),
      Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
      BatchNormalization(),
      MaxPooling2D(pool_size=(2,2)), #40x8x8x3

      Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
      BatchNormalization(), Dropout(0.3),
      Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
      BatchNormalization(),
      MaxPooling2D(pool_size=(2,2)), #80x4x4x3

      Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'), #160x4x4x3
      BatchNormalization(), Dropout(0.3),
      Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'), #160x4x4x3
      BatchNormalization(),
      Dropout(0.3),
      Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'), #160x4x4x3
      BatchNormalization(),
      MaxPooling2D(pool_size=(2,2)), #160x2x2x3

      Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'), #160x2x2x3
      BatchNormalization(), Dropout(0.3),
      Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'), #160x2x2x3
      BatchNormalization(), Dropout(0.3),
      Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'), #160x2x2x3
      BatchNormalization(),
      MaxPooling2D(pool_size=(2,2)), #160x1x1x3

      Flatten(),
      Dense(512,activation='relu'),
      BatchNormalization(), Dropout(0.3),

      Dense(num_classes,activation='softmax')
    ]
  )

  opt=optimizers.Adam()
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

model=create_model()
#from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='vgg_prelayer.png', show_shapes=True); model.summary()

history=model.fit(ax, ay, batch_size=batch_size, epochs=epochs, verbose=2, 
                  validation_data=(qx, qy))
impHistoria(history)

score = model.evaluate(qx, qy, verbose=0)
print('Test loss:', score[0]); print('Test accuracy:', score[1])
#model.save('vgg4_prelayer.h5')