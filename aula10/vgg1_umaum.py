import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

def impHistoria(history):
  print(history.history.keys())
  plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy'); plt.ylabel('accuracy'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()
  plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
  plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()

batch_size = 100; num_classes = 10; epochs = 30
nl, nc = 32,32; input_shape = (nl, nc, 3)
(ax, ay), (qx, qy) = cifar10.load_data()
ax = ax.astype('float32'); ax /= 255; ax = 2*(ax-0.5) #-1 a +1
qx = qx.astype('float32'); qx /= 255; qx = 2*(qx-0.5) #-1 a +1
ay = keras.utils.to_categorical(ay, num_classes)
qy = keras.utils.to_categorical(qy, num_classes)

model = Sequential() #32x32x3
model.add(Conv2D(20, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2))) #16x16x20
model.add(Conv2D(40, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2))) #8x8x40
model.add(Conv2D(80, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2))) #4x4x80
model.add(Conv2D(160, kernel_size=(3,3), activation='relu', padding='same')) #160x4x4
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='vgg1.png', show_shapes=True)
model.summary()
opt=optimizers.Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(ax, ay, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(qx, qy))
impHistoria(history)

score = model.evaluate(qx, qy, verbose=0)
print('Test loss:', score[0]); print('Test accuracy:', score[1])
model.save('vgg1_umaum.h5')