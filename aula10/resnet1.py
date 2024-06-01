#Rede inspirada em resnet para classificar CIFAR-10
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from inspect import currentframe, getframeinfo
import numpy as np; import os; import sys
import matplotlib.pyplot as plt; import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

def impHistoria(history):
  print(history.history.keys())
  plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy'); plt.ylabel('accuracy'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()
  plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
  plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()

def resnet_layer(inputs, num_filters=16, kernel_size=3,
                 strides=1, activation='relu', batch_normalization=True):
  x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
             padding='same', kernel_initializer='he_normal',
             kernel_regularizer=l2(1e-4))(inputs)
  if batch_normalization: x = BatchNormalization()(x)
  if activation is not None: x = Activation(activation)(x)
  return x

def lr_schedule(epoch):
  lr = 1e-3
  if epoch > 180:   lr *= 0.5e-3
  elif epoch > 160: lr *= 1e-3
  elif epoch > 120: lr *= 1e-2
  elif epoch > 80:  lr *= 1e-1
  print('Learning rate: ', lr)
  return lr

nomeprog="resnet1"
batch_size = 32; num_classes = 10; epochs = 200
nl, nc = 32,32
(ax, ay), (qx, qy) = cifar10.load_data()
input_shape = (nl, nc, 3)
ax = ax.astype('float32'); ax /= 255 #0 a 1
qx = qx.astype('float32'); qx /= 255 #0 a 1
ax -= 0.5; qx -= 0.5 #-0.5 a +0.5
ay = keras.utils.to_categorical(ay, num_classes); 
qy = keras.utils.to_categorical(qy, num_classes)

inputs = Input(shape=input_shape)
x = resnet_layer(inputs=inputs)

num_filters = 16
y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

num_filters *= 2
y = resnet_layer(inputs=x, num_filters=num_filters, strides=2)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1,
                 strides=2, activation=None, batch_normalization=False)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

num_filters *= 2
y = resnet_layer(inputs=x, num_filters=num_filters, strides=2)

y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1,
                 strides=2, activation=None, batch_normalization=False)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = keras.layers.add([x, y]); x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)
y = Flatten()(x)
outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

model = Model(inputs=inputs, outputs=outputs); model.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file=nomeprog+'.png', show_shapes=True)

opt=Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [lr_reducer, lr_scheduler]

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
  fill_mode='nearest', horizontal_flip=True)

datagen.fit(ax)
history=model.fit(datagen.flow(ax, ay, batch_size=batch_size),
          epochs=epochs, verbose=2, workers=4, validation_data=(qx, qy),
          callbacks=callbacks)
impHistoria(history)

score = model.evaluate(qx, qy, verbose=0)
print('Test loss: %.4f'%(score[0]))
print('Test accuracy: %.2f %%'%(100*score[1]))
print('Test error: %.2f %%'%(100*(1-score[1])))
model.save(nomeprog+'.h5')