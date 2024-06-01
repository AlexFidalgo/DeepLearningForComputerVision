#Rede inspirada em inception para classificar CIFAR-10
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from inspect import currentframe, getframeinfo
import numpy as np; import os
import matplotlib.pyplot as plt; import numpy as np

def impHistoria(history):
  print(history.history.keys())
  plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy'); plt.ylabel('accuracy'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()
  plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
  plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left'); plt.show()

nomeprog="inception2";
batch_size = 100; num_classes = 10; epochs = 300
nl, nc = 32,32; input_shape = (nl, nc, 3)
(ax, ay), (qx, qy) = cifar10.load_data()
ax = ax.astype('float32'); ax /= 255; ax -= 0.5 #-0.5 a +0.5
qx = qx.astype('float32'); qx /= 255; qx -= 0.5 #-0.5 a +0.5

ay = keras.utils.to_categorical(ay, num_classes)
qy = keras.utils.to_categorical(qy, num_classes)

def moduloInception(nfiltros, x):

  kweight=5e-4
  bweight=5e-4
  tower_0 = Conv2D(  nfiltros,  (1,1), padding='same', activation='relu',
                   kernel_regularizer=l2(kweight),bias_regularizer=l2(bweight))(x) #conv2d_1
  tower_1 = Conv2D(2*nfiltros,  (1,1), padding='same', activation='relu',
                   kernel_regularizer=l2(kweight),bias_regularizer=l2(bweight))(x) #conv2d_2 
  tower_1 = Conv2D(2*nfiltros,  (3,3), padding='same', activation='relu',
                   kernel_regularizer=l2(kweight),bias_regularizer=l2(bweight))(tower_1) #conv2d_3
  tower_2 = Conv2D(nfiltros//2, (1,1), padding='same', activation='relu',
                   kernel_regularizer=l2(kweight),bias_regularizer=l2(bweight))(x) #conv2d_4
  tower_2 = Conv2D(nfiltros//2, (5,5), padding='same', activation='relu',
                   kernel_regularizer=l2(kweight),bias_regularizer=l2(bweight))(tower_2)#conv2d_5
  tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x) #max_pooling2d_1
  tower_3 = Conv2D(nfiltros//2, (1,1), padding='same', activation='relu',
                   kernel_regularizer=l2(kweight),bias_regularizer=l2(bweight))(tower_3)#conv2d_6
  x = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
  # x=Dropout(0.3)(x)
  x = BatchNormalization()(x)
  return x

inputs = Input(shape=input_shape)
x = inputs
x = moduloInception(64,x); x = moduloInception(64,x)
x = MaxPooling2D(2)(x); #(64+128+32+32)x16x16
x = moduloInception(64,x); x = moduloInception(64,x)
x = MaxPooling2D(2)(x); #(64+128+32+32)x8x8
x = moduloInception(64,x); x = moduloInception(64,x) #(64+128+32+32)x8x8=16384   
output = AveragePooling2D(8)(x) #(64+128+32+32)x1x1=256x1x1
output = Flatten()(output)
outputs= Dense(10, activation='softmax')(output)

#Pode escolher entre construir modelo novo ou continuar o treino de onde parou
model = Model(inputs=inputs, outputs=outputs)
#model = load_model(nomeprog+'.h5')

#from tensorflow.keras.utils import plot_model
#plot_model(model, to_file=nomeprog+'.png', show_shapes=True); 
model.summary()

opt=Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
  rotation_range=15, fill_mode='nearest', horizontal_flip=True)
#datagen.fit(ax)

reduce_lr = ReduceLROnPlateau(monitor='accuracy',
  factor=0.9, patience=2, min_lr=0.0001, verbose=True)
history=model.fit(datagen.flow(ax, ay, batch_size=batch_size),
          epochs=epochs, verbose=2, validation_data=(qx, qy),
          steps_per_epoch=ax.shape[0]//batch_size,callbacks=[reduce_lr])
impHistoria(history)

score = model.evaluate(qx, qy, verbose=0)
print('Test loss: %.4f'%(score[0]))
print('Test accuracy: %.2f %%'%(100*score[1]))
print('Test error: %.2f %%'%(100*(1-score[1])))
model.save(nomeprog+'.h5')