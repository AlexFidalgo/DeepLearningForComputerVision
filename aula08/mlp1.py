import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Normalization
from tensorflow.keras import optimizers
import numpy as np; import sys

(AX, AY), (QX, QY) = mnist.load_data()
AX=255-AX; QX=255-QX

nclasses = 10
AY2 = keras.utils.to_categorical(AY, nclasses)
QY2 = keras.utils.to_categorical(QY, nclasses)

nl, nc = AX.shape[1], AX.shape[2] #28, 28

#Pseudo-normalizacao1 para intervalo [-0.5, +0.5]
#AX = (AX.astype('float32')/255.0)-0.5 # -0.5 a +0.5
#QX = (QX.astype('float32')/255.0)-0.5 # -0.5 a +0.5

#Normalizacao2 - distribuicao normal de media zero e desvio 1
#media=np.mean(AX); desvio=np.std(AX) 
#AX=AX.astype('float32'); AX=AX-media; AX=AX/desvio; QX=QX.astype('float32'); QX=QX-media; QX=QX/desvio

#Normalizacao3 - inserir camada de normalizacao na rede
model = Sequential()
model.add(Normalization(input_shape=(nl,nc))) #Normaliza
model.add(Flatten())
model.add(Dense(400, activation='sigmoid'))
model.add(Dense(nclasses, activation='sigmoid'))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='mlp1.png', show_shapes=True)
model.summary()

opt=optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

model.get_layer(index=0).adapt(AX) #Calcula media e desvio
model.fit(AX, AY2, batch_size=100, epochs=40, verbose=2)
score = model.evaluate(QX, QY2, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('mlp1.keras')