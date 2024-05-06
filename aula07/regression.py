#regression.py - 2024
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
import numpy as np

#Define modelo de rede
model = Sequential()
model.add(Dense(2, activation='sigmoid', input_dim=2))
model.add(Dense(2, activation='linear'))

sgd=optimizers.SGD(learning_rate=1)
model.compile(optimizer=sgd,loss='mse')

AX = np.matrix('0.9 0.1; 0.1 0.9',dtype='float32')
AY = np.matrix('0.1 0.9; 0.9 0.1',dtype='float32')
print("AX"); print(AX)
print("AY"); print(AY)

# As opcoes sao usar batch_size=2 ou 1
model.fit(AX, AY, epochs=100, batch_size=1, verbose=2)

QX = np.matrix('0.9 0.1; 0.1 0.9; 0.8 0.0; 0.2 0.9',dtype='float32')
print(QX)
QP=model.predict(QX, verbose=2)
print(QP)