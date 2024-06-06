import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
import cv2
import numpy as np
import tensorflow.keras as keras
import keras.backend as K
from tensorflow.keras import optimizers, callbacks, regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from inspect import currentframe, getframeinfo

def leCsv(nomeDir, nomeArq, nl=0, nc=0, menosmais=False):
  #nomeDir = Diretorio onde estao treino.csv, teste.csv e imagens nnna.jpg e nnnb.jpg.
  #Ex: nomeDir = "/home/hae/haebase/fei/feiFrontCor"
  #Imagens sao redimensionadas para nlXnc (se diferentes de zero).
  st=os.path.join(nomeDir,nomeArq)
  arq=open(st,"rt")
  lines=arq.readlines()
  arq.close()
  n=len(lines)

  linhas_separadas=[]
  for linha in lines:
    linha=linha.strip('\n')
    linha=linha.split(';')
    linhas_separadas.append(linha)
  ay=np.empty((n),dtype='float32')
  ax=np.empty((n,nl,nc,3),dtype='float32')
  for i in range(len(linhas_separadas)):
    linha=linhas_separadas[i]
    t=cv2.imread(os.path.join(nomeDir,linha[0]),1)
    if nl>0 and nc>0:
      t=cv2.resize(t,(nc,nl),interpolation=cv2.INTER_AREA)
    ax[i]=np.float32(t)/255.0; #Entre 0 e 1
    if menosmais:
      ax[i]=ax[i]-0.5 #-0.5 a +0.5
    ay[i]=np.float32(linha[1]); #0=m ou 1=f
  return ax, ay

#<<<<<<<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#Original: 280x200, redimensionado: 112x80
nl=112; nc=80
diretorioBd="."
ax, ay = leCsv(diretorioBd,"treino.csv", nl=nl, nc=nc, menosmais=True) #200 imagens
qx, qy = leCsv(diretorioBd,"teste.csv",  nl=nl, nc=nc, menosmais=True)  #100 imagens
vx, vy = leCsv(diretorioBd,"valida.csv", nl=nl, nc=nc, menosmais=True)  #100 imagens
input_shape = (nl,nc,3)
batch_size = 10
epochs = 50

model = Sequential()
model.add(Conv2D(30, kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2))) #56x40 
model.add(Conv2D(40, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #28x20
model.add(Conv2D(50, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #14x10
model.add(Conv2D(60, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #7x5
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='linear'))

#from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='ep2g.png', show_shapes=True)
#model.summary()

opt=optimizers.Adam();
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
model.fit(ax, ay, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(vx,vy))

score = model.evaluate(ax, ay, verbose=0)
print('Training loss:', score)
score = model.evaluate(vx, vy, verbose=0)
print('Validation loss:', score)
score = model.evaluate(qx, qy, verbose=0)
print('Test loss:', score)