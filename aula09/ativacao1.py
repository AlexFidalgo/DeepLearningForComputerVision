#ativacao1.py
url='http://www.lps.usp.br/hae/apostila/cnn1.h5'
import os; nomeArq=os.path.split(url)[1]
if not os.path.exists(nomeArq):
  print("Baixando o arquivo",nomeArq,"para diretorio default",os.getcwd())
  os.system("wget -U 'Firefox/50.0' "+url)
else:
  print("O arquivo",nomeArq,"ja existe no diretorio default",os.getcwd())
  
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras.models import Model
import numpy as np; import cv2
from matplotlib import pyplot as plt;
import matplotlib.patches as patches
import sys

(_,_), (qx, QY) = mnist.load_data()
qx=255-qx

nl, nc = qx.shape[1], qx.shape[2] #28, 28
QX = qx.astype('float32') / 255.0 # 0 a 1

model=models.load_model("cnn1.h5")

lista=[]; digito=[]
ndig=2; #quantidade procurado
j=0 #indice de QY
for dig in (1,3):
  for i in range(ndig):
    while QY[j]!=dig and j<QY.shape[0]:
      j+=1
    if j>=QY.shape[0]:
      sys.exit("Erro inesperado")
    lista.append(j); digito.append(dig)
    j+=1

intermediate_layer_model1 = Model(inputs=model.input,
                                  outputs=model.get_layer(index=0).output)
for j,dig in zip(lista,digito):
  print("Imagem digito=%d, indice=%d"%(dig,j))

  st="di_%1d_%03d_dig.png"%(dig,j); cv2.imwrite(st,qx[j])
  plt.imshow(qx[j],cmap="gray"); plt.axis("off"); plt.show()

  x=QX[j].copy(); x=np.expand_dims(x,axis=0); x=np.expand_dims(x,axis=-1);
  y = intermediate_layer_model1.predict(x)
  y = np.squeeze(y,0)
  y2=np.empty( (y.shape[2], y.shape[0], y.shape[1]) )
  for i in range(y2.shape[0]):
    y2[i,:,:]=y[:,:,i]

  fig, axes = plt.subplots(nrows=4, ncols=5)
  i=0
  for ax in axes.flat:
    im=ax.imshow(y2[i], vmin=0, vmax=1, cmap="gray"); i+=1
    ax.axis('off')
    rect = patches.Rectangle((50,100),40,30,linewidth=10,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
  fig.subplots_adjust(right=0.8)

  cbar_ax=fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(im,cax=cbar_ax)

  st="a0_%1d_%03d_dig.png"%(dig,j)
  plt.savefig(st)
  plt.show()
