import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2; import sys

url='http://www.lps.usp.br/hae/apostila/cifar_pretreinado.zip'
nomeArq=os.path.split(url)[1]

if not os.path.exists(nomeArq):
  print("Baixando o arquivo",nomeArq,"para diretorio default",os.getcwd())
  os.system("wget -U 'Firefox/50.0' "+url)
else:
  print("O arquivo",nomeArq,"ja existe no diretorio default",os.getcwd())
print("Descompactando arquivos novos de",nomeArq)  
os.system("unzip -u "+nomeArq)

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

X = []
a=cv2.imread('lenna.jpg',1);    a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); X.append(a)
a=cv2.imread('mandrill.jpg',1); a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); X.append(a)
a=cv2.imread('face.jpg',1);     a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); X.append(a)
X=np.array(X).astype("float32")

datagen=ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
it=datagen.flow(X, batch_size=3, seed=7)
for l in range(3):
  batch = it.next()
  for c in range(3):
    plt.subplot(3, 3, 3*l+c+1)
    image = batch[c].astype("uint8") 
    plt.imshow(image)
    plt.axis("off")
plt.show()
