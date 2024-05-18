import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
import numpy as np; import sys; import cv2
from matplotlib import pyplot as plt

url='http://www.lps.usp.br/hae/apostila/cnn1.h5'
import os; nomeArq=os.path.split(url)[1]
if not os.path.exists(nomeArq):
  print("Baixando o arquivo",nomeArq,"para diretorio default",os.getcwd())
  os.system("wget -U 'Firefox/50.0' "+url)
else:
  print("O arquivo",nomeArq,"ja existe no diretorio default",os.getcwd())

model=keras.models.load_model("cnn1.h5")

(filters, biases) = model.get_layer(index=0).get_weights()
filters=np.squeeze( filters )
print(filters.shape)
filters2=np.empty( (filters.shape[2], filters.shape[0], filters.shape[1]) )
print(filters2.shape)
for i in range(filters.shape[2]):
  filters2[i,:,:]=filters[:,:,i]

f = plt.figure()
for i in range(20):
  f.add_subplot(4,5,i+1)
  plt.imshow(filters2[i],vmin=-0.25, vmax=0.25, cmap="gray")
  plt.axis('off')
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.15, 0.06, 0.7])
plt.colorbar(cax=cax)
plt.savefig("filtros0.png")
plt.show()