import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation
import matplotlib.pyplot as plt; import numpy as np

(ax, ay), (qx, qy) = cifar10.load_data()
nl, nc = ax.shape[1], ax.shape[2] #32x32

ax = ax.astype('float32'); ax /= 255; #0 a 1
qx = qx.astype('float32'); qx /= 255; #0 a 1

prelayer = Sequential(
  [ RandomRotation(60/360,fill_mode="nearest",interpolation="bilinear"), 
    #RandomTranslation(0.1, 0.1,fill_mode="nearest",interpolation="bilinear"),
    #RandomFlip("horizontal"),
    #Outras transformacoes
  ]
)

fig = plt.figure()
fig.set_size_inches(10, 8)
nc=5; nl=5
X=ax[0:nc,:,:]
for c in range(nc):
  a = fig.add_subplot(nl, nc, c+1)
  image=X[c]; a.imshow(image); a.axis("off")
for l in range(1,nl):
  transformedX=prelayer(X).numpy()
  for c in range(nc):
    a = fig.add_subplot(nl, nc, nc*l+c+1)
    image = transformedX[c]; a.imshow(image); a.axis("off")
plt.show()