#prelayer2.py
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation
import matplotlib.pyplot as plt
import cv2; import sys

X = []
a=cv2.imread('lenna.jpg',1);    a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); X.append(a)
a=cv2.imread('mandrill.jpg',1); a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); X.append(a)
a=cv2.imread('face.jpg',1);     a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); X.append(a)
X=np.array(X).astype("float32")

layer = Sequential(
    [
      RandomRotation(15/360,fill_mode="nearest",interpolation="bilinear"), 
      RandomTranslation(0.1, 0.1,fill_mode="nearest",interpolation="bilinear"),
      RandomFlip("horizontal")
    ]
)

for l in range(3):
  transformedX=layer(X).numpy()
  for c in range(3):
    plt.subplot(3, 3, 3*l+c+1)
    image = transformedX[c].astype("uint8") 
    plt.imshow(image)
    plt.axis("off")
plt.show()
