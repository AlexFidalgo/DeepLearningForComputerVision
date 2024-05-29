import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2; import sys

data = cv2.imread("lenna.jpg",1); data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
samples = np.expand_dims(data, 0).astype("float32")
datagen = ImageDataGenerator(brightness_range=(0.5,1.5))
it = datagen.flow(samples, batch_size=1, seed=7)

for i in range(5):
  plt.subplot(1, 5, 1+i)
  batch = it.next()
  image = batch[0].astype("uint8") 
  plt.imshow(image)
  plt.axis("off")
# plt.savefig("brightness.png")
plt.show()