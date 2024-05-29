import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2; import sys

datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2)
it = datagen.flow_from_directory(".", classes=["lenna","mandrill","face"], batch_size=3, seed=7)
for l in range(3):
  batch = it.next()
  for c in range(3):
    plt.subplot(3, 3, 3*l+c+1)
    image = batch[0][c].astype("uint8") 
    plt.text(0,-3,str(batch[1][c]),color="b")
    plt.imshow(image)
    plt.axis("off")
# plt.savefig("from_directory.png")
plt.show()
