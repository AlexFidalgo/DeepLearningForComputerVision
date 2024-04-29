import tensorflow.keras as keras
from keras.datasets import mnist
import cv2
import numpy as np
import time

# Load MNIST dataset
(AX, ay), (QX, qy) = mnist.load_data()

# Define function to shift images to the left
def shift_left(image):
    shifted = np.roll(image, -1, axis=1)
    shifted[:, -1] = 0  # Fill the empty space with zeros
    return shifted

# Define function to shift images to the right
def shift_right(image):
    shifted = np.roll(image, 1, axis=1)
    shifted[:, 0] = 0  # Fill the empty space with zeros
    return shifted

# Define function to shift images upwards
def shift_up(image):
    shifted = np.roll(image, -1, axis=0)
    shifted[-1, :] = 0  # Fill the empty space with zeros
    return shifted

# Define function to shift images downwards
def shift_down(image):
    shifted = np.roll(image, 1, axis=0)
    shifted[0, :] = 0  # Fill the empty space with zeros
    return shifted

# Apply data augmentation to create augmented datasets
augmented_AX_left = np.array([shift_left(img) for img in AX])
augmented_AX_right = np.array([shift_right(img) for img in AX])
augmented_AX_up = np.array([shift_up(img) for img in AX])
augmented_AX_down = np.array([shift_down(img) for img in AX])

# Concatenate original dataset with augmented datasets
augmented_AX = np.concatenate((AX, augmented_AX_left, augmented_AX_right, augmented_AX_up, augmented_AX_down))
augmented_ay = np.concatenate((ay, ay, ay, ay, ay))  # Corresponding labels remain the same

# Reshape augmented data
augmented_ax = augmented_AX.reshape(augmented_AX.shape[0], augmented_AX.shape[1] * augmented_AX.shape[2]).astype("float32") / 255

# Resize query images to match training data dimensions
qx_resized = np.empty((QX.shape[0], 28, 28))
for i in range(QX.shape[0]):
    qx_resized[i] = cv2.resize(QX[i], (28, 28), cv2.INTER_NEAREST)
qx_resized = qx_resized.reshape(qx_resized.shape[0], qx_resized.shape[1] * qx_resized.shape[2]).astype("float32") / 255

# Perform k-nearest neighbors search using FLANN
t1 = time.time()
FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
flann = cv2.flann_Index(augmented_ax, flann_params)
t2 = time.time()
matches, dists = flann.knnSearch(qx_resized, 1)
t3 = time.time()

# Extract predicted labels
qp = augmented_ay[matches].flatten()

# Calculate errors
errors = np.count_nonzero(qp != qy)

# Print results
print("Errors=%5.2f%%" % (100.0 * errors / qy.shape[0]))
print("Training time: %f" % (t2 - t1))
print("Prediction time: %f" % (t3 - t2))
