from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras as keras
from keras.datasets import mnist
import cv2
import numpy as np
import time

class MNISTData:
    def __init__(self, AX, ay, QX, qy):
        self.ax = AX.reshape(AX.shape[0], -1).astype(np.float32) / 255
        self.ay = ay
        self.qx = QX.reshape(QX.shape[0], -1).astype(np.float32) / 255
        self.qy = qy
        self.nq = len(qy)
        self.qp = np.zeros(self.nq, dtype=np.float32)

    # Function to count errors
    def count_errors(self):
        return np.count_nonzero(self.qp != self.qy)

(AX, ay), (QX, qy) = mnist.load_data()

# Create an instance of MNISTData
mnist = MNISTData(AX, ay, QX, qy)

# Train K Nearest Neighbors
t1 = time.time()
knn = KNeighborsClassifier(n_neighbors=1, algorithm='auto')  # Using 1 nearest neighbor
knn.fit(mnist.ax, mnist.ay)
t2 = time.time()

# Predict using trained K Nearest Neighbors
mnist.qp = knn.predict(mnist.qx)
t3 = time.time()

# Print results
errors = mnist.count_errors()
print("Errors: %5.2f%%" % (100.0 * errors / mnist.nq))
print("Training time: %f" % (t2 - t1))
print("Prediction time: %f" % (t3 - t2))
