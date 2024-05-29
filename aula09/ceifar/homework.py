import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model

def impHistoria(history):
  print(history.history.keys())
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

batch_size = 100
num_classes = 2
epochs = 30

nl, nc = 32,32
(ax, ay), (qx, qy) = cifar10.load_data()

input_shape = (nl, nc, 3)

ax = ax.astype('float32')
ax = ax/255
ax = ax - 0.5
qx = qx.astype('float32')
qx = qx.astype('float32')
qx = qx/255
qx = qx - 0.5

ay = (ay == 6).astype(int) # 6 é sapo (doente)
qy = (qy == 6).astype(int)

model = Sequential()
model.add(Conv2D(8, kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

plot_model(model, show_shapes=True)
model.summary()

opt=optimizers.Adam()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(ax, ay, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(qx, qy))
impHistoria(history)

score = model.evaluate(qx, qy, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from sklearn.metrics import confusion_matrix

# Predict probabilities
predicted_probabilities = model.predict(qx)

# Convert probabilities to binary class labels based on a threshold (usually 0.5 for binary classification)
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

# Generate the confusion matrix
tn, fp, fn, tp = confusion_matrix(qy, predicted_labels).ravel()

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(qy, predicted_probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

fpr, tpr, thresholds = roc_curve(qy, predicted_probabilities)

# Calculate the false negative rate (fnr)
fnr = 1 - tpr

# Find the point where the false positive rate is closest to the false negative rate
eer_index = np.nanargmin(np.abs(fpr - fnr))
eer_threshold = thresholds[eer_index]
EER = fpr[eer_index]

print("Equal Error Rate (EER):", EER)