"""
~/deep/algpi/transf/efficientnet_transf_cifar10.py
Original file: https://colab.research.google.com/drive/1bD_ckH-KiPL_lgheQ7SDQOv6x4QO_EM0
Baseado em: https://www.kaggle.com/code/nikhilpandey360/transfer-learning-using-xception
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np; import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import tensorflow as tf; print(tf.__version__)
from keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Activation,\
  Dropout, GlobalAveragePooling2D, MaxPooling2D, RandomFlip, RandomZoom, RandomRotation
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train=to_categorical(y_train); y_test=to_categorical(y_test)
print((x_train.shape, y_train.shape)); print((x_test.shape, y_test.shape))

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=y_train.shape[1])
base_model.trainable = False
#base_model.summary()

data_augmentation = Sequential(
    [RandomFlip("horizontal"), RandomRotation(0.1), RandomZoom(0.1)]
)
"""Consider the image resolution that the imagenet was trained on.
The original image resolution of CIFAR-10 is 32x32, which is too low for EfficientNetB0 (min. 71x71)
O tamanho padrao de entrada desta rede e' 224x224
"""
inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224,224)))(inputs) 
# x = tf.keras.layers.Resizing(224,224)(inputs)
x = data_augmentation(x)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation=('softmax'))(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 20
history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, verbose=1)

def plot_history(history):
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    return
plot_history(history)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy", test_acc); print("Test loss", test_loss)

"""# Fine Tuning"""
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10
history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy", test_acc); print("Test loss", test_loss)
plot_history(history)

"""# Confusion Matrix"""
class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predictions=model.predict(x_test)
y_pred_classes = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 9))
c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
c.set(xticklabels=class_names, yticklabels=class_names)