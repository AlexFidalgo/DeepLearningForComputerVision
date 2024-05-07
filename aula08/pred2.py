import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

(_,_), (QX, QY) = mnist.load_data()
QX=255-QX

nclasses = 10
QY2 = keras.utils.to_categorical(QY, nclasses)
nl, nc = QX.shape[1], QX.shape[2] #28, 28

model=load_model('mlp2.keras')
score = model.evaluate(QX, QY2, verbose=False)
print('Test loss: %.4f'%(score[0]))
print('Test accuracy: %.2f %%'%(100*score[1]))
print('Test error: %.2f %%'%(100*(1-score[1])))

QP2 = model.predict(QX)
QP = QP2.argmax(axis=-1)
print(QP2[0])
print(QP[0])
#Aqui QP contem as 10000 predicoes
print("Imagem-teste 0: Rotulo verdadeiro=%d, predicao=%d"%(QY[0],QP[0]))
print("Imagem-teste 1: Rotulo verdadeiro=%d, predicao=%d"%(QY[1],QP[1]))

