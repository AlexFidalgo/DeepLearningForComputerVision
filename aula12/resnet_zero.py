import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np; 
import tensorflow.keras as keras; import keras.backend as K;
from tensorflow.keras import optimizers, callbacks, regularizers;
from tensorflow.keras.regularizers import l2;
from tensorflow.keras.models import Sequential, Model;
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten;
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def leCsv(nomeDir, nomeArq, nl=0, nc=0):
  st=os.path.join(nomeDir,nomeArq); 
  arq=open(st,"rt"); lines=arq.readlines(); arq.close(); n=len(lines)

  linhas_separadas=[]
  for linha in lines:
    linha=linha.strip('\n'); linha=linha.split(';'); linhas_separadas.append(linha);

  ay=np.empty((n),dtype='float32'); ax=np.empty((n,nl,nc,3),dtype='float32');
  for i in range(len(linhas_separadas)):
    linha=linhas_separadas[i];
    img_path=os.path.join(nomeDir,linha[0])
    t = image.load_img(img_path, target_size=(nl,nc))
    x = image.img_to_array(t)
    x = np.expand_dims(x, axis=0)
    ax[i] = preprocess_input(x)
    ay[i] = np.float32(linha[1]); #0=m ou 1=f
  return ax, ay;

#<<<<<<<<<<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
nomeprog="resnet_zero0"
#Original: 280x200, redimensionado: 224x224
num_classes=2; nl=224; nc=224
diretorioBd="."
ax, ay = leCsv(diretorioBd,"treino.csv", nl=nl, nc=nc); #200 imagens
qx, qy = leCsv(diretorioBd,"teste.csv", nl=nl, nc=nc);  #100 imagens
vx, vy = leCsv(diretorioBd,"valida.csv", nl=nl, nc=nc);  #100 imagens
ay = keras.utils.to_categorical(ay, num_classes)
qy = keras.utils.to_categorical(qy, num_classes)
vy = keras.utils.to_categorical(vy, num_classes)

input_shape = (nl,nc,3); batch_size = 10; 
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
# base_model.summary()
x = base_model.output
x = GlobalAveragePooling2D()(x) #
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

#Nao permite treinar base_model. So as camadas densas sao treinadas:
for layer in base_model.layers: layer.trainable = False
#Treina com learning rate grande
otimizador=keras.optimizers.Adam(learning_rate=1e-4)
model.compile(otimizador, loss='categorical_crossentropy', metrics =['accuracy'])
model.fit(ax, ay, batch_size=batch_size, epochs=40, verbose=2, validation_data=(vx,vy))

score = model.evaluate(ax, ay, verbose=0); print('Training loss:', score)
score = model.evaluate(vx, vy, verbose=0); print('Validation loss:', score)
score = model.evaluate(qx, qy, verbose=0); print('Test loss:', score)

#Libera todos layers do model (incluindo modelo-base) para treinar:
for layer in model.layers: layer.trainable = True
#Treina com learning rate pequena todas as camadas
model.learning_rate=1e-7
model.fit(ax, ay, batch_size=batch_size, epochs=40, verbose=2, validation_data=(vx,vy))

score = model.evaluate(ax, ay, verbose=0); print('Training loss:', score)
score = model.evaluate(vx, vy, verbose=0); print('Validation loss:', score)
score = model.evaluate(qx, qy, verbose=0); print('Test loss:', score)
model.save(nomeprog+".h5")