import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3';
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf; import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image; from matplotlib import pyplot as plt
import numpy as np; import sys; from sys import argv

# Use os comandos abaixo se for chamar do prompt
# if (len(argv)!=2):
#   print("classif1.py nomeimg.ext");
#   sys.exit("Erro: Numero de argumentos invalido.");

# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# model = ResNet50(weights='imagenet')
# target_size = (224, 224)

# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
# model = InceptionV3(weights='imagenet')
# target_size = (299, 299)

# from tensorflow.keras.applications.inception_resnet_v2 import \
#  InceptionResNetV2, preprocess_input, decode_predictions
# model = InceptionResNetV2(weights='imagenet')
# target_size = (299, 299)

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
model = EfficientNetB0(weights='imagenet')
target_size = (224, 224)

#img_path = argv[1] #Escreva aqui o diretorio e nome da imagem
img_path = "orangotango.jpg"
img = image.load_img(img_path, target_size=target_size)
plt.imshow(img); plt.axis("off")
plt.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
p=decode_predictions(preds, top=3)[0]
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#print('Predicted:', p)
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), ... ]

for i in range(len(p)):
  print("%8.2f%% %s"%(100*p[i][2],p[i][1]))