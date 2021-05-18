# -*- coding: utf-8 -*-
"""cifar10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z65UAWlg5PBq3jT_QBlYWGppm50ibZHZ

# Criando uma Rede Neural para o dataset Cifar10
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.utils import np_utils
from keras import datasets
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

"""## Plotando uma exemplo"""

plt.imshow(x_train[15], cmap="gray")
plt.title(f"Classe:{y_train[15]}")

"""# Pré-Processamento da base de dados

### Normalizando a escala dos valores
"""

prev_train = x_train / 255
prev_test = x_test / 255

print(prev_test)

"""### Transformando as classes em valores tipo dummy"""

class_train = np_utils.to_categorical(y_train, 10)
class_test = np_utils.to_categorical(y_test, 10)

print(class_test)

"""# Criando a estrutura de Rede Neural"""

# Instanciando a estrutura de rede neural
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3),padding="same", activation="relu"))
classificador.add(Dropout(0.2))
classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size=(2, 2),strides=2, padding="valid"))

classificador.add(Conv2D(64, (3, 3),padding="same", activation="relu"))
classificador.add(Dropout(0.2))
classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size=(2, 2),strides=2, padding="valid"))

classificador.add(Conv2D(128, (3, 3),padding="same", activation="relu"))
classificador.add(Dropout(0.2))
classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size=(2, 2),strides=2, padding="valid"))

# Adicionando a camada de flatten necessária para conectar a rede neural convolucional as camadas densas de uma
# rede neural densa
classificador.add(Flatten())

# Adicionando as camadas densas da rede
classificador.add(Dense(units=128, activation="relu"))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation="relu"))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation="softmax"))

# compilando o modelo
classificador.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])


classificador.summary()

# Treinando o modelo
classificador.fit(prev_train,
                  class_train,
                  batch_size=128,
                  epochs=15,
                  validation_data=(prev_test, class_test))

