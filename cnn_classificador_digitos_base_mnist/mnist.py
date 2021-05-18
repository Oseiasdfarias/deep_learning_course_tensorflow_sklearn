#!/usr/bin/env python
# coding: utf-8

# # Rede Neural Convolucional com Dataset mnist

# In[3]:


import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPool2D
import numpy as np


# In[4]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[5]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Plotando um exemplo de imagem

# In[6]:


plt.imshow(x_train[15], cmap="gray")
plt.title(f"Classe:{y_train[15]}")


# ### Pegando apenas um canal de cor das imagens

# In[7]:


prev_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
prev_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

prev_train = prev_train.astype("float32")
prev_test = prev_test.astype("float32")

plt.imshow(prev_train[15], cmap="gray")
plt.title(f"Classe:{y_train[15]}")


#  

# ### Normalizando a escala dos valores

# In[8]:


prev_train /= 255
prev_test /= 255

print(prev_train[15])
plt.imshow(prev_train[15], cmap="gray")
plt.title(f"Classe:{y_train[15]}")


#  

# ### Transformando as classes em valores tipo dummy

# In[9]:


class_train = np_utils.to_categorical(y_train, 10)
class_test = np_utils.to_categorical(y_test, 10)


# In[10]:


print(class_test[6])


# ### Criando uma rede neural convolucional

# In[13]:


classificador = Sequential() # instanciando a estrutura da rede neural

# Criando o operador de convolução
classificador.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))

# Criando o kernel de Max Pooling
classificador.add(MaxPool2D(pool_size = (2, 2)))

# Criando a canada de flatten 
classificador.add(Flatten()) 

# Criando a parte da rede neural densa
classificador.add(Dense(units=128, activation="relu"))
classificador.add(Dense(units=10, activation="softmax"))

# Conpilando a rede neural
classificador.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Iniciando o treinamento 
classificador.fit(prev_train,
                  class_train,
                  batch_size=128,
                  epochs=5,
                  validation_data=(prev_test, class_test))


# In[ ]:




