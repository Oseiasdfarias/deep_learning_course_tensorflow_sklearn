import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

# Carregamento da base de dados (na primeira execução será feito o download)
(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

# Mostra a imagem e a respectiva classe, de acordo com o índice passado como parâmetro
# Você pode testar os seguintes índices para visualizar uma imagem de cada classe
# Avião - 650
# Pássaro - 6
# Gato - 9
# Veado - 3
# Cachorro - 813
# Sapo - 651
# Cavalo - 652
# Barco - 811
# Caminhão - 970
# Automóvel - 4
plt.imshow(X_treinamento[4])
plt.title('Classe '+ str(y_treinamento[4]))

# As dimensões dessas imagens é 32x32 e o número de canails é 3 pois vamos utilizar as imagens coloridas
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)

# Conversão para float para podermos aplicar a normalização
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# Normalização para os dados ficarem na escala entre 0 e 1 e agilizar o processamento
previsores_treinamento /= 255
previsores_teste /= 255

# Criação de variáveis do tipo dummy, pois teremos 10 saídas
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# Criação da rede neural com duas camadas de convolução
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

# Rede neural densa com duas camadas ocultas
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', 
                      optimizer="adam", metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, 
                  batch_size=128, epochs=5, 
                  validation_data=(previsores_teste, classe_teste), verbose=2)



