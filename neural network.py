import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import importlib
import time
from random import randint
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import numpy as np
import h5py

allonge=np.load("allongé.npy")
assis=np.load("assis.npy")
autre=np.load("autre.npy")
debout=np.load("debout.npy")
test=np.load("test.npy")

train_data=[]
target_data=[]

for i in range(len(autre)):
    train_data.append(autre[i])
    target_data.append([0])
for i in range(len(allonge)):
    train_data.append(allonge[i])
    target_data.append([1])
for i in range(len(assis)):
    train_data.append(assis[i])
    target_data.append([2])
for i in range(len(debout)):
    train_data.append(debout[i])
    target_data.append([3])


train_data = np.array(train_data,"float32")
target_data = np.array(target_data,"float32")
model = Sequential()
model.add(Flatten(input_shape=(25,2)))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.4))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.4))
model.add(Dense(128, activation=tf.math.sigmoid))
model.add(Dense(4, activation=tf.nn.softmax))
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(train_data, target_data, batch_size = 40,epochs = 500, validation_split = 0.3)

tmp=[]
result=model.predict(test)
for i in range(len(result)):
    tmp.append([i+1,np.argmax(result[i])])

print(tmp)
