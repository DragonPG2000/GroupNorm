import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Layer
from model import make_simple_model
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
Y_train,Y_test=tf.keras.utils.to_categorical(Y_train,num_classes=10),tf.keras.utils.to_categorical(Y_test,num_classes=10)


models={'group_norm':make_simple_model(norm='group'),'batch_norm':make_simple_model(norm='batch')}

for norm,model in models.items():
  print(f'Running with {norm}')
  history=model.fit(X_train,Y_train,batch_size=32,epochs=10,validation_data=(X_test,Y_test))