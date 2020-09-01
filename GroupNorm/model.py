import tensorflow as tf
from tensorflow import keras
from groupnorm import GroupNorm
def make_simple_model(input_shape=(28,28,1),norm='group'):
  inp=keras.layers.Input(input_shape)
  model_gn=keras.layers.Conv2D(128,kernel_size=3,strides=(1,1),padding='same')(inp)
  if norm=='group':
    model_gn=GroupNorm()(model_gn)
  else:
    keras.layers.BatchNormalization()(model_gn)
  model_gn=keras.layers.GlobalAveragePooling2D()(model_gn)
  model_gn=keras.layers.Dense(10,activation='softmax')(model_gn)
  model_gn=keras.models.Model(inputs=[inp],outputs=[model_gn])
  model_gn.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(1e-4),metrics=['accuracy'])
  return model_gn