import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import image_data_format

class GroupNorm(Layer):
  """
  Reimplementation of GroupNorm using the excellent post 
  https://amaarora.github.io/2020/08/09/groupnorm.html
  """
  def __init__(self,groups=32,**kwargs):
    """
    Arguments:
    groups: The number of groups that the channels are divided into (Default value=32)
    eps: The value used in order to prevent zero by division errors
    """
    super(GroupNorm,self).__init__(**kwargs)
    self.g=groups
    self.eps=1e-5
    if image_data_format()=='channels_first':
      self.axis=1
    else:
      self.axis=-1  
  def build(self,input_shape):
    """
    Arguments:
    input_shape: The shape of the feature maps in the form N*H*W*C
    """
    shape=[1,1,1,1]
    shape[self.axis]=int(input_shape[self.axis])
    self.gamma=self.add_weight('gamma',
                                shape=shape)
    self.beta=self.add_weight('gamma',
                                shape=shape)
    super().build(input_shape)
  def call(self,inputs):
    """
    Arguments:
    inputs: The transformed features from the previous layers
    """
    input_shape=K.int_shape(inputs)
    n,h,w,c=input_shape
    tensor_shape=tf.shape(inputs)
    shape=[tensor_shape[i] for i in range(len(input_shape))]
    shape[self.axis]=shape[self.axis]//self.g
    shape.insert(self.axis,self.g)
    shape=tf.stack(shape) 
    x=tf.reshape(inputs,shape=shape)
    mean,variance=tf.nn.moments(x,axes=[1,2,3],keepdims=True)
    x_transformed=(x-mean)/tf.sqrt(variance+self.eps)
    x_transformed=tf.reshape(x_transformed,shape=tensor_shape)
    x_transformed=self.gamma*x_transformed+self.beta
    return x_transformed