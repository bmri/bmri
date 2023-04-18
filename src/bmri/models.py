import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import tensorflow as tf

from tensorflow.math import divide as division
from tensorflow import keras
# import tensorflow_addons as tfa
from functools import partial

np.random.seed(1008)

#reg_param   = 1e-5
Act_func   = 'ReLU'
Act_param  = 0.2
use_bias   = True

if Act_func == 'ReLU':
  activation_func = keras.layers.ReLU(negative_slope=Act_param)
elif Act_func == 'tanh':  
  activation_func = keras.layers.Activation('tanh')
elif Act_func == 'sigmoid':  
  activation_func = keras.layers.Activation('sigmoid') 
elif Act_func == 'sin':
  activation_func = tf.math.sin   

def CondInsNorm(input_X,input_Z,reg_param=1.0e-7):
  N,H,W,Nx = input_X.shape

  Nz = input_Z.shape[3]
  S1 = keras.layers.Conv2D(filters=Nx,
                           kernel_size=1,
                           strides=1,
                           padding='valid',
                           activation=activation_func,
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias
                           )(input_Z)
  S2 = keras.layers.Conv2D(filters=Nx,
                           kernel_size=1,
                           strides=1,
                           padding='valid',
                           activation=activation_func,
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias
                           )(input_Z)
  
  X  = keras.layers.Reshape((H*W,Nx), input_shape=(H,W,Nx))(input_X)
  Xs = keras.layers.Lambda(lambda x: division(x - tf.reduce_mean(x,axis=1,keepdims=True),
                                               1.0e-6+tf.sqrt(tf.math.reduce_variance(x,axis=1,keepdims=True))))(X)
  Xs = keras.layers.Reshape((H,W,Nx), input_shape=(H*W,Nx))(Xs)
  Xs = keras.layers.Multiply()([Xs,S1])
  Y  = keras.layers.Add()([Xs,S2])

  return Y

def ResBlock(input_X,input_Z=None,normalization=None,reg_param=1.0e-7):
  N,H,W,Nx = input_X.shape

  padding = tf.constant([[0,0],[1,1],[1,1], [0, 0]])

  X = input_X

  if normalization =='cin':
    assert input_Z is not None
    X = CondInsNorm(input_X=X,input_Z=input_Z)
  elif normalization == 'in':  
    X = tfa.layers.InstanceNormalization()(X)
  elif normalization == 'bn':
    X = keras.layers.BatchNormalization()(X)
  elif normalization == 'ln':
    X = keras.layers.LayerNormalization()(X)  
    

  X1  = keras.layers.Conv2D(filters=Nx,kernel_size=1,strides=1,padding='valid',kernel_regularizer=keras.regularizers.l2(reg_param))(X)  

  X2  = activation_func(X)
  X2  = tf.pad(X2,padding,"REFLECT")
  X2  = keras.layers.Conv2D(filters=Nx,
                             kernel_size=3,
                             strides=1,
                             padding='valid',
                             kernel_regularizer=keras.regularizers.l2(reg_param),
                             use_bias=use_bias
                             )(X2)

  if normalization =='cin':
    assert input_Z is not None
    X2 = CondInsNorm(input_X=X2,input_Z=input_Z)  
  elif normalization == 'in':  
    X2 = tfa.layers.InstanceNormalization()(X2)  
  elif normalization == 'bn':
    X2 = keras.layers.BatchNormalization()(X2)
  elif normalization == 'ln':
    X2 = keras.layers.LayerNormalization()(X2)      
    
  X2 = activation_func(X2)    

  X2  = tf.pad(X2,padding,"REFLECT")
  X2  = keras.layers.Conv2D(filters=Nx,
                             kernel_size=3,
                             strides=1,
                             padding='valid',
                             kernel_regularizer=keras.regularizers.l2(reg_param),
                             use_bias=use_bias
                             )(X2) 

  Y = keras.layers.Add()([X1,X2])                                                

  return Y  

def DownSample(input_X,k,downsample=True,activation=True,reg_param=1.0e-7):

  padding = tf.constant([[0,0],[1,1],[1,1], [0, 0]])

  X  = tf.pad(input_X,padding,"REFLECT")
  X  = keras.layers.Conv2D(filters=k,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias)(X)
  if activation:
    X = activation_func(X)                          
  if downsample:
    X = keras.layers.AveragePooling2D(pool_size=2,strides=2)(X)
                                                
  return X


def UpSample(input_X,k,old_X=None,concat=False,upsample=True,activation=True,reg_param=1.0e-7):

  padding = tf.constant([[0,0],[1,1],[1,1], [0, 0]])

  if concat:
    X = keras.layers.Concatenate()([input_X,old_X])
  else:
    X = input_X  

  X  = tf.pad(X,padding,"REFLECT")
  X  = keras.layers.Conv2D(filters=k,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias)(X)
  if activation:
    X = activation_func(X)

  if upsample:
    X = keras.layers.UpSampling2D(size=2)(X)
                                                
  return X  

def generator_BE(Input_W=64,Input_H=64,Input_C=1,Z_Dim=50,k0=32,reg_param=1.0e-7):
  input_X = keras.Input(shape=(Input_W,Input_H,Input_C))
  input_Z = keras.Input(shape=(1,1,Z_Dim))

  # Downsampling + ResBlock
  X1  = DownSample(input_X=input_X,k=k0,downsample=False,reg_param=reg_param)
  X1  = ResBlock(input_X=X1,input_Z=input_Z,reg_param=reg_param)
  
  X2  = DownSample(input_X=X1,k=2*k0,reg_param=reg_param)
  X2  = ResBlock(input_X=X2,input_Z=input_Z,normalization='cin',reg_param=reg_param)

  X3  = DownSample(input_X=X2,k=4*k0,reg_param=reg_param)
  X3  = ResBlock(input_X=X3,input_Z=input_Z,normalization='cin',reg_param=reg_param)

  # Final downsampling + ResBlock (not to be concatenated)
  X4  = DownSample(input_X=X3,k=8*k0,reg_param=reg_param)
  X4  = ResBlock(input_X=X4,input_Z=input_Z,normalization='cin',reg_param=reg_param)

  # Coarsest level ResBlock
  X5  = ResBlock(input_X=X4,input_Z=input_Z,normalization='cin',reg_param=reg_param)
  
  # Upsampling + ResBlock
  X6  = UpSample(input_X=X5,k=8*k0,reg_param=reg_param)
  X6  = ResBlock(input_X=X6,input_Z=input_Z,normalization='cin',reg_param=reg_param)

  # Upsampling + Concat + ResBlock
  X7  = UpSample(input_X=X6,k=4*k0,concat=True,old_X=X3,reg_param=reg_param)
  X7  = ResBlock(input_X=X7,input_Z=input_Z,normalization='cin',reg_param=reg_param)

  X8  = UpSample(input_X=X7,k=2*k0,concat=True,old_X=X2,reg_param=reg_param)
  X8  = ResBlock(input_X=X8,input_Z=input_Z,normalization='cin',reg_param=reg_param)

  X9  = UpSample(input_X=X8,k=k0,concat=True,old_X=X1,upsample=False,reg_param=reg_param)
  X9  = ResBlock(input_X=X9,input_Z=input_Z,normalization='cin',reg_param=reg_param)

  X10  = UpSample(input_X=X9,k=Input_C,upsample=False,activation=False,reg_param=reg_param)

  X11  = keras.activations.sigmoid(X10)

  model = keras.Model(inputs=[input_X, input_Z], outputs=X11)

  return model  
