from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Lambda, merge
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.optimizers import SGD,Adam
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

learningRate = 0.005 #Burde tweakes
inputShape = (105,105,3) #Må endres til (x,y,1) hvis vi går over til svarthvitt

def w_init(shape,name=None):
    values = np.random.normal(loc = 0, scale = 1e-2, size = shape)
    return K.variable(values, name=name)

def b_init(shape,name=None):
    values = np.random.normal(loc = 0, scale = 1e-2, size=shape)
    return K.variable(values,name=name)

def createModel(shape, learningRate):
    L1Distance = lambda x: K.abs(x[0] - x[1])
    
    leftInput = Input(shape)
    rightInput = Input(shape)

    conv.add(conv2D(64,(10,10),activation='relu',input_shape=input_shape,kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
    conv.add(MaxPooling2D())
    conv.add(Conv2D(128,(7,7),activation='relu', kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
    conv.add(MaxPooling2D())
    conv.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    conv.add(MaxPooling2D())
    conv.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    conv.add(Flatten())
    conv.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))

    leftConv = conv(leftInput)
    rightConv = conv(rightInput)

    twoLeggedConv = merge([leftConv, rightConv], mode = L1Distance, output_shape = lambda x: x[0])
    prediction = Dense(1,activation = 'sigmoid', bias_initializer = b_init)(twoLeggedConvNet)

    siameseNet = Model(input=[leftInput, rightInput], output = prediction)
    siameseNet.compile(loss='binary_crossentropy',optimizer=Adam(learningRate))
    siameseNet.summary()

    return siameseNet
