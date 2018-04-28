from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Lambda, merge
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.optimizers import SGD,Adam
from keras import backend as K

import numpy as np
import os
import matplotlib.pyplot as plt
from datasetLoader import Whale_Loader

learningRate = 0.00005 #Burde tweakes
batch_size = 32
inputShape = (150,150,1) #Må endres til (x,y,1) hvis vi går over til svarthvitt

def w_init(shape,name=None):
    values = np.random.normal(loc = 0, scale = 1e-2, size = shape)
    return K.variable(values, name=name)

def b_init(shape,name=None):
    values = np.random.normal(loc = 0.5, scale = 1e-2, size=shape)
    return K.variable(values,name=name)

def createModel(shape, learningRate):
    L1Distance = lambda x: K.abs(x[0] - x[1])
    
    leftInput = Input(shape)
    rightInput = Input(shape)

    conv = Sequential()
    conv.add(Conv2D(64, (10, 10), activation='relu', input_shape=inputShape, kernel_initializer=w_init, kernel_regularizer=l2(2e-4)))
    conv.add(MaxPooling2D())
    conv.add(Conv2D(128, (7, 7), activation='relu',  kernel_regularizer=l2(2e-4), kernel_initializer=w_init, bias_initializer=b_init))
    conv.add(MaxPooling2D())
    conv.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=w_init, kernel_regularizer=l2(2e-4), bias_initializer=b_init))
    conv.add(MaxPooling2D())
    conv.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=w_init, kernel_regularizer=l2(2e-4), bias_initializer=b_init))
    conv.add(Flatten())
    conv.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=w_init, bias_initializer=b_init))

    leftConv = conv(leftInput)
    rightConv = conv(rightInput)

    twoLeggedConv = merge([leftConv, rightConv], mode = L1Distance, output_shape = lambda x: x[0])
    prediction = Dense(1,activation = 'sigmoid', bias_initializer = b_init)(twoLeggedConv)

    siameseNet = Model(input=[leftInput, rightInput], output = prediction)
    siameseNet.compile(loss='binary_crossentropy',optimizer=Adam(learningRate))
    siameseNet.summary()

    return siameseNet


evaluate_every = 1000 # interval for evaluating on one-shot tasks
loss_every=50 # interval for printing loss (iterations)
n_iter = 90000
N_way = 20 # how many classes for testing one-shot tasks>
n_val = 250 #how mahy one-shot tasks to validate on?
best = 9999
#weights_path = os.path.join(PATH, "weights")
loader = Whale_Loader()
siamese_net = createModel((150, 150, 1), 0.005)

numTests = loader.get_num_test_pictures()
(inputs,labels) = loader.get_single_test(0)

print("training")
for i in range(1, n_iter):
    (inputs,targets)=loader.get_training_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        print("evaluating")
        print("... sortof")
#        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
#        if val_acc >= best:
#            print("saving")
#            siamese_net.save(weights_path)
#            best=val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))


