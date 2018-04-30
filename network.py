from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Lambda, merge, add
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.optimizers import SGD,Adam
from keras import backend as K

import numpy as np
import os
import matplotlib.pyplot as plt
from datasetLoader import Whale_Loader

learningRate = 0.01 #Burde tweakes
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
    conv.add(Conv2D(64, (10, 10), activation='relu', input_shape=shape, kernel_initializer=w_init, kernel_regularizer=l2(2e-4)))
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
    siameseNet.compile(loss='binary_crossentropy',optimizer=SGD(learningRate))
    siameseNet.summary()

    return siameseNet

evaluate_every = 700 # interval for evaluating on one-shot tasks
loss_every=30 # interval for printing loss (iterations)
n_iter = 90000
N_way = 20 # how many classes for testing one-shot tasks>
n_val = 250 #how mahy one-shot tasks to validate on?
best = 0
weights_path = os.path.join(os.path.dirname(__file__), "weights")
loader = Whale_Loader()
siamese_net = createModel((150, 150, 1), 0.005)

numTests = loader.get_num_test_pictures()

print("training")
for i in range(1, n_iter):
    (inputs,targets)=loader.get_training_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        print("Evaluating")
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val)
        with open(os.path.join(os.path.dirname(__file__), "accuracy.csv"), 'a') as outfile:
            print(i, val_acc, sep=",", file=outfile)
        if val_acc >= best:
            siamese_net.save(weights_path)
            best=val_acc

    if i % loss_every == 0:
        with open(os.path.join(os.path.dirname(__file__), "loss.csv"), 'a') as outfile:
            print(i, loss, sep=",", file=outfile)


## Classifying test images
print("Loading weights")
siamese_net.load_weights(weights_path)

print("Predicting")
threshold = 0.75
predictions = []
filenames = loader.get_test_filenames()

with open(os.path.join(os.path.dirname(__file__), "predictions.csv"), 'a') as outfile:
    print("Image", "id", sep=',', file=outfile)

for i in range(9100, 10000):
    print("Tester bilde:", i, "ved navn", filenames[i])
    test_pairs, labels = loader.get_single_test(i)
    result = siamese_net.predict(test_pairs, batch_size = 256, verbose=0)

    best_guess = "new_whale"
    best_guess_val = 0.0
    
    for j in range(len(labels)):
        if result[j] > threshold and result[j] > best_guess_val:
            best_guess = labels[j]
            best_guess_val = result[j]

    predictions.append(best_guess)

    with open(os.path.join(os.path.dirname(__file__), "predictions1_5.csv"), 'a') as outfile:
        print(filenames[i], best_guess, sep=',', file=outfile)

