import os
from collections import defaultdict
from scipy.misc import imread, imresize
from random import sample, choice
import numpy as np
import numpy.random as nrandom

curDir = os.getcwd()
testDir = os.path.join(curDir,'dataset','test')
trainDir = os.path.join(curDir,'dataset','train')

class Whale_Generator:

    def __init__(self, path, size_x, size_y):
        self.path = path
        self.filenames = []
        self.size_x = size_x
        self.size_y = size_y
        for filename in os.listdir(path):
            self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, key):
        filename = self.filenames[key]
        imgpath = os.path.join(self.path, filename)
        image = imresize(imread(imgpath, flatten=True), (self.size_x, self.size_y))
        return image

class Whale_Loader:

    size_x = 150
    size_y = 150

    def __init__(self):
        self.trainImgs = self.load_imgs(trainDir)
        self.labelSet = set(self.trainImgs.keys())

    def get_folder_generator(self, path):
        return Whale_Generator(path, self.size_x, self.size_y)

    def load_imgs(self, path):
        d = {}
        for label in os.listdir(path):
            labelpath = os.path.join(path, label)
            d[label] = self.get_folder_generator(labelpath)
        return d

    def get_batch(self, size):
        labels = sample(self.trainImgs.keys(), size)
        pairs =[np.zeros((size, self.size_y, self.size_x, 1)) for i in range(2)]
        targets = np.zeros((size,))
        targets[size//2:] = 1

        for i in range(size):
            label = labels[i]
            otherLabels = self.labelSet - {label} 
            i1 = nrandom.randint(0, len(self.trainImgs[label]))
            pairs[0][i,:,:,:] =  self.trainImgs[label][i1].view().reshape((self.size_y, self.size_x, 1))
            label2 = label if i >= size//2 else choice(list(otherLabels))
            i2 = nrandom.randint(0, len(self.trainImgs[label2]))
            pairs[1][i,:,:,:] = self.trainImgs[label2][i2].view().reshape((self.size_y, self.size_x, 1))
        return pairs, targets
