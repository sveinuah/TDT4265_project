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
        image = imresize(imread(imgpath, flatten=True), (self.size_x, self.size_y)).reshape(self.size_y, self.size_x, 1)
        return image

class Whale_Loader:

    size_x = 150
    size_y = 150

    test_whales = {}

    def __init__(self):
        self.trainImgs = self.load_imgs(trainDir)
        self.labelSet = set(self.trainImgs.keys())
        self.load_test_whales()
        self.numLabels = len(self.labelSet)

        self.testImgs = self.load_imgs(testDir)
        self.testlabels = list(self.testImgs.keys())
        self.numTestPictures = len(self.testImgs[self.testlabels[0]])

    def get_num_labels(self):
        return self.numLabels

    def load_test_whales(self):
        for key in self.trainImgs:
            self.test_whales[key] = self.trainImgs[key][0]
    
    def get_num_test_pictures(self):
        return self.numTestPictures

    def get_folder_generator(self, path):
        return Whale_Generator(path, self.size_x, self.size_y)

    def load_imgs(self, path):
        d = {}
        for label in os.listdir(path):
            labelpath = os.path.join(path, label)
            d[label] = self.get_folder_generator(labelpath)
        return d

    def get_training_batch(self, size):
        labels = sample(self.trainImgs.keys(), size)
        
        try:
            new_whale_index = label.index('new_whale')
            if new_whale_index >= size//2:
                whale = labels.pop(new_whale_index)
                labels.insert(0,whale)
        except ValueError:
            pass
        
        pairs =[np.zeros((size, self.size_y, self.size_x, 1)) for i in range(2)]
        targets = np.zeros((size,))
        targets[size//2:] = 1

        for i in range(size):
            label = labels[i]
            otherLabels = self.labelSet - {label} 
            i1 = nrandom.randint(0, len(self.trainImgs[label]))
            pairs[0][i,:,:,:] =  self.trainImgs[label][i1]
            label2 = label if i >= size//2 else choice(list(otherLabels))
            i2 = nrandom.randint(0, len(self.trainImgs[label2]))
            pairs[1][i,:,:,:] = self.trainImgs[label2][i2]
        return pairs, targets

    def get_single_test(self, whaleIndex):
        
        if whaleIndex >= self.numTestPictures:
            raise IndexError("Whale index out of range")
        
        pairs = [np.zeros((self.numLabels, self.size_y, self.size_x,1)) for i in range(2)]
        picture = self.testImgs[self.testlabels[0]][whaleIndex]
        labels = list(self.labelSet)

        for i in range(self.numLabels):
            pairs[0][i,:,:,:] = picture
            
            label = labels[i]
            pairs[1][i,:,:,:] = self.test_whales[label]

        return pairs, labels

