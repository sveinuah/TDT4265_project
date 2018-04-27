import os
from collections import defaultdict
from scipy.misc import imread, imresize
from random import sample
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
            print("\tLegger til fil", filename, "i mappen", path)
            self.filenames.append(filename)

    def __len__(self):
        print("Det finnes", len(self.filenames), "filer her")
        return len(self.filenames)

    def __getitem__(self, key):
        if(key >= len(self)):
            print("WTF?!")
        print("Henter ut bilde nr", key, "fra path", self.path)
        filename = self.filenames[key]
        print("Det heter", filename)
        imgpath = os.path.join(self.path, filename)
        image = imresize(imread(imgpath, flatten=True), (self.size_x, self.size_y))
        return image

class Whale_Loader:

    size_x = 150
    size_y = 150

    def __init__(self):
        self.trainImgs = self.load_imgs(trainDir)

    def get_folder_generator(self, path):
        return Whale_Generator(path, self.size_x, self.size_y)

    def load_imgs(self, path):
        d = {}
        for label in os.listdir(path):
            print("Loading label:", label)
            labelpath = os.path.join(path, label)
            d[label] = self.get_folder_generator(labelpath)
        return d

    def get_batch(self, size):
        labels = sample(self.trainImgs.keys(), size)
        #labels = ["new_whale" for n in range(size)]
        pairs =[np.zeros((size, self.size_y, self.size_x, 1)) for i in range(2)]
        targets = np.zeros((size,))
        targets[size//2:] = 1
        for i in range(size):
            label = labels[i]
            i1 = nrandom.randint(0, len(self.trainImgs[label]))
            pairs[0][i,:,:,:] = self.trainImgs[label][i1].reshape(self.size_y, self.size_x, 1)
            i2 = nrandom.randint(0, len(self.trainImgs[label]))
            label2 = label if i >= size//2 else (label + nrandom.randint(1, len(self.trainImgs))) % len(self.trainImgs)
            pairs[1][i,:,:,:] = self.trainImgs[label2][i2].reshape(self.size_y, self.size_x, 1)
        return pairs, targets

loader = Whale_Loader()
paris, targets = loader.get_batch(8)
