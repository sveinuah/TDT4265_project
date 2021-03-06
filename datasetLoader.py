import os
from collections import defaultdict
from scipy.misc import imread, imresize
from random import sample, choice
import numpy as np
import numpy.random as nrandom
from copy import deepcopy

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
        if isinstance(key, slice):
            new_gen = deepcopy(self)
            new_gen.filenames = new_gen.filenames[key]
            return new_gen
        else:
            filename = self.filenames[key]
            imgpath = os.path.join(self.path, filename)
            image = (1.0/255.0)*imresize(imread(imgpath, flatten=True), (self.size_x, self.size_y)).reshape(self.size_y, self.size_x, 1)
            #print("Maks:", np.amax(image))
            return image

    def get_filenames(self):
        return self.filenames

class Whale_Loader:

    size_x = 150
    size_y = 150

    test_whales = {}

    def __init__(self):
        self.trainImgs, self.valImgs = self.load_imgs(trainDir, 0.25)
        self.labelSet = set(self.trainImgs.keys())
        self.load_test_whales()
        self.numLabels = len(self.labelSet)

        self.testImgs, _ = self.load_imgs(testDir, 0.0)
        self.testlabels = list(self.testImgs.keys())
        self.numTestPictures = len(self.testImgs[self.testlabels[0]])

    def get_num_labels(self):
        return self.numLabels

    def get_test_filenames(self):
        whalegen = self.get_folder_generator(os.path.join(testDir,"unclassified"))
        return whalegen.get_filenames()
        

    def load_test_whales(self):
        for key in self.trainImgs:
            self.test_whales[key] = self.trainImgs[key][0]
    
    def get_num_test_pictures(self):
        return self.numTestPictures

    def get_folder_generator(self, path):
        return Whale_Generator(path, self.size_x, self.size_y)

    def load_imgs(self, path, split_point):
        train = {}
        val = {}
        for label in os.listdir(path):
            labelpath = os.path.join(path, label)
            whalegen = self.get_folder_generator(labelpath)
            valgen = whalegen[:int(split_point*len(whalegen))]
            traingen = whalegen[int(split_point*len(whalegen)):]
            train[label] = traingen
            if len(valgen) > 0:
                val[label] = valgen
        return train, val

    def get_training_batch(self, size):
        labels = sample(self.trainImgs.keys(), size)
        
        try:
            new_whale_index = labels.index('new_whale')
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
    
    def make_oneshot_task(self, N):
        pairs = [np.zeros((N, self.size_y, self.size_x, 1)) for i in range(2)]
        label_subset = sample(self.valImgs.keys(), N)

        target_label = label_subset[0]
        test_img = choice(self.valImgs[target_label])
        for i in range(N):
            pairs[0][i,:,:,:] = test_img
            
            label = label_subset[i]
            pairs[1][i,:,:,:] = self.test_whales[label]
        
        targets = np.zeros((N,))
        targets[0] = 1
        return pairs, targets

    def test_oneshot(self, model, N, k):
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            outputs = model.predict(inputs)
            if np.argmax(outputs) == 0:
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        return percent_correct

    def get_single_test(self, whaleIndex):
        
        if whaleIndex >= self.numTestPictures:
            raise IndexError("Whale index out of range")


        labels = list(self.labelSet - {"new_whale"})
        pairs = [np.zeros((len(labels), self.size_y, self.size_x,1)) for i in range(2)]
        picture = self.testImgs[self.testlabels[0]][whaleIndex]

        for i in range(len(labels)):
            pairs[0][i,:,:,:] = picture
            
            label = labels[i]
            pairs[1][i,:,:,:] = self.test_whales[label]

        return pairs, labels

