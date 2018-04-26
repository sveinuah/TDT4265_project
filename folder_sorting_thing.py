#!/usr/bin/python3

import os
import shutil

curDir = os.getcwd()
kaggleFolder = os.path.join(curDir, 'kaggle_whale')
trainingcsv = os.path.join('kaggle_whale', 'train.csv')

trainingData = []

with open(trainingcsv) as f:
    for line in f:
        temp = line.strip('\n').split(',')
        #print(temp)
        trainingData.append((temp[0],temp[1]))
trainingData = trainingData[1:]

print('Training data length:',len(trainingData))

labels = []
for whale in trainingData:
    if whale[1] not in labels:
        labels.append(whale[1])

print('Label list length:',len(labels))

try:
    os.mkdir(os.path.join(curDir, 'dataset'))
    os.mkdir(os.path.join(curDir, 'dataset', 'train'))

except FileExistsError as e:
    #print(e)
    pass

except:
    print("something went worng")

for label in labels:
    try:
        os.mkdir(os.path.join(curDir, 'dataset', 'train', str(label)))
    except FileExistsError as e:
#	print(e)
        pass

##### Soriting files into correct structure #####

errorCount = 0

for fileName in os.listdir(kaggleFolder +'/train'):
    for i in range(0,len(trainingData)):
        if fileName == trainingData[i][0]:
            try:
                shutil.move(
                    os.path.join(kaggleFolder, 'train', fileName),
                    os.path.join(curDir, 'dataset', 'train', trainingData[i][1], fileName)
                    )
            except:
                print("Could not copy file")
                errorCount += 1

print("Sorting completed with {} errors".format(errorCount))
