from keras.preprocessing.image import ImageDataGenerator

import os

curDir = os.getcwd()
testDir = os.path.join(curDir,'dataset','test')
trainDir = os.path.join(curDir,'dataset','train')

def generateDataset(imageHeight, imageWidth):
    trainDatagen = ImageDataGenerator(
            rescale=1./255
            #Kan legge til mer
            )

    testDatagen = ImageDataGenerator(rescale = 1./255)

    trainGenerator = trainDatagen.flow_from_directory(
            trainDir,
            target_size = (imageHeight,imageWidth),
            batch_size = 1,
            class_mode = 'categorical'
            )

    testGenerator = testDatagen.flow_from_directory(
            testDir,
            target_size = (imageHeight,imageWidth),
            batch_size = 1,
            class_mode = None
            )


            
    return testGenerator,trainGenerator

