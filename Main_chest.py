import os
import numpy as np
import time
import sys

from DensenetTrainer import DensenetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    
    runTest()
    #runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/media/hdd/images'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 5
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 128
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 224
    imgtransCrop = 224
        
    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    DensenetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    print ('Testing the trained model')
    DensenetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/media/hdd/images'
    pathFileTest = './dataset/train_1.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 5
    trBatchSize = 100
    imgtransResize = 224
    imgtransCrop = 224
    
    pathModel = './m-13122018-095903.pth.tar'
    
    timestampLaunch = ''
    
    DensenetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()




