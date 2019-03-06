import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class ImageGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, images, labels, transform):
    
        self.images = images
        self.labels = labels
        self.transform = transform
    
    #--------------------------------------------------------------------------------
    
    def __getitem__(self, index):
        
        imageData = Image.fromarray(self.images[index]).convert('RGB')
        if len(self.labels) != 0:
            imageLabel= torch.FloatTensor(self.labels[index])
        else:
            imageLabel = []
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.images)
    
 #-------------------------------------------------------------------------------- 
    