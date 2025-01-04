import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms
import os
import torch

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'eval':
        df = pd.read_csv('dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    else:
        df = pd.read_csv('dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        # load img and label
        img_path = os.path.join(self.root, self.img_name[index])
        # print(img_path)
        img = Image.open(img_path)
        # print(img)
        label = self.label[index]
            
        # transform the img
        if self.mode == 'train':
            transformations =  transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=(224, 224), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
        else:     
            transformations =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
        img = transformations(img)
        # print(img)
        # print(img.shape)

        return img, label

# dataloader = BufferflyMothLoader('dataset', 'train')
# img, label = dataloader[0]
# print(f"{type(img) = }, {img.dtype = }, {img.shape = }, {label = }")


