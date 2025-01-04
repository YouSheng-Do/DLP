import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

class dataset(Dataset):
    def __init__(self, annotations_file, label_map, img_dir, transform=None):
        with open(label_map, 'r') as file:
            self.label_map = json.load(file)
        with open(annotations_file, 'r') as file:
            self.img_label = json.load(file)
        self.img_dir = img_dir
        self.transform = transform
        self.keys = list(self.img_label.keys())

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_name = self.keys[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.img_label[img_name]
        label = self.labels_to_multihot(label)
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def labels_to_multihot(self, labels):
        multihot = torch.zeros(len(self.label_map), dtype=torch.float32)
        for label in labels:
            if label in self.label_map:
                multihot[self.label_map[label]] = 1.0
        return multihot