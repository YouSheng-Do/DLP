import argparse
import os
import numpy as np
import math
import json

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch

from dataloader import dataset
from evaluator import evaluation_model

os.makedirs("../images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--ckpt_path", type=str, required=True, help="path to the checkpoint")
args = parser.parse_args()
print(args)

img_shape = (args.channels, args.img_size, args.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(nn.Linear(args.n_classes, 24*8*8), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.latent_dim + args.n_classes*8*8, args.img_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.img_size * 8),
            nn.ReLU(True),
            # state size. ``(args.img_size*8) x 4 x 4``
            nn.ConvTranspose2d(args.img_size * 8, args.img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.img_size * 4),
            nn.ReLU(True),
            # state size. ``(args.img_size*4) x 8 x 8``
            nn.ConvTranspose2d( args.img_size * 4, args.img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.img_size * 2),
            nn.ReLU(True),
            # state size. ``(args.img_size*2) x 16 x 16``
            nn.ConvTranspose2d( args.img_size * 2, args.img_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.img_size),
            nn.ReLU(True),
            # state size. ``(args.img_size) x 32 x 32``
            nn.ConvTranspose2d( args.img_size, args.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(args.channels) x 64 x 64``
        )

    def forward(self, noise, labels):
        # print(labels.shape)
        labels = labels.to(torch.float32)
        labels = self.linear(labels).view(labels.size(0), 24*8*8, 1, 1)
        # print(noise.shape)
        input = torch.cat([labels, noise], dim=1)
        # print(input.shape)
        output = self.main(input)
        # print(output.shape)
        return output
    
# Initialize generator
generator = Generator()
checkpoint = torch.load(args.ckpt_path)
generator.load_state_dict(checkpoint["model"])
generator.eval()

if cuda:
    generator.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

with open('../objects.json', 'r') as file:
    label_map = json.load(file)
with open('../test.json', 'r') as file:
    test_label = json.load(file)
with open('../new_test.json', 'r') as file:
    new_test_label = json.load(file)

test_label_list = []
for labels in test_label:
    multihot = torch.zeros(len(label_map), dtype=torch.float32)
    for label in labels:
        if label in label_map:
            multihot[label_map[label]] = 1.0
    test_label_list.append(multihot)
test_label_tensor = torch.stack(test_label_list)

normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

"""Saves a grid of generated digits ranging from 0 to n_classes"""
# Sample noise
z = Variable(FloatTensor(np.random.normal(0, 1, (32, args.latent_dim, 1, 1))))
# Get labels ranging from 0 to n_classes for n rows
labels = test_label_tensor.cuda()
gen_imgs = generator(z, labels)
save_image(gen_imgs.data, "../images/test.png", nrow=8, normalize=True)
normalized_gen_img = normalization(gen_imgs)
evaluator = evaluation_model()
acc = evaluator.eval(normalized_gen_img, labels)
print(acc)

new_test_label_list = []
for labels in new_test_label:
    multihot = torch.zeros(len(label_map), dtype=torch.float32)
    for label in labels:
        if label in label_map:
            multihot[label_map[label]] = 1.0
    new_test_label_list.append(multihot)
new_test_label_tensor = torch.stack(new_test_label_list)

"""Saves a grid of generated digits ranging from 0 to n_classes"""
# Sample noise
z = Variable(FloatTensor(np.random.normal(0, 1, (32, args.latent_dim, 1, 1))))
# Get labels ranging from 0 to n_classes for n rows
labels = new_test_label_tensor.cuda()
gen_imgs = generator(z, labels)
save_image(gen_imgs.data, "../images/new_test.png", nrow=8, normalize=True)
normalized_gen_img = normalization(gen_imgs)
evaluator = evaluation_model()
acc = evaluator.eval(normalized_gen_img, labels)
print(acc)
