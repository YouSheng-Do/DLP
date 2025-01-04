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
os.makedirs("../ckpt/generator", exist_ok=True)
os.makedirs("../ckpt/discriminator", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
parser.add_argument("--resume", action='store_true', help="resume training from previous epoch or not")
parser.add_argument("--g_ckpt", type=str, help="path to generator checkpoint for resume training")
parser.add_argument("--d_ckpt", type=str, help="path to discriminator checkpoint for resume training")
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Sequential(nn.Linear(24, 3*64*64), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is ``(args.channels) x 64 x 64``
            nn.Conv2d(args.channels*2, args.img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(args.img_size) x 32 x 32``
            nn.Conv2d(args.img_size, args.img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(args.img_size*2) x 16 x 16``
            nn.Conv2d(args.img_size * 2, args.img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(args.img_size*4) x 8 x 8``
            nn.Conv2d(args.img_size * 4, args.img_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(args.img_size*8) x 4 x 4``
            nn.Conv2d(args.img_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        # print(images.shape)
        # print(labels.shape)
        labels = labels.to(torch.float32)
        labels = self.linear(labels).view(labels.size(0), 3, 64, 64)
        # print(labels.shape)
        input = torch.cat([labels, images], dim=1)
        return self.main(input).view(input.size(0), 1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Loss fuargs.channelstions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)

if args.resume:
    checkpoint_G = torch.load(os.path.join(args.g_ckpt))
    checkpoint_D = torch.load(os.path.join(args.d_ckpt))
    
    generator.load_state_dict(checkpoint_G['model'])
    optimizer_G.load_state_dict(checkpoint_G['optimizer'])
    scheduler_G.load_state_dict(checkpoint_G['scheduler'])
    start_epoch = checkpoint_G['epoch'] + 1
    
    discriminator.load_state_dict(checkpoint_D['model'])
    optimizer_D.load_state_dict(checkpoint_D['optimizer'])
    scheduler_D.load_state_dict(checkpoint_D['scheduler'])
    print(f"Resumed training from epoch {start_epoch}")
else:
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    start_epoch = 0

# Configure data loader
dataloader = DataLoader(
    dataset(
        annotations_file="../train.json",
        label_map="../objects.json",
        img_dir="../iclevr",
        transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, args.latent_dim, 1, 1))))
    # # Get labels ranging from 0 to n_classes for n rows
    # labels = Variable(LongTensor(np.random.randint(0, 2, size=(n_row ** 2, 24))))
    # gen_imgs = generator(z, labels)
    # save_image(gen_imgs.data, "../images/%d.png" % epoch, nrow=n_row, normalize=True)
    
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (32, args.latent_dim, 1, 1))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = test_label_tensor.cuda()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"../images/{epoch}_test.png", nrow=8, normalize=True)
    normalized_gen_img = normalization(gen_imgs)
    evaluator = evaluation_model()
    acc = evaluator.eval(normalized_gen_img, labels)
    print(acc)
    
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (32, args.latent_dim, 1, 1))))
    labels = new_test_label_tensor.cuda()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"../images/{epoch}_new_test.png", nrow=8, normalize=True)
    normalized_gen_img = normalization(gen_imgs)
    evaluator = evaluation_model()
    acc = evaluator.eval(normalized_gen_img, labels)
    print(acc)

        
    
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

new_test_label_list = []
for labels in new_test_label:
    multihot = torch.zeros(len(label_map), dtype=torch.float32)
    for label in labels:
        if label in label_map:
            multihot[label_map[label]] = 1.0
    new_test_label_list.append(multihot)
new_test_label_tensor = torch.stack(new_test_label_list)


# ----------
#  Training
# ----------

for epoch in range(start_epoch, args.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        # print(labels.shape)
        # print(real_imgs.shape)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim, 1, 1))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, 2, size=(batch_size, 24))))
        # print(gen_labels)
        # Generate a batch of images
        gen_imgs = generator(z, labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, labels)
        # print(validity.shape)
        # print(valid.shape)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
    
    scheduler_G.step()
    scheduler_D.step()
            
    # save generator model
    state_dict = {
        "model": generator.state_dict(),
        "optimizer": optimizer_G.state_dict(),
        "scheduler": scheduler_G.state_dict(),
        "epoch": epoch,
    }
    torch.save(state_dict, os.path.join("../ckpt/generator", f"g_epoch_{epoch}.pth"))
    del state_dict

    # save discriminator model
    save_path = os.path.join("../ckpt/discriminator", f"d_epoch_{epoch}.pth")
    state_dict = {
        "model": discriminator.state_dict(),
        "optimizer": optimizer_D.state_dict(),
        "scheduler": scheduler_D.state_dict(),
        "epoch": epoch,
    }
    torch.save(state_dict, save_path)
    
    if epoch % args.sample_interval == 0:
        sample_image(n_row=10, epoch=epoch)