import argparse
import os
import numpy as np
import math
import json

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch.nn as nn
import torch

from dataloader import dataset
from evaluator import evaluation_model

os.makedirs("../images/ddpm", exist_ok=True)
os.makedirs("../ckpt/ddpm", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--resume", action='store_true', help="resume training from previous epoch or not")
parser.add_argument("--ckpt", type=str, help="path to ddpm checkpoint for resume training")
args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()
    
    self.linear = nn.Sequential(nn.Linear(args.n_classes, class_emb_size), nn.LeakyReLU())

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=64,           # the target image resolution
        in_channels=3 + class_emb_size, # Additional input channels for class cond.
        out_channels=3,           # the number of output channels
        layers_per_block=4,       # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256), 
        down_block_types=(
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D", 
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # class conditioning in right shape to add as additional input channels
    class_cond = self.linear(class_labels)
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 3, 64, 64) and class_cond is now (bs, 4, 64, 64)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 7, 64, 64)
    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 3, 64, 64)

# Initialize Conditional DDPM model
ddpm_model = ClassConditionedUnet(num_classes=24, class_emb_size=1).to(device)
optimizer = torch.optim.Adam(ddpm_model.parameters(), lr=args.lr)

# Create noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')

evaluator = evaluation_model()

if args.resume:
    checkpoint = torch.load(os.path.join(args.ckpt))
    ddpm_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed training from epoch {start_epoch}")
else:
    start_epoch = 0

# Configure data loader
dataloader = DataLoader(
    dataset(
        annotations_file="../train.json",
        label_map="../objects.json",
        img_dir="../iclevr",
        transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.CenterCrop((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

def sample_image(n_row, epoch):
    """Saves a grid of generated images"""
    x = torch.randn(32, args.channels, args.img_size, args.img_size, device=device)
    y = test_label_tensor.to(device)
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residual = ddpm_model(x, t, y)  # Again, note that we pass in our labels y

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample
    
    score = evaluator.eval(x, y)
    total_score = score
    print(score)
    grid = make_grid(x, nrow=n_row, normalize=True)
    save_image(grid.data, f"../images/ddpm/{epoch}_test.png")
    
    x = torch.randn(32, args.channels, args.img_size, args.img_size, device=device)
    y = new_test_label_tensor.to(device)
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residual = ddpm_model(x, t, y)  # Again, note that we pass in our labels y

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample
    
    score = evaluator.eval(x, y)
    total_score += score
    print(score)
    grid = make_grid(x, nrow=n_row, normalize=True)
    save_image(grid.data, f"../images/ddpm/{epoch}_new_test.png")
    
    return total_score / 2

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

criterion = nn.MSELoss()
best_score = 0.0
avg_score = 0.0
losses = []
for epoch in range(start_epoch, args.n_epochs):
    ddpm_model.train()
    epoch_loss = 0.0
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        pred = ddpm_model(noisy_x, timesteps, y)

        loss = criterion(pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(dataloader)
    print(f"Loss of Epoch {epoch} : {epoch_loss}")
    losses.append(epoch_loss)

    if epoch % args.sample_interval == 0:
        avg_score = sample_image(n_row=8, epoch=epoch)
        
    # save model
    state_dict = {
        "model": ddpm_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state_dict, os.path.join("../ckpt/ddpm", f"epoch_{epoch}.pth"))
    
    if avg_score > best_score:
        state_dict = {
            "model": ddpm_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(state_dict, os.path.join("../ckpt/ddpm", f"ddpm_best.pth"))

plt.plot(losses)
plt.savefig('./ddpm_loss_accuracy_curve.png')