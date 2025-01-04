import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.args = args
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.criterion = torch.nn.CrossEntropyLoss()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        for i, data in enumerate(tqdm(train_loader, desc="Training")):
            data = data.to(self.args.device)
            self.optim.zero_grad()
            logits, targets = self.model(data)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Training loss: {avg_loss:.4f}")

    def eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                data = data.to(self.args.device)
                logits, targets = self.model(data)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
        
        avg_loss = total_loss / len(val_loader)
        print(f"Epoch {epoch}, Validation loss: {avg_loss:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.96))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint_path', type=str, default='./transformer_checkpoints/cosine', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=1, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_transformer.train_one_epoch(train_loader, epoch)
        train_transformer.eval_one_epoch(val_loader, epoch)
        train_transformer.scheduler.step()
        
        if epoch % args.save_per_epoch == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}.pth")
            torch.save(train_transformer.model.transformer.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")