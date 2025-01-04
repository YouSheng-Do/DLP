import argparse
import os
import csv
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score
from evaluate import evaluate

def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # since it is binary semantic segmentation task, num_classes=1
    model = UNet(num_channels=3, num_classes=1) if args.model == "unet" else ResNet34_UNet(num_classes=1)

    model = model.to(device)

    train_dataset = load_dataset(args.data_path, "train")
    val_dataset = load_dataset(args.data_path, "valid")
    
    print("Loading dataset...")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = nn.BCEWithLogitsLoss()


    best_score = 0.0
    print(f"Start training {args.model}...")

    with open(f'{args.model}_dice_scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Epoch', 'Train Dice Score', 'Val Dice Score'])

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            train_score = 0.0
            model.train()
            for sample in train_dataloader:
                images, masks = sample['image'].to(device), sample['mask'].to(device)
                pred_masks = model(images)
                optimizer.zero_grad()

                loss = criterion(pred_masks.squeeze(1), masks.float())
                score = dice_score(pred_mask=torch.sigmoid(pred_masks.squeeze(1)), gt_mask=masks.float())
                loss += (1. - score)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_score += score.item()
                
            train_score /= len(train_dataloader)
            epoch_loss /= len(train_dataloader)

            val_score = evaluate(model=model, args=args, val_dataloader=val_dataloader, device=device)
            
            writer.writerow([epoch, train_score, val_score])
            print(f'Epoch {epoch}: Train BCE + Dice loss: {epoch_loss}, Train Dice Score: {train_score}, Valid Dice Score: {val_score}')
        
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), os.path.join(args.save_path, f'best_{args.model}.pth'))
                print(f'Saved best {args.model} with dice score: {best_score}')
     
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet",help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--save_path', type=str, default="../saved_models",help='path of saved models')
    parser.add_argument('--model', type=str, default="resnet34_unet", help="choose which model you want to train")

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args=args)
