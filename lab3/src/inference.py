import argparse
import torch
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score
from utils import save_prediction


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--weight', default='../saved_models/best_resnet34_unet.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet",help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--model', type=str, default="resnet34_unet", help="choose which model you want to train")
    parser.add_argument('--save_path', type=str, default="resnet34_unet_predict_images")

    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    try:
        os.mkdir(args.save_path)
    except FileExistsError:
        print("Directory exists!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(num_channels=3, num_classes=1) if args.model == "unet" else ResNet34_UNet(num_classes=1)
    model = model.to(device)
    model.load_state_dict(torch.load(args.weight))
    model.eval()

    test_dataset = load_dataset(args.data_path, "test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    total_score = 0.0
    with torch.no_grad():
        for sample in test_loader:
            images, masks, filenames = sample['image'].to(device), sample['mask'].to(device), sample['filename']
            pred_masks = model(images)
            pred_masks = torch.sigmoid(pred_masks.squeeze(1))
            pred_masks_bin = (pred_masks > 0.5).float()
            score = dice_score(pred_mask=pred_masks_bin, gt_mask=masks.float())
            total_score += score.item()

            # for image, mask, pred_mask, filename in zip(images, masks, pred_masks_bin, filenames):
            #     image = image.cpu().numpy()
            #     mask = mask.cpu().numpy()
            #     pred_mask = pred_mask.cpu().numpy()
            #     image = np.transpose(image, (1, 2, 0))
            #     image = (image - image.min()) / (image.max() - image.min())
            #     filename = os.path.join(args.save_path, filename)
            #     # print(filename)
            #     save_prediction(image, mask, pred_mask, filename)

    average_score = total_score / len(test_loader)
    print(f'Average Dice Score: {average_score}')
