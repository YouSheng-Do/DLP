import torch
import pandas as pd
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask, epsilon: float = 1e-6):
    # implement the Dice score here
    
    pred_mask = pred_mask.contiguous().view(-1)
    gt_mask = gt_mask.contiguous().view(-1)

    intersection = (pred_mask * gt_mask).sum()
    dice_score = (2. * intersection + epsilon) / (pred_mask.sum() + gt_mask.sum() + epsilon)
    
    return dice_score

def save_prediction(image, mask, pred_mask, filename):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('True Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    # plt.show()

    plt.savefig(f'{filename}.png')

    plt.close()

def plot_dice_score():
    df_unet = pd.read_csv("unet_dice_scores.csv")
    df_resnet34_unet = pd.read_csv("resnet34_unet_dice_scores.csv")

    plt.plot(df_unet["Train Dice Score"], label='UNet_train_score')
    plt.plot(df_unet["Val Dice Score"], label='UNet_val_score')       
    plt.plot(df_resnet34_unet["Train Dice Score"], label='ResNet34_UNet_train_scpre')
    plt.plot(df_resnet34_unet["Val Dice Score"], label='ResNet34_UNet_val_score')

    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Dice Score Curve')
    plt.savefig('dice_score_comparision.png')
    plt.show()

if __name__ == "__main__":
    plot_dice_score()