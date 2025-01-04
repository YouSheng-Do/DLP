import torch
from utils import dice_score

def evaluate(model, args, val_dataloader, device):
    # implement the evaluation function here
    model.eval()
    val_score = 0.0
    with torch.no_grad():
        for sample in val_dataloader:
            images, masks = sample['image'].to(device), sample['mask'].to(device)
            pred_masks = model(images)
            pred_masks = torch.sigmoid(pred_masks.squeeze(1))
            pred_masks_bin = (pred_masks > 0.5).float()
            score = dice_score(pred_mask=pred_masks_bin, gt_mask=masks.float())

            val_score += score.item()
    
    val_score /= len(val_dataloader)

    return val_score