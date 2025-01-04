from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision

# # 創建一個示例圖像（假設是一個隨機的 numpy 數組）
# example_image = np.random.rand(224, 224, 3).astype(np.float32)

# # 將 numpy 數組轉換為 PIL 圖像
# example_image_pil = Image.fromarray((example_image * 255).astype(np.uint8))

# # 定義轉換
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# # 應用轉換
# transformed_image = transform(example_image_pil)

# # 檢查轉換後的維度
# print(f"原始圖像維度 (H, W, C): {example_image.shape}")
# print(f"轉換後的維度 (C, H, W): {transformed_image.shape}")

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
print(input)
print(target)
print(output)
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()
print(input)
print(target)
print(output)