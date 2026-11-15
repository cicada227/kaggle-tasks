import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
from tqdm import tqdm

BASE_DIR = '/kaggle/input/competitions/2026task2/fer_data/fer_data'
TEST_DIR = os.path.join(BASE_DIR, 'test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 112
BATCH_SIZE = 128


class FERDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(self.images[idx])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
test_dataset = FERDataset(test_images, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


model = models.efficientnet_b0(weights=None)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
model = model.to(device)


total_preds = np.zeros((len(test_images), 6))
models_used = 0

for fold in range(4): # 0, 1, 2, 3
    model_path = f'/kaggle/working/best_model_fold_{fold}.pth'
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到 {model_path}，跳过")
        continue
        
    print(f"🚀 正在加载模型 Fold {fold} 并进行 TTA 预测...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models_used += 1
    
    fold_preds = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)
            out1 = torch.softmax(model(images), dim=1)
            out2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            images_small = torch.nn.functional.interpolate(images, size=(96, 96), mode='bilinear', align_corners=False)
            out3 = torch.softmax(model(images_small), dim=1)
            
            out = (out1 + out2 + out3) / 3.0
            fold_preds.append(out.cpu().numpy())
            
    total_preds += np.concatenate(fold_preds, axis=0)

if models_used > 0:
    total_preds /= models_used 
    final_predictions = np.argmax(total_preds, axis=1)
    submission = pd.DataFrame({
        'ID': [os.path.basename(f) for f in test_images],
        'Emotion': final_predictions
    })
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print(submission.head())
else:
    print("❌ 没有找到可用的模型文件。")
