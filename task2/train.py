import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import math

BASE_DIR = '/kaggle/input/competitions/2026task2/fer_data/fer_data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

LABEL_MAP = {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}
CLASSES = list(LABEL_MAP.keys())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 112      
BATCH_SIZE = 96    
EPOCHS = 18
FOLDS = 5
LR = 5e-4         


class FERDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, is_test=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, os.path.basename(self.images[idx])
        return image, self.labels[idx]


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.TrivialAugmentWide(), # 替换掉手动增强，自动寻找最佳增强组合
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)) 
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


all_train_images, all_train_labels = [], []
for cls in CLASSES:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    if not os.path.exists(cls_dir): continue
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_train_images.append(os.path.join(cls_dir, img_name))
            all_train_labels.append(LABEL_MAP[cls])

test_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
test_dataset = FERDataset(test_images, is_test=True, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels), y=all_train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"类别权重: {class_weights}")


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)


    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_images), 6))


criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

for fold, (train_idx, val_idx) in enumerate(skf.split(all_train_images, all_train_labels)):
    print(f"\n🚀 ========== Fold {fold+1}/{FOLDS} ==========")
    
    train_dataset = FERDataset([all_train_images[i] for i in train_idx], [all_train_labels[i] for i in train_idx], transform=train_transform)
    val_dataset = FERDataset([all_train_images[i] for i in val_idx], [all_train_labels[i] for i in val_idx], transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
  
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    def warmup_cosine(epoch):
        if epoch < 2: return (epoch + 1) / 2
        return 0.5 * (1 + math.cos(math.pi * (epoch - 2) / (EPOCHS - 2)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 手动实现 CutMix
            if np.random.rand() < 0.3: 
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(images.size()[0]).to(device)
                target_a, target_b = labels, labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {100*correct/total:.2f}% | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')

    model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
    model.eval()
    
    fold_test_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            out1 = torch.softmax(model(images), dim=1)
            out2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            images_small = torch.nn.functional.interpolate(images, size=(96, 96), mode='bilinear', align_corners=False)
            out3 = torch.softmax(model(images_small), dim=1)
            
            out = (out1 + out2 + out3) / 3.0
            fold_test_preds.append(out.cpu().numpy())
            
    test_preds += np.concatenate(fold_test_preds, axis=0) / FOLDS


final_predictions = np.argmax(test_preds, axis=1)
submission = pd.DataFrame({'ID': [os.path.basename(f) for f in test_images], 'Emotion': final_predictions})
submission.to_csv('/kaggle/working/submission.csv', index=False)
