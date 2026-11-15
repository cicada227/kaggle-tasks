!pip install ultralytics -q

import os
import xml.etree.ElementTree as ET
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.model_selection import KFold

DATA_DIR = '/kaggle/input/competitions/2026task3/detection'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')
XML_DIR = os.path.join(DATA_DIR, 'train_location')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')

DATASET_DIR = '/kaggle/working/yolo_dataset'
os.makedirs(f'{DATASET_DIR}/images/train', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/labels/train', exist_ok=True)

print(">>> 正在将 XML 转换为 YOLO 数据集...")
file_map = {f.split('.')[0].lstrip('0'): f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.jpg', '.png'))}
xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]

valid_data = []

for xml_file in xml_files:
    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    root = tree.getroot()
    clean_id = xml_file.split('.')[0].lstrip('0')
    real_filename = file_map.get(clean_id)
    if not real_filename: continue

    src_img = os.path.join(TRAIN_IMG_DIR, real_filename)
    dest_img = os.path.join(DATASET_DIR, 'images/train', real_filename)
    shutil.copy(src_img, dest_img)

    size = root.find('size')
    w = float(size.find('width').text) if size is not None and size.find('width') is not None else 2992.0
    h = float(size.find('height').text) if size is not None and size.find('height') is not None else 2000.0

    bndbox = root.find('.//bndbox')
    if bndbox is not None:
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        label_str = f"0 {x_center} {y_center} {box_w} {box_h}"
        txt_name = real_filename.rsplit('.', 1)[0] + '.txt'
        with open(os.path.join(DATASET_DIR, 'labels/train', txt_name), 'w') as f:
            f.write(label_str)
        
        valid_data.append((dest_img, os.path.join(DATASET_DIR, 'labels/train', txt_name)))


print(">>> 开始 3 折交叉验证训练 YOLOv8s (预计 4-6 分钟)...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)

all_fold_predictions = []

test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpg')]

for fold, (train_idx, val_idx) in enumerate(kf.split(valid_data)):
    print(f"\n--- 正在训练 Fold {fold+1}/3 ---")
    
    train_imgs = [valid_data[i][0] for i in train_idx]
    train_labels = [valid_data[i][1] for i in val_idx] # val 数据也可以作为验证集

    with open(f'{DATASET_DIR}/dataset_fold_{fold}.yaml', 'w') as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n") # 因为数据太少，验证集就用训练集的子集
        f.write("names:\n  0: fovea\n")


    model = YOLO('yolov8s.pt') 
    model.train(
        data=f'{DATASET_DIR}/dataset_fold_{fold}.yaml', 
        epochs=40, 
        imgsz=1024, 
        batch=4, 
        device=0, 
        verbose=False,
        project='yolo_cv',
        name=f'fold_{fold}'
    )
    
    print(f"Fold {fold+1} 训练完成，开始预测测试集...")
    # 预测当前 Fold 的测试集结果
    fold_preds = {}
    for img_name in test_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        # augment=True 开启 TTA（测试时增强）
        results = model(img_path, imgsz=1024, conf=0.05, augment=True, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            best_box = boxes[0]
            cx = best_box.xywhn[0][0].item()
            cy = best_box.xywhn[0][1].item()
            fold_preds[img_name] = (cx, cy)
        else:
            fold_preds[img_name] = (0.483, 0.527) 
            
    all_fold_predictions.append(fold_preds)


print("\n>>> 3 个 Fold 预测完成，正在取中位数...")
final_predictions = {}

for img_name in test_images:
    # 获取 3 个模型对该图的预测
    preds = [all_fold_predictions[i][img_name] for i in range(3)]
    preds = np.array(preds)
    
    # 沿着模型维度取中位数 (0维度)
    median_pred = np.median(preds, axis=0)
    
    orig_w, orig_h = Image.open(os.path.join(TEST_IMG_DIR, img_name)).size
    pred_x = median_pred[0] * orig_w
    pred_y = median_pred[1] * orig_h
    
    base_name = img_name.split('.')[0]
    final_predictions[base_name] = (pred_x, pred_y)
    final_predictions[base_name.lstrip('0')] = (pred_x, pred_y)

sub_df = pd.read_csv(SAMPLE_SUB)
for idx, row in sub_df.iterrows():
    parts = str(row['ImageID']).split('_')
    img_id = parts[0].lstrip('0')
    coord_type = parts[-1]
    if img_id in final_predictions:
        sub_df.at[idx, 'value'] = final_predictions[img_id][0] if coord_type == 'X' else final_predictions[img_id][1]

sub_df.to_csv('submission_yolo_cv.csv', index=False)
