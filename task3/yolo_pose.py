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

DATASET_DIR = '/kaggle/working/yolo_pose_dataset'
os.makedirs(f'{DATASET_DIR}/images/train', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/labels/train', exist_ok=True)

print(">>> 正在解析 XML 并转换为 Pose 关键点格式...")
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
    w = float(size.find('width').text) if size is not None else 2992.0
    h = float(size.find('height').text) if size is not None else 2000.0

    bndbox = root.find('.//bndbox')
    if bndbox is not None:
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        label_str = f"0 {cx} {cy} {bw} {bh} {cx} {cy} 2"
        txt_name = real_filename.rsplit('.', 1)[0] + '.txt'
        with open(os.path.join(DATASET_DIR, 'labels/train', txt_name), 'w') as f:
            f.write(label_str)
        
        valid_data.append((dest_img, txt_name))


with open(f'{DATASET_DIR}/dataset.yaml', 'w') as f:
    f.write(f"path: {DATASET_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/train\n") 
    f.write("names:\n  0: fovea\n")
    f.write("kpt_shape: [1, 3]\n") # 1个关键点，3个值 (x, y, visibility)


kf = KFold(n_splits=5, shuffle=True, random_state=2024)
test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpg')]
all_fold_predictions = []

for fold, (train_idx, val_idx) in enumerate(kf.split(valid_data)):
    print(f"\n--- 正在训练 Fold {fold+1}/5 ---")

    model = YOLO('yolov8s-pose.pt') 
    model.train(
        data=f'{DATASET_DIR}/dataset.yaml', 
        epochs=60,         
        imgsz=1280,         
        batch=4,            
        device=0, 
        fliplr=0.5,       
        flipud=0.0,      
        hsv_v=0.4,         
        mosaic=1.0,         
        close_mosaic=10,    
        verbose=False,
        project='yolo_pose_runs',
        name=f'fold_{fold}'
    )

    fold_preds = {}
    for img_name in test_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        results = model(img_path, imgsz=1280, conf=0.05, augment=True, verbose=False)
        
        if len(results[0].keypoints) > 0 and len(results[0].keypoints.xyn) > 0:
            kpts = results[0].keypoints.xyn[0]
      
            fold_preds[img_name] = (kpts[0][0].item(), kpts[0][1].item())
        else:
          
            fold_preds[img_name] = (0.483, 0.527) 
            
    all_fold_predictions.append(fold_preds)

print("\n>>> 5 折预测完成，正在提取中位数...")
final_predictions = {}
for img_name in test_images:
    preds = [all_fold_predictions[i][img_name] for i in range(5)]
    median_pred = np.median(np.array(preds), axis=0)
    
    orig_w, orig_h = Image.open(os.path.join(TEST_IMG_DIR, img_name)).size
    base_name = img_name.split('.')[0]
    final_predictions[base_name] = (median_pred[0] * orig_w, median_pred[1] * orig_h)
    final_predictions[base_name.lstrip('0')] = (median_pred[0] * orig_w, median_pred[1] * orig_h)

sub_df = pd.read_csv(SAMPLE_SUB)
for idx, row in sub_df.iterrows():
    parts = str(row['ImageID']).split('_')
    img_id = parts[0].lstrip('0')
    coord_type = parts[-1]
    if img_id in final_predictions:
        sub_df.at[idx, 'value'] = final_predictions[img_id][0] if coord_type == 'X' else final_predictions[img_id][1]

sub_df.to_csv('submission_pose.csv', index=False)
