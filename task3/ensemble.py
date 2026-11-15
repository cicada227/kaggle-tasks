import os
import pandas as pd

path_20 = None
path_19 = None

for root, dirs, files in os.walk('/kaggle/input'):
    if 'submission_yolo_cv.csv' in files:
        path_20 = os.path.join(root, 'submission_yolo_cv.csv')
    if 'submission_pose.csv' in files:
        path_19 = os.path.join(root, 'submission_pose.csv')

if path_20 and path_19:
    print(f"找到 20 分文件路径: {path_20}")
    print(f"找到 19.6 分文件路径: {path_19}")

    sub_20 = pd.read_csv(path_20)
    sub_19 = pd.read_csv(path_19)
    
    sub_ens = sub_20.copy()
    sub_ens['value'] = (sub_20['value'] + sub_19['value']) / 2.0
    sub_ens.to_csv('/kaggle/working/submission_14_dream.csv', index=False)
    print("提交")
else:
    print("没找到文件。")
    if not path_20: print("缺失: submission_yolo_cv.csv")
    if not path_19: print("缺失: submission_pose.csv")
