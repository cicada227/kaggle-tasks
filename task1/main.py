# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, os, glob
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold

tf.keras.mixed_precision.set_global_policy('mixed_float16')
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
np.random.seed(42); tf.random.set_seed(42)

ROOT = '/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2'
train_dir, test_dir = f'{ROOT}/train', f'{ROOT}/test'
sample_path = glob.glob('/kaggle/input/**/submission*.csv', recursive=True)[0]

class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
NUM_CLASSES = len(class_names)
print('classes:', class_names, flush=True)

IMG_SIZE = (300, 300)
BATCH = 16
N_FOLDS = 5
EPOCHS2 = 50

X_all, y_all = [], []
for ci, c in enumerate(class_names):
    for f in sorted(os.listdir(os.path.join(train_dir, c))):
        img = image.load_img(os.path.join(train_dir, c, f), target_size=IMG_SIZE)
        X_all.append(image.img_to_array(img)); y_all.append(ci)
X_all = np.array(X_all, np.float32)
y_all = np.array(y_all)
y_cat = tf.keras.utils.to_categorical(y_all, NUM_CLASSES)
print('train:', X_all.shape, flush=True)

test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
X_test = np.array([image.img_to_array(image.load_img(os.path.join(test_dir, f), target_size=IMG_SIZE))
                   for f in test_files], np.float32)

def tta10(arr):
    h = arr.shape[0]; c = int(h * 0.9); o = (h - c) // 2
    return np.array([
        arr, np.fliplr(arr), np.flipud(arr),
        np.rot90(arr, 1).copy(), np.rot90(arr, 3).copy(),
        np.clip(arr * 1.1, 0, 255), np.clip(arr * 0.9, 0, 255),
        tf.image.resize(arr[o:o+c, o:o+c], (h, h)).numpy(),
        tf.image.resize(np.fliplr(arr)[o:o+c, o:o+c], (h, h)).numpy(),
        np.rot90(np.fliplr(arr), 1).copy() # 第10路
    ])


oof_test_probs = np.zeros((len(test_files), NUM_CLASSES), np.float64)
val_accs = []
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"\n================ FOLD {fold+1}/{N_FOLDS} ================", flush=True)

    if os.path.exists(f'best_fold_{fold}.keras'):
        print(f"Fold {fold} 模型已存在，跳过训练，直接加载...", flush=True)
        best_model = models.load_model(f'best_fold_{fold}.keras')
    else:
        x_tr, x_va = X_all[train_idx], X_all[val_idx]
        y_tr, y_va = y_cat[train_idx], y_cat[val_idx]

        aug = ImageDataGenerator(
            rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
            horizontal_flip=True, vertical_flip=True, zoom_range=0.15,
            brightness_range=(0.8, 1.2), fill_mode='nearest')
        gen_tr = aug.flow(x_tr, y_tr, batch_size=BATCH, shuffle=True)
        gen_va = ImageDataGenerator().flow(x_va, y_va, batch_size=BATCH, shuffle=False)

        inp = layers.Input(shape=(*IMG_SIZE, 3))
        base = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=inp)
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='swish')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
        model = models.Model(inp, out)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        model.compile(optimizers.AdamW(1e-3, weight_decay=1e-4), loss, metrics=['accuracy'])
        model.fit(gen_tr, validation_data=gen_va, epochs=6, verbose=1,
                  callbacks=[EarlyStopping('val_accuracy', patience=3, restore_best_weights=True)])

        base.trainable = True
        for layer in base.layers[:-50]:
            layer.trainable = False
        steps = EPOCHS2 * max(1, len(gen_tr))
        lr_sched = optimizers.schedules.CosineDecay(1e-5, decay_steps=steps, alpha=0.01)
        model.compile(optimizers.AdamW(lr_sched, weight_decay=1e-4), loss, metrics=['accuracy'])
        model.fit(gen_tr, validation_data=gen_va, epochs=EPOCHS2, verbose=1,
                  callbacks=[EarlyStopping('val_accuracy', patience=8, restore_best_weights=True),
                             ModelCheckpoint(f'best_fold_{fold}.keras', 'val_accuracy', save_best_only=True)])
        best_model = models.load_model(f'best_fold_{fold}.keras')

 
    gen_va_eval = ImageDataGenerator().flow(X_all[val_idx], y_cat[val_idx], batch_size=BATCH, shuffle=False)
    va_loss, va_acc = best_model.evaluate(gen_va_eval, verbose=0)
    val_accs.append(va_acc)
    print(f'Fold {fold} val_acc = {va_acc:.4f}', flush=True)

    fold_probs = np.zeros((len(test_files), NUM_CLASSES), np.float64)
    for i in range(0, len(test_files), BATCH):
        batch_arr = X_test[i:i+BATCH]
        batch_tta = np.stack([tta10(x) for x in batch_arr]).reshape(-1, *IMG_SIZE, 3)
        p = best_model.predict(batch_tta, batch_size=32, verbose=0).reshape(len(batch_arr), 10, -1)
        fold_probs[i:i+len(batch_arr)] = p.mean(axis=1)
    
    oof_test_probs += fold_probs / N_FOLDS
    del model, best_model, gen_va_eval
    tf.keras.backend.clear_session()

print(f'\n5折验证准确率: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}', flush=True)
print('各折:', [f'{a:.4f}' for a in val_accs], flush=True)

pred_labels = [class_names[i] for i in np.argmax(oof_test_probs, axis=1)]
sample = pd.read_csv(sample_path)
id_col = sample.columns[0]
m_exact = {os.path.basename(str(s)): l for s, l in zip(test_files, pred_labels)}
m_stem  = {os.path.splitext(os.path.basename(str(s)))[0]: l for s, l in zip(test_files, pred_labels)}
vals = []
for s in sample[id_col]:
    v = m_exact.get(os.path.basename(str(s)))
    if v is None:
        v = m_stem.get(os.path.splitext(os.path.basename(str(s)))[0])
    vals.append(v)
sample['Category'] = vals
assert sample['Category'].notna().all(), 'ID mismatch!'
sample.to_csv('submission.csv', index=False)
print( flush=True)
