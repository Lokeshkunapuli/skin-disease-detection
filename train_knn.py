# Simple KNN trainer using OpenCV + NumPy only
# Trains on folder structure under skin_data/<class_name>/*.jpg

import os
import cv2
import numpy as np
import json

DATA_DIR = 'skin_data'
MODEL_DIR = 'model_knn'
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature extractor: resize to 128x128, compute color hist (BGR) + HOG-like gradients

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    # Color histograms per channel
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
    # Simple gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Histogram of magnitudes
    hist_mag = cv2.calcHist([mag.astype(np.uint8)], [0], None, [32], [0, 256]).flatten()
    feat = np.concatenate([hist_b, hist_g, hist_r, hist_mag]).astype(np.float32)
    # Normalize
    norm = np.linalg.norm(feat) + 1e-6
    feat /= norm
    return feat

# Load dataset
class_names = []
features = []
labels = []

for cls in sorted(os.listdir(DATA_DIR)):
    cls_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    class_index = len(class_names)
    class_names.append(cls)
    for fname in os.listdir(cls_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        fpath = os.path.join(cls_path, fname)
        feat = extract_features(fpath)
        if feat is None:
            continue
        features.append(feat)
        labels.append(class_index)

if not features:
    raise RuntimeError('No training images found in skin_data/*/')

X = np.vstack(features)
Y = np.array(labels, dtype=np.int32)

# Save artifacts
np.save(os.path.join(MODEL_DIR, 'X.npy'), X)
np.save(os.path.join(MODEL_DIR, 'Y.npy'), Y)
with open(os.path.join(MODEL_DIR, 'classes.json'), 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print('Training data prepared:', X.shape, 'classes:', len(class_names))
