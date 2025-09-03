import os
import json
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2

MODEL_DIR = 'model_knn'

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hist_mag = cv2.calcHist([mag.astype(np.uint8)], [0], None, [32], [0, 256]).flatten()
    feat = np.concatenate([hist_b, hist_g, hist_r, hist_mag]).astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-6
    feat /= norm
    return feat

# Load artifacts
X = np.load(os.path.join(MODEL_DIR, 'X.npy'))
Y = np.load(os.path.join(MODEL_DIR, 'Y.npy'))
with open(os.path.join(MODEL_DIR, 'classes.json'), 'r', encoding='utf-8') as f:
    CLASS_NAMES = json.load(f)


def knn_predict(feat, k=3):
    # Compute distances
    dists = np.linalg.norm(X - feat, axis=1)
    idx = np.argsort(dists)[:k]
    votes = Y[idx]
    # majority vote
    vals, counts = np.unique(votes, return_counts=True)
    winner = vals[np.argmax(counts)]
    # confidence: average inverse distance of top-k
    topk = dists[idx]
    conf = float(1.0 / (np.mean(topk) + 1e-6))
    return int(winner), conf


def select_file():
    path = filedialog.askopenfilename(title='Select an image', filetypes=[('Images', '*.jpg;*.jpeg;*.png;*.bmp')])
    if not path:
        return
    feat = extract_features(path)
    if feat is None:
        print('Failed to read image')
        return
    pred, conf = knn_predict(feat)
    print('Prediction:', CLASS_NAMES[pred], 'Confidence:', conf)


root = tk.Tk()
root.title('Skin Disease Classifier')
root.geometry('400x120')

btn = tk.Button(root, text='Select Image', command=select_file)
btn.pack(pady=20)

root.mainloop()
