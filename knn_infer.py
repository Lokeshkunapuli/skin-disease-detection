import os
import json
import numpy as np
import cv2

MODEL_DIR = 'model_knn'

def _load_artifacts():
    x_path = os.path.join(MODEL_DIR, 'X.npy')
    y_path = os.path.join(MODEL_DIR, 'Y.npy')
    classes_path = os.path.join(MODEL_DIR, 'classes.json')
    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(classes_path)):
        raise FileNotFoundError('KNN artifacts not found. Please run train_knn.py')
    X = np.load(x_path)
    Y = np.load(y_path)
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    return X, Y, classes

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

def knn_predict_image(image_path, k=3):
    X, Y, classes = _load_artifacts()
    feat = extract_features(image_path)
    if feat is None:
        raise ValueError('Failed to read image for KNN fallback')
    dists = np.linalg.norm(X - feat, axis=1)
    idx = np.argsort(dists)[:k]
    votes = Y[idx]
    vals, counts = np.unique(votes, return_counts=True)
    winner = int(vals[np.argmax(counts)])
    topk = dists[idx]
    conf = float(1.0 / (np.mean(topk) + 1e-6))
    label = classes[winner] if 0 <= winner < len(classes) else 'Unknown'
    return label, conf

def knn_score_vector_bgr(image_bgr, k=5):
    """Return per-class pseudo-scores using inverse-distance of top-k neighbors."""
    X, Y, classes = _load_artifacts()
    img = cv2.resize(image_bgr, (128, 128))
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hist_mag = cv2.calcHist([mag.astype(np.uint8)], [0], None, [32], [0, 256]).flatten()
    feat = np.concatenate([hist_b, hist_g, hist_r, hist_mag]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-6)
    dists = np.linalg.norm(X - feat, axis=1)
    idx = np.argsort(dists)[:k]
    topk = dists[idx]
    votes = Y[idx]
    # inverse distance weights
    weights = 1.0 / (topk + 1e-6)
    num_classes = int(np.max(Y)) + 1
    scores = np.zeros(num_classes, dtype=np.float32)
    for cls, w in zip(votes, weights):
        scores[int(cls)] += float(w)
    # normalize to sum 1
    ssum = float(np.sum(scores)) + 1e-6
    scores = scores / ssum
    return scores, classes


