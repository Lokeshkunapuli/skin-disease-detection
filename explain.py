import numpy as np
import cv2

def occlusion_sensitivity(image_bgr, score_fn, target_index, patch_size=32, stride=16):
    h, w = image_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Baseline score on original image
    base_scores = score_fn(image_bgr)
    if base_scores is None:
        return None
    base = float(base_scores[target_index]) if target_index < len(base_scores) else 0.0
    if base <= 0:
        base = 1e-6

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, y2 = y, min(y + patch_size, h)
            x1, x2 = x, min(x + patch_size, w)
            occluded = image_bgr.copy()
            occluded[y1:y2, x1:x2] = 0
            scores = score_fn(occluded)
            if scores is None:
                continue
            s = float(scores[target_index]) if target_index < len(scores) else 0.0
            drop = max(0.0, base - s)
            heatmap[y1:y2, x1:x2] += drop

    # Normalize 0-1
    if np.max(heatmap) > 0:
        heatmap = heatmap / (np.max(heatmap) + 1e-6)

    return heatmap

def overlay_heatmap(image_bgr, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    hm_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    overlay = cv2.addWeighted(hm_color, alpha, image_bgr, 1 - alpha, 0)
    return overlay


