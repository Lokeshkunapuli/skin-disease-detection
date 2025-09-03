"""
@author: denil gabani

"""

import cv2
import numpy as np
from inference import Network
from knn_infer import knn_predict_image, knn_score_vector_bgr
from explain import occlusion_sensitivity, overlay_heatmap
import os
from clinical_severity import calculate_clinical_severity

# CPU extension is obsolete in modern OpenVINO versions
CPU_EXTENSION = None

# Path of converted skin disease model in xml
MODEL = "model/model_tf.xml"

SKIN_CLASSES = {
    0: 'akiec, Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'bcc, Basal Cell Carcinoma',
    2: 'bkl, Benign Keratosis',
    3: 'df, Dermatofibroma',
    4: 'mel, Melanoma',
    5: 'nv, Melanocytic Nevi',
    6: 'vasc, Vascular skin lesion'
}

def preprocessing(input_image, height, width):
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, height, width)
    return image

def pred_at_edge(input_img):
    try:
        plugin = Network()
        plugin.load_model(MODEL, "CPU", CPU_EXTENSION)
        net_input_shape = plugin.get_input_shape()
        img = cv2.imread(input_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image not found or cannot be read.")
        # Ensure image has 3 channels
        if len(img.shape) == 2 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Input shape is typically [N, C, H, W]
        final_img = preprocessing(img, net_input_shape[2], net_input_shape[3])
        plugin.async_inference(final_img)
        if plugin.wait() == 0:
            results = plugin.extract_output()
            if results is None:
                raise ValueError("No results returned from inference.")
            # Flatten model outputs and compute softmax to get 0-1 probabilities
            logits = results.flatten()
            # Numerical stability for softmax
            max_logit = np.max(logits)
            exp_shifted = np.exp(logits - max_logit)
            probs = exp_shifted / (np.sum(exp_shifted) + 1e-12)
            pred = int(np.argmax(probs))
            disease = SKIN_CLASSES.get(pred, "Unknown")
            accuracy = float(probs[pred])
            print(f"Disease: {disease}, Accuracy (softmax): {accuracy}")
            # Build a score function over original image resolution for occlusion
            def score_fn(img_bgr):
                final = preprocessing(img_bgr, net_input_shape[2], net_input_shape[3])
                plugin.async_inference(final)
                if plugin.wait() == 0:
                    out = plugin.extract_output()
                    if out is None:
                        return None
                    return out.flatten()
                return None
            heatmap = occlusion_sensitivity(img, score_fn, pred, patch_size=32, stride=24)
            heatmap_path = None
            if heatmap is not None:
                overlay = overlay_heatmap(img, heatmap, alpha=0.45)
                base, ext = os.path.splitext(os.path.basename(input_img))
                heatmap_filename = f"{base}_heatmap.png"
                heatmap_full = os.path.join("static", "data", heatmap_filename)
                cv2.imwrite(heatmap_full, overlay)
                heatmap_path = heatmap_filename
            # Use clinical severity assessment
            level, sev_score, advice = calculate_clinical_severity(img)
            return disease, accuracy, heatmap_path, 'OpenVINO', level, sev_score, advice
        else:
            raise RuntimeError("Inference failed.")
    except Exception as e:
        # Fallback to KNN if OpenVINO model is incompatible or missing
        try:
            # Build KNN score function for occlusion heatmap
            img = cv2.imread(input_img, cv2.IMREAD_COLOR)
            def score_fn_knn(bgr):
                scores, _ = knn_score_vector_bgr(bgr)
                return scores
            # choose target index by max score
            scores_full, classes = knn_score_vector_bgr(img)
            # Apply softmax over scores to get 0-1 probabilities
            max_s = float(np.max(scores_full))
            exp_scores = np.exp(scores_full - max_s)
            probs_knn = exp_scores / (np.sum(exp_scores) + 1e-12)
            target = int(np.argmax(probs_knn))
            heatmap = occlusion_sensitivity(img, score_fn_knn, target, patch_size=32, stride=24)
            heatmap_path = None
            if heatmap is not None:
                overlay = overlay_heatmap(img, heatmap, alpha=0.45)
                base, ext = os.path.splitext(os.path.basename(input_img))
                heatmap_filename = f"{base}_heatmap.png"
                heatmap_full = os.path.join("static", "data", heatmap_filename)
                cv2.imwrite(heatmap_full, overlay)
                heatmap_path = heatmap_filename
            # Use clinical severity assessment
            level, sev_score, advice = calculate_clinical_severity(img)
            label = classes[target] if 0 <= target < len(classes) else 'Unknown'
            accuracy = float(probs_knn[target])
            print(f"Disease: KNN: {label}, Accuracy (softmax): {accuracy}")
            return f"KNN: {label}", accuracy, heatmap_path, 'KNN', level, sev_score, advice
        except Exception as knn_e:
            print("Error during prediction:", str(e))
            print("KNN fallback failed:", str(knn_e))
            return "Error: " + str(e), 0.0, None, None, None, 0.0, ""

