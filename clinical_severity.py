import cv2
import numpy as np

def analyze_lesion_characteristics(image_bgr):
    """Extract clinical features that dermatologists use for severity assessment."""
    img = cv2.resize(image_bgr, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Lesion size and area
    # Use Otsu thresholding to segment lesion
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get largest contour (main lesion)
    main_contour = max(contours, key=cv2.contourArea)
    lesion_area = cv2.contourArea(main_contour)
    total_area = img.shape[0] * img.shape[1]
    area_ratio = lesion_area / total_area
    
    # 2. Border irregularity (perimeter/area ratio)
    perimeter = cv2.arcLength(main_contour, True)
    if lesion_area > 0:
        irregularity = (perimeter * perimeter) / (4 * np.pi * lesion_area)
    else:
        irregularity = 0
    
    # 3. Color variation (heterogeneity)
    # Extract lesion region
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [main_contour], 255)
    lesion_pixels = img[mask > 0]
    
    if len(lesion_pixels) > 0:
        color_std = np.std(lesion_pixels, axis=0)
        color_heterogeneity = np.mean(color_std) / 255.0
    else:
        color_heterogeneity = 0
    
    # 4. Asymmetry (shape analysis)
    # Fit ellipse and compare with contour
    if len(main_contour) >= 5:
        ellipse = cv2.fitEllipse(main_contour)
        ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
        asymmetry = abs(lesion_area - ellipse_area) / max(lesion_area, ellipse_area)
    else:
        asymmetry = 0
    
    # 5. Edge sharpness (border definition)
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    edge_density = np.sum(edges > 0) / total_area
    
    # 6. Texture complexity (using Local Binary Pattern approximation)
    # Calculate local variance as texture measure
    kernel = np.ones((3, 3), np.float32) / 9
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
    texture_complexity = np.mean(local_variance) / (255 * 255)
    
    return {
        'area_ratio': area_ratio,
        'irregularity': irregularity,
        'color_heterogeneity': color_heterogeneity,
        'asymmetry': asymmetry,
        'edge_density': edge_density,
        'texture_complexity': texture_complexity
    }

def calculate_clinical_severity(image_bgr):
    """Calculate severity based on clinical features."""
    features = analyze_lesion_characteristics(image_bgr)
    if features is None:
        return "Low", 0.2, "Unable to analyze lesion characteristics"
    
    # Clinical scoring system (based on dermatology guidelines)
    score = 0.0
    
    # Size factor (larger lesions are more concerning)
    if features['area_ratio'] > 0.15:  # >15% of image
        score += 0.25
    elif features['area_ratio'] > 0.08:  # 8-15%
        score += 0.15
    elif features['area_ratio'] > 0.03:  # 3-8%
        score += 0.05
    
    # Border irregularity (ABCD rule)
    if features['irregularity'] > 2.5:  # Very irregular
        score += 0.20
    elif features['irregularity'] > 1.8:  # Moderately irregular
        score += 0.10
    
    # Color heterogeneity (multiple colors = concerning)
    if features['color_heterogeneity'] > 0.3:  # High variation
        score += 0.20
    elif features['color_heterogeneity'] > 0.15:  # Moderate variation
        score += 0.10
    
    # Asymmetry (asymmetric lesions are concerning)
    if features['asymmetry'] > 0.4:  # Very asymmetric
        score += 0.15
    elif features['asymmetry'] > 0.2:  # Moderately asymmetric
        score += 0.08
    
    # Edge definition (well-defined borders can be concerning)
    if features['edge_density'] > 0.1:  # Sharp borders
        score += 0.10
    elif features['edge_density'] > 0.05:  # Moderate borders
        score += 0.05
    
    # Texture complexity (complex patterns are concerning)
    if features['texture_complexity'] > 0.02:  # High complexity
        score += 0.10
    elif features['texture_complexity'] > 0.01:  # Moderate complexity
        score += 0.05
    
    # Normalize to 0-1
    severity_score = min(1.0, score)
    
    # Determine severity level
    if severity_score >= 0.7:
        level = "Very High"
        advice = "Multiple concerning features detected. Urgent dermatologist evaluation recommended."
    elif severity_score >= 0.5:
        level = "High"
        advice = "Several concerning features present. Schedule dermatology consultation."
    elif severity_score >= 0.3:
        level = "Moderate"
        advice = "Some features require monitoring. Consider dermatology follow-up."
    else:
        level = "Low"
        advice = "Lesion appears benign. Continue regular skin monitoring."
    
    return level, severity_score, advice
