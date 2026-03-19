import io
import math
from typing import Dict, Any

import numpy as np
from PIL import Image, ImageStat, ImageFilter


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def estimate_blur_score(image: Image.Image) -> float:
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # تقريب Laplacian بدون OpenCV
    kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)

    h, w = arr.shape
    if h < 3 or w < 3:
        return 0.0

    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for i in range(h - 2):
        for j in range(w - 2):
            patch = arr[i:i+3, j:j+3]
            out[i, j] = np.sum(patch * kernel)

    var_lap = float(np.var(out))
    return var_lap


def estimate_brightness_score(image: Image.Image) -> float:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean_val = stat.mean[0]  # 0..255

    # الأفضل في الوسط تقريبًا
    # 128 ممتازة، قرب 0 أو 255 أسوأ
    score = max(0.0, 100.0 - (abs(mean_val - 128.0) / 128.0) * 100.0)
    return float(score)


def estimate_contrast_score(image: Image.Image) -> float:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    std_val = stat.stddev[0]  # تباين تقريبي

    # نطبعها بين 0..100
    score = min(100.0, std_val * 2.5)
    return float(score)


def estimate_resolution_score(image: Image.Image) -> float:
    w, h = image.size
    pixels = w * h

    if pixels >= 1200000:
        return 100.0
    if pixels >= 800000:
        return 90.0
    if pixels >= 500000:
        return 75.0
    if pixels >= 250000:
        return 55.0
    return 30.0


def assess_image_quality(image: Image.Image) -> Dict[str, Any]:
    blur_raw = estimate_blur_score(image)
    brightness = estimate_brightness_score(image)
    contrast = estimate_contrast_score(image)
    resolution = estimate_resolution_score(image)

    # تحويل blur إلى 0..100 تقريبًا
    if blur_raw >= 250:
        blur_score = 100.0
    elif blur_raw >= 150:
        blur_score = 85.0
    elif blur_raw >= 80:
        blur_score = 65.0
    elif blur_raw >= 40:
        blur_score = 45.0
    else:
        blur_score = 20.0

    overall = (
        0.40 * blur_score +
        0.20 * brightness +
        0.20 * contrast +
        0.20 * resolution
    )

    issues = []
    if blur_score < 45:
        issues.append("الصورة غير واضحة أو مهزوزة")
    if brightness < 35:
        issues.append("الإضاءة غير مناسبة")
    if contrast < 25:
        issues.append("التباين ضعيف")
    if resolution < 50:
        issues.append("دقة الصورة منخفضة")

    if overall >= 85:
        label = "ممتازة"
    elif overall >= 65:
        label = "جيدة"
    elif overall >= 45:
        label = "متوسطة"
    else:
        label = "ضعيفة"

    return {
        "quality_score": round(overall, 2),
        "quality_label": label,
        "blur_score": round(blur_score, 2),
        "brightness_score": round(brightness, 2),
        "contrast_score": round(contrast, 2),
        "resolution_score": round(resolution, 2),
        "issues": issues or ["لا يوجد"]
    }
