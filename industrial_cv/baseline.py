import numpy as np
import cv2


def fit_template(normal_images):
    raw = np.asarray(normal_images)
    if raw.dtype != np.uint8:
        raise ValueError("Expected uint8 training images")
    images = raw.astype(np.float32) / 255.0
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) < 2:
        raise ValueError("Expected N RGB images, N>=2")
    return np.median(images, axis=0).astype(np.float32)


def score_template(rgb, template):
    raw = np.asarray(rgb)
    if raw.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    rgb = raw.astype(np.float32) / 255
    template = np.asarray(template, dtype=np.float32)
    if rgb.shape != template.shape:
        raise ValueError("Image must be aligned to template")
    score = np.mean(np.abs(rgb - template), axis=2)
    return cv2.GaussianBlur(score, (5, 5), 0)


def learn_threshold(normals, template, quantile=0.995):
    if not (0 < quantile < 1):
        raise ValueError("quantile must be within (0,1)")
    scores = np.concatenate([score_template(im, template).ravel() for im in normals])
    return float(np.quantile(scores, quantile))


def anomaly_view(rgb, heatmap):
    s = np.asarray(heatmap, dtype=float)
    upper = max(float(np.quantile(s, 0.99)), 0.00001)
    scaled = np.clip(s / upper, 0, 1)
    heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip(0.42 * np.asarray(rgb) + 0.58 * heat, 0, 255).astype(np.uint8)
