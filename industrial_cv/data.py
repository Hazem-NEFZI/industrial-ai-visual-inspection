"""Small deterministic simulated components. Outputs RGB uint8 and binary defect mask."""

import numpy as np
import cv2


def component(seed=0, defect=False, size=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    texture = (
        145
        + 7 * np.sin(xx / 12)
        + 4 * np.sin(yy / 8)
        + rng.normal(0, 2.5, (size, size))
    )
    rgb = np.stack([texture * 0.92, texture, texture * 1.05], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.rectangle(rgb, (8, 8), (size - 9, size - 9), (170, 179, 189), -1)
    cv2.rectangle(rgb, (8, 8), (size - 9, size - 9), (90, 105, 124), 2)
    for cx, cy in [(19, 19), (size - 20, 19), (19, size - 20), (size - 20, size - 20)]:
        cv2.circle(rgb, (cx, cy), 5, (60, 73, 88), -1, cv2.LINE_AA)
        cv2.circle(rgb, (cx - 1, cy - 1), 2, (210, 219, 225), -1, cv2.LINE_AA)
    cv2.line(rgb, (28, 43), (size - 28, 43), (108, 126, 139), 2, cv2.LINE_AA)
    cv2.line(rgb, (28, 55), (size - 28, 55), (108, 126, 139), 2, cv2.LINE_AA)
    mask = np.zeros((size, size), np.uint8)
    if defect:
        if seed % 2 == 0:
            # Scratched metal: thin irregular trace.
            pts = np.array([[27, 26], [40, 35], [49, 48], [64, 59], [72, 73]], np.int32)
            pts[:, 0] += int(rng.integers(-4, 5))
            pts[:, 1] += int(rng.integers(-4, 5))
            cv2.polylines(mask, [pts], False, 1, 4, cv2.LINE_AA)
            cv2.polylines(rgb, [pts], False, (35, 49, 67), 4, cv2.LINE_AA)
        else:
            center = (
                int(size * 0.57 + rng.integers(-5, 6)),
                int(size * 0.47 + rng.integers(-5, 6)),
            )
            angle = int(rng.integers(0, 50))
            cv2.ellipse(mask, center, (11, 8), angle, 0, 360, 1, -1)
            cv2.ellipse(rgb, center, (11, 8), angle, 0, 360, (44, 96, 160), -1)
            for _ in range(8):
                x = center[0] + int(rng.integers(-10, 11))
                y = center[1] + int(rng.integers(-7, 8))
                if 0 <= x < size and 0 <= y < size and mask[y, x]:
                    cv2.circle(rgb, (x, y), int(rng.integers(1, 3)), (73, 121, 169), -1)
    # Mild camera noise applies to normal and defective samples alike.
    rgb = np.clip(
        rgb.astype(np.float32) + rng.normal(0, 1.8, rgb.shape), 0, 255
    ).astype(np.uint8)
    return rgb, mask


def dataset(n, start_seed=0, defect=False, size=96):
    pairs = [component(seed=start_seed + i, defect=defect, size=size) for i in range(n)]
    return np.stack([x for x, _ in pairs]), np.stack([m for _, m in pairs])
