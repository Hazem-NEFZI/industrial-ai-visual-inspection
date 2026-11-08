"""Local RGB/gradient patch descriptors with bounded nearest-neighbor memory.

This is an explainable handcrafted descriptor baseline, NOT PatchCore.
Only normal training images enter memory; thresholds use normal validation data.
"""

import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors


class PatchMemory:
    def __init__(self, size=96, grid=12, max_patches=4000, seed=7):
        if (
            not all(isinstance(x, int) for x in (size, grid, max_patches))
            or size < 16
            or not 2 <= grid <= size
            or max_patches < 2
        ):
            raise ValueError("Invalid patch memory dimensions")
        self.size, self.grid, self.max_patches, self.seed = (
            size,
            grid,
            max_patches,
            seed,
        )
        self.memory = None

    def descriptors(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected uint8 RGB image")
        if min(rgb.shape[:2]) < 2 or rgb.shape[0] * rgb.shape[1] > 24_000_000:
            raise ValueError("Invalid image dimensions")
        x = cv2.resize(rgb, (self.size, self.size)).astype(np.float32) / 255
        mean = cv2.blur(x, (7, 7))
        variance = np.maximum(cv2.blur(x * x, (7, 7)) - mean * mean, 0)
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        grad = cv2.blur(np.sqrt(gx * gx + gy * gy), (5, 5))
        f = np.dstack([mean, np.sqrt(variance), grad])
        return cv2.resize(
            f, (self.grid, self.grid), interpolation=cv2.INTER_AREA
        ).reshape(-1, 7)

    def fit(self, normal_images):
        # Stream input into bounded reservoir instead of concatenating an entire dataset.
        rng = np.random.default_rng(self.seed)
        reservoir = []
        seen = 0
        for image in normal_images:
            for descriptor in self.descriptors(image):
                seen += 1
                if len(reservoir) < self.max_patches:
                    reservoir.append(descriptor.copy())
                else:
                    j = int(rng.integers(seen))
                    if j < self.max_patches:
                        reservoir[j] = descriptor.copy()
        if seen < 2:
            raise ValueError("At least two normal descriptors required")
        raw = np.asarray(reservoir)
        self.center = raw.mean(axis=0)
        self.scale = np.maximum(raw.std(axis=0), 0.02)
        self.memory = (raw - self.center) / self.scale
        self._index = NearestNeighbors(n_neighbors=1).fit(self.memory)
        return self

    def score(self, image):
        if self.memory is None:
            raise ValueError("Fit or load the model first")
        f = (self.descriptors(image) - self.center) / self.scale
        distance = self._index.kneighbors(f, return_distance=True)[0][:, 0]
        heat = cv2.resize(
            distance.reshape(self.grid, self.grid),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return float(np.quantile(distance, 0.99)), heat

    def save(self, path):
        if self.memory is None:
            raise ValueError("Model is not fitted")
        np.savez_compressed(
            path,
            memory=self.memory,
            center=self.center,
            scale=self.scale,
            config=np.array([self.size, self.grid, self.max_patches, self.seed]),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            size, grid, limit, seed = map(int, f["config"])
            obj = cls(size, grid, limit, seed)
            obj.memory = f["memory"].copy()
            obj.center = f["center"].copy()
            obj.scale = f["scale"].copy()
        if (
            obj.memory.ndim != 2
            or obj.memory.shape[1] != 7
            or len(obj.memory) > limit
            or len(obj.memory) < 2
            or obj.center.shape != (7,)
            or obj.scale.shape != (7,)
            or np.any(obj.scale <= 0)
            or not all(
                np.isfinite(x).all() for x in (obj.memory, obj.center, obj.scale)
            )
        ):
            raise ValueError("Invalid model arrays")
        obj._index = NearestNeighbors(n_neighbors=1).fit(obj.memory)
        return obj
