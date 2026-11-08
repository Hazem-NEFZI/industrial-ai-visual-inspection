"""Evaluate a local MVTec-style category without downloading data.
Layout: train/good/*, test/good/*, test/<defect>/*.
An entire category is evaluated; no files from test enter model or threshold fit.
"""

import argparse, csv, json, time, hashlib
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from .patch_memory import PatchMemory


def files(path):
    return sorted(
        p
        for p in Path(path).rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )


def read(path):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f"Cannot decode {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def evaluate(root, output, seed=7):
    root = Path(root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    normal = files(root / "train/good")
    test = files(root / "test")
    if len(normal) < 8 or not test:
        raise ValueError("Need >=8 train/good images and a labeled test tree")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normal))
    nval = max(2, len(normal) // 5)
    val = [normal[i] for i in order[:nval]]
    train = [normal[i] for i in order[nval:]]
    # Exact byte duplicates between splits invalidate this benchmark.
    seen = {}
    for split, paths in [("train", train), ("validation", val), ("test", test)]:
        for p in paths:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f"Identical image bytes across splits: {p}")
            seen[digest] = split
    model = PatchMemory(seed=seed).fit(read(p) for p in train)
    threshold = float(np.quantile([model.score(read(p))[0] for p in val], 0.95))
    rows = []
    for p in test:
        image = read(p)
        start = time.perf_counter()
        score, _ = model.score(image)
        rows.append(
            {
                "file": str(p.relative_to(root)),
                "label": int(p.relative_to(root / "test").parts[0] != "good"),
                "score": score,
                "seconds": time.perf_counter() - start,
            }
        )
    y = np.array([r["label"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    if len(np.unique(y)) < 2:
        raise ValueError("Test data must include normal and anomalous images")
    report = {
        "method": "handcrafted patch-memory baseline",
        "seed": seed,
        "train_images": len(train),
        "validation_images": len(val),
        "test_images": len(test),
        "image_auroc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "normal_validation_threshold": threshold,
        "confusion_matrix": confusion_matrix(
            y, scores > threshold, labels=[0, 1]
        ).tolist(),
        "median_inference_seconds": float(np.median([r["seconds"] for r in rows])),
        "split_files": {
            "train": [str(p.relative_to(root)) for p in train],
            "validation": [str(p.relative_to(root)) for p in val],
        },
    }
    model.save(out / "patch_memory.npz")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("category")
    p.add_argument("--output", default="outputs/benchmark")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(json.dumps(evaluate(a.category, a.output, a.seed), indent=2))


if __name__ == "__main__":
    main()
