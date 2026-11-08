import argparse, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from .data import component, dataset
from .baseline import fit_template, score_template, learn_threshold, anomaly_view


def main(output="outputs"):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    reference, _ = dataset(36, start_seed=1000)
    valid, _ = dataset(12, start_seed=3000)
    template = fit_template(reference)
    threshold = learn_threshold(valid, template)
    xnormal, _ = dataset(24, start_seed=5000)
    xdefect, defect_masks = dataset(24, start_seed=6000, defect=True)
    scores = np.array(
        [
            float(np.quantile(score_template(x, template), 0.99))
            for x in np.concatenate([xnormal, xdefect])
        ]
    )
    labels = np.array([0] * len(xnormal) + [1] * len(xdefect))
    image_auroc = float(roc_auc_score(labels, scores))
    pixel_scores = np.concatenate(
        [score_template(im, template).ravel() for im in xdefect]
    )
    pixel_labels = defect_masks.ravel().astype(bool)
    pixel_auroc = float(roc_auc_score(pixel_labels, pixel_scores))
    predicted_pixels = pixel_scores > threshold
    intersection = np.logical_and(predicted_pixels, pixel_labels).sum()
    union = np.logical_or(predicted_pixels, pixel_labels).sum()
    pixel_iou = float(intersection / union) if union else 0.0
    # Image-level threshold uses 99th-percentile heatmap scores from validation images.
    val_image_scores = [np.quantile(score_template(im, template), 0.99) for im in valid]
    image_threshold = float(np.quantile(val_image_scores, 0.95))
    decisions = scores > image_threshold
    metrics = {
        "dataset": "synthetic, aligned components",
        "method": "fixed normal-reference median (not a neural network)",
        "image_auroc": image_auroc,
        "defect_only_pixel_auroc": pixel_auroc,
        "defect_only_pixel_iou_at_normal_validation_threshold": pixel_iou,
        "validation_image_threshold": image_threshold,
        "false_positive_rate_on_test_normals": float(decisions[: len(xnormal)].mean()),
        "true_positive_rate_on_test_defects": float(decisions[len(xnormal) :].mean()),
        "pixel_heatmap_visualization_threshold": threshold,
        "test_normal_images": len(xnormal),
        "test_defective_images": len(xdefect),
    }
    (path / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    rgb, truth = component(seed=6000, defect=True)
    score = score_template(rgb, template)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.45), layout="constrained")
    ax[0].imshow(template)
    ax[0].set_title("Normal appearance")
    ax[1].imshow(rgb)
    ax[1].set_title("Inspected part")
    ax[2].imshow(
        score, cmap="turbo", vmin=0, vmax=max(float(np.quantile(score, 0.99)), 0.01)
    )
    ax[2].set_title("Anomaly heatmap")
    ax[3].imshow(anomaly_view(rgb, score))
    ax[3].contour(truth, levels=[0.5], colors="white", linewidths=[1.3])
    ax[3].set_title("Heatmap + true mask")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("INDUSTRIAL VISUAL INSPECTION | SYNTHETIC BASELINE", weight="bold")
    fig.savefig(path / "demo.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs")
    a = p.parse_args()
    main(a.output)
