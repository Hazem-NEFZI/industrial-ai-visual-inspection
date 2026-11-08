import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .data import dataset
from .model import ConvAutoencoder, anomaly_map


def train(output="outputs/autoencoder", epochs=5, seed=43):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_images, _ = dataset(96, 1000)
    validation, _ = dataset(28, 3000)
    test_normal, _ = dataset(28, 5000)
    test_defect, test_masks = dataset(28, 6000, defect=True)
    x = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32) / 255
    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x), batch_size=16, shuffle=True
    )
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    training_losses = []
    for epoch in range(epochs):
        model.train()
        total = 0
        for (images,) in data:
            images = images.to(device)
            optimizer.zero_grad()
            output_img = model(images)
            loss = torch.nn.functional.l1_loss(output_img, images)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        training_losses.append(total / len(x))
        print(
            f"Epoch {epoch + 1}/{epochs} | reconstruction L1={training_losses[-1]:.5f}"
        )

    # Score each image using its top-1% pixel residual; no test labels used to choose threshold.
    def score_images(images):
        values = []
        for s in range(0, len(images), 8):
            maps = anomaly_map(model, images[s : s + 8], device)
            values.extend(np.quantile(maps, 0.99, axis=(1, 2)).tolist())
        return np.array(values)

    val_scores = score_images(validation)
    cutoff = float(np.quantile(val_scores, 0.95))
    scores0 = score_images(test_normal)
    scores1 = score_images(test_defect)
    scores = np.r_[scores0, scores1]
    labels = np.r_[np.zeros(len(scores0)), np.ones(len(scores1))]
    stats = {
        "data": "synthetic",
        "architecture": "convolutional autoencoder",
        "device": device,
        "epochs": epochs,
        "train_reconstruction_l1": training_losses,
        "validation_threshold_95pct": cutoff,
        "test_image_auroc": float(roc_auc_score(labels, scores)),
        "test_normal_false_positive_rate": float(np.mean(scores0 > cutoff)),
        "test_defect_true_positive_rate": float(np.mean(scores1 > cutoff)),
        "test_normal_images": len(scores0),
        "test_defective_images": len(scores1),
    }
    torch.save(
        {"state_dict": model.cpu().state_dict(), "seed": seed, "image_size": 96},
        out / "model.pt",
    )
    (out / "metrics.json").write_text(json.dumps(stats, indent=2))
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("train_reconstruction_l1",)},
            indent=2,
        )
    )
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/autoencoder")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=43)
    a = p.parse_args()
    train(a.output, a.epochs, a.seed)
