# Design: anomaly inspection

## Three distinct methods

| Method | Training signal | Main assumption | Failure mode |
|---|---|---|---|
| Median reference | Aligned normal images | Fixed part pose and illumination | Misalignment resembles a defect |
| Patch memory | Normal RGB/gradient descriptors | Local appearance resembles training | Weak sensitivity to semantic/contextual defects |
| CNN autoencoder | Normal-image reconstruction | Defects reconstruct poorly | Model can reconstruct defects or blur normal edges |

The patch memory uses seven handcrafted channel statistics, standardizes them using training data, retains a seeded bounded reservoir and queries nearest neighbors. It is not PatchCore and uses no pretrained backbone. Memory is saved as numeric NPZ with pickle disabled.

## Evaluation contract

Folder benchmarking uses train/good for training and normal validation. The validation 95th-percentile image score sets the decision threshold; test labels only enter evaluation. Exact image-byte duplicates across splits are rejected. This is not a perceptual duplicate detector: near duplicates and production-batch leakage require a curated split. Scores and per-image timings are exported.

The folder benchmark reports image-level AUROC, AP and confusion counts. Pixel-level metrics are currently provided only for the synthetic reference demo; do not infer real-dataset pixel localization quality from it.

## Honest neural baseline

The included five-epoch example is a small reconstruction model, not a state-of-the-art inspection model. Review its reported false-positive and true-positive rates. External data, tuned architecture and independently held-out evaluation are needed before operational claims.
