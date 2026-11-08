# Validation record — 7 November 2026

## Executed locally

Python 3.12.14, Linux x86-64, CPU execution. Core tests plus available optional app/neural tests:

```text
.........                                                                [100%]
9 passed in 3.15s
```

The suite includes success paths and malformed inputs. It does not prove absence of bugs. Static lint and format checks were run. Multi-version GitHub Actions is configured but has not run on GitHub as part of this task. Windows, CUDA, real robots and external pretrained SAM were not tested.

## Reproduced demonstrations

- `python -m industrial_cv.demo --output outputs/demo`: exit 0.
- `python -m industrial_cv.train --output outputs/autoencoder --epochs 5`: exit 0.

Machine-readable sample results are in `assets/`. All included benchmark inputs were synthetic. Real-data folder and image-pair workflows were smoke-tested with generated fixtures, not public industrial benchmarks.

## Dependency environment

- numpy: 2.3.5
- scipy: 1.17.0
- scikit-learn: 1.8.0
- opencv-contrib-python-headless: 4.14.0.94
- torch: 2.15.0
- matplotlib: 3.10.8
- pandas: 2.2.3
- pillow: 13.0.0
- pytest: 9.2.0
- ruff: 0.17.4
- streamlit: 1.67.0
- joblib: 1.5.3
