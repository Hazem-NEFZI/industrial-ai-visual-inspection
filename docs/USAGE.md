# Usage

```bash
python -m pip install -e ".[dev,app,deep]"
python -m streamlit run app.py
python -m industrial_cv.train --epochs 5 --output outputs/autoencoder
python -m industrial_cv.evaluate_folder /path/to/category --output outputs/benchmark
```

External category layout:

```text
category/
  train/good/normal_001.png
  test/good/normal_test.png
  test/scratch/defective_test.png
```

Use at least eight normal training images and both test classes. Example paths are illustrative, not bundled external files. Your dataset license controls redistribution. Original MVTec source: https://www.mvtec.com/company/research/datasets/mvtec-ad .

The app uses synthetic training references by default and resizes uploads to 96×96. It displays anomaly scores, not pass/fail claims for arbitrary uploaded photos. Train your own model through the folder workflow before interpreting real-image scores.
