import cv2
import pytest
from industrial_cv.data import component
from industrial_cv.evaluate_folder import evaluate


def test_folder_protocol(tmp_path):
    root = tmp_path / "data"
    for split, n, seed, defect in [
        ("train/good", 10, 10, False),
        ("test/good", 3, 100, False),
        ("test/scratch", 3, 200, True),
    ]:
        p = root / split
        p.mkdir(parents=True)
        for i in range(n):
            im, _ = component(seed + i, defect)
            cv2.imwrite(str(p / f"{i}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    report = evaluate(root, tmp_path / "out")
    assert report["test_images"] == 6
    assert report["image_auroc"] > 0.8
    (root / "test/good/0.png").write_bytes((root / "train/good/0.png").read_bytes())
    with pytest.raises(ValueError, match="Identical"):
        evaluate(root, tmp_path / "duplicates")
