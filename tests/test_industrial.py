import numpy as np
from industrial_cv.data import component, dataset
from industrial_cv.baseline import fit_template, score_template, learn_threshold


def test_dataset_has_ground_truth_masks():
    x, y = component(seed=10, defect=True)
    assert x.shape == (96, 96, 3)
    assert y.shape == (96, 96)
    assert int(y.sum()) > 0
    normal, masks = dataset(3, 10)
    assert not masks.any()


def test_reference_scores():
    normal, _ = dataset(8, 1000)
    ref = fit_template(normal)
    a = score_template(normal[0], ref)
    assert a.shape == (96, 96)
    assert np.isfinite(a).all()
    assert learn_threshold(normal, ref) > 0


def test_network_forward_and_heatmap():
    import pytest

    torch = pytest.importorskip("torch")
    from industrial_cv.model import ConvAutoencoder, anomaly_map

    m = ConvAutoencoder()
    im, _ = dataset(2, 0)
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 96, 96)).shape == (1, 3, 96, 96)
    s = anomaly_map(m, im)
    assert s.shape == (2, 96, 96)
    assert np.isfinite(s).all()
