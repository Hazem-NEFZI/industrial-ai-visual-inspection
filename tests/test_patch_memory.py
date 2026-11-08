import numpy as np
import pytest
from industrial_cv.patch_memory import PatchMemory
from industrial_cv.data import dataset, component


def test_memory_bound_and_anomaly_order(tmp_path):
    normals, _ = dataset(12, 1000)
    m = PatchMemory(max_patches=500).fit(normals)
    assert len(m.memory) == 500
    normal, _ = component(3000)
    bad, _ = component(6000, True)
    a, _ = m.score(normal)
    b, heat = m.score(bad)
    assert b > a
    assert heat.shape == bad.shape[:2]
    m.save(tmp_path / "model.npz")
    loaded = PatchMemory.load(tmp_path / "model.npz")
    assert np.allclose(loaded.score(bad)[1], heat)


def test_not_fitted_or_invalid():
    with pytest.raises(ValueError):
        PatchMemory().score(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        PatchMemory().fit([])
    with pytest.raises(ValueError):
        PatchMemory(grid=0)


def test_repeatability():
    x, _ = dataset(4, 10)
    a = PatchMemory(max_patches=100, seed=9).fit(x)
    b = PatchMemory(max_patches=100, seed=9).fit(x)
    assert np.array_equal(a.memory, b.memory)
