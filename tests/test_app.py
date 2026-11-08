import pytest


def test_default_app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(
        timeout=30
    )
    assert len(app.exception) == 0
    assert len(app.error) == 0


def test_other_inspection_modes():
    pytest.importorskip("streamlit")
    pytest.importorskip("torch")
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(
        timeout=30
    )
    for mode in ["Patch memory", "CNN autoencoder"]:
        app.selectbox[0].select(mode).run(timeout=30)
        assert not app.exception
        assert not app.error
