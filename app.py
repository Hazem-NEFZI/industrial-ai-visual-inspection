"""Local anomaly-inspection workspace."""

import io, json, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from industrial_cv.data import dataset, component
from industrial_cv.baseline import fit_template, score_template, anomaly_view
from industrial_cv.patch_memory import PatchMemory

st.set_page_config(page_title="Industrial Vision Lab", page_icon="◈", layout="wide")
st.caption("HAZEM NEFZI / INDUSTRIAL AI LAB")
st.title("Industrial Visual Inspection")
st.markdown(
    "Compare normal appearance, local patch descriptors, and neural reconstruction residuals."
)
with st.sidebar:
    mode = st.selectbox(
        "Inspection method", ["Reference baseline", "Patch memory", "CNN autoencoder"]
    )
    scenario = st.selectbox(
        "Synthetic example", ["Scratched part", "Corrosion", "Normal part"]
    )
    uploaded = st.file_uploader("Or upload an RGB image", type=["png", "jpg", "jpeg"])
    st.info(
        "Models here learn simulated 96×96 parts. A real image requires separately trained and validated models."
    )
try:
    if uploaded:
        image = Image.open(uploaded)
        if image.width * image.height > 24_000_000:
            raise ValueError("Resize to below 24 megapixels")
        im = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((96, 96)))
    else:
        im, _ = component(
            6001 if scenario == "Corrosion" else 6000, defect=scenario != "Normal part"
        )

    @st.cache_resource
    def get_reference():
        normals, _ = dataset(36, 1000)
        return fit_template(normals)

    @st.cache_resource
    def get_memory():
        normals, _ = dataset(36, 1000)
        return PatchMemory(max_patches=1500).fit(normals)

    start = time.perf_counter()
    if mode == "Reference baseline":
        heat = score_template(im, get_reference())
        score = float(np.quantile(heat, 0.99))
    elif mode == "Patch memory":
        score, heat = get_memory().score(im)
    else:
        import torch
        from industrial_cv.model import ConvAutoencoder, anomaly_map

        path = Path("assets/synthetic_autoencoder_5epochs.pt")

        @st.cache_resource
        def get_network(mtime):
            m = ConvAutoencoder()
            m.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
            )
            m.eval()
            return m

        heat = anomaly_map(get_network(path.stat().st_mtime_ns), im)[0]
        score = float(np.quantile(heat, 0.99))
    elapsed = time.perf_counter() - start
except Exception as exc:
    st.error(f"Inspection failed: {exc}")
    st.stop()
cols = st.columns(3)
cols[0].image(im, caption="Inspected part", use_container_width=True)
normalized = np.clip(heat / max(float(np.quantile(heat, 0.99)), 1e-8), 0, 1)
cols[1].image(
    normalized,
    caption="Residual intensity (display normalized)",
    use_container_width=True,
)
cols[2].image(
    anomaly_view(im, heat), caption="Anomaly overlay", use_container_width=True
)
a, b = st.columns(2)
a.metric("Anomaly score", f"{score:.5f}")
b.metric("Compute incl. cold setup", f"{elapsed * 1000:.1f} ms")
st.caption(
    "Scores differ by method and are not comparable probabilities. No uncalibrated pass/fail decision is shown."
)
buffer = io.BytesIO()
Image.fromarray(anomaly_view(im, heat)).save(buffer, format="PNG")
st.download_button("Export overlay", buffer.getvalue(), "inspection.png", "image/png")
st.download_button(
    "Export score JSON",
    json.dumps(
        {
            "method": mode,
            "score": score,
            "seconds_including_cold_setup": elapsed,
            "data": "uploaded" if uploaded else "synthetic",
        },
        indent=2,
    ),
    "inspection.json",
    "application/json",
)
