import io

import numpy as np
import streamlit as st
from PIL import Image
import cv2

import foto_para_desenho as engine


st.set_page_config(page_title="Foto → Desenho", layout="wide")

st.title("Foto → Desenho (Local)")
st.caption("Envie uma foto, escolha o efeito, ajuste parâmetros e baixe o resultado em PNG.")

uploaded = st.file_uploader("Escolha uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

mode = st.selectbox(
    "Efeito",
    ["sketch", "colorir_linhas_pro", "xdog", "hq", "cartoon", "edges", "grayscale", "blur"],
    index=0,
)

# Parâmetros (os que fazem diferença na prática)
col1, col2, col3, col4 = st.columns(4)

with col1:
    sketch_strength = st.slider("Sketch strength", 0.0, 1.0, 0.6, 0.05)

with col2:
    line_thickness = st.slider("Espessura (HQ/PRO)", 1, 6, 2, 1)

with col3:
    poster_levels = st.slider("Poster levels (HQ)", 2, 12, 6, 1)

with col4:
    blur_ksize = st.slider("Blur ksize", 1, 51, 9, 2)

st.divider()
st.subheader("Ajustes avançados (PRO / XDoG / Bordas)")

a1, a2, a3, a4 = st.columns(4)
with a1:
    block_size = st.slider("PRO block_size", 3, 51, 19, 2)  # o motor ajusta p/ ímpar se necessário
with a2:
    C = st.slider("PRO C", 0, 20, 7, 1)
with a3:
    close_size = st.slider("close_size", 1, 7, 2, 1)
with a4:
    contrast = st.slider("PRO contrast", 1.0, 2.5, 1.6, 0.1)

b1, b2, b3, b4 = st.columns(4)
with b1:
    sigma = st.slider("XDoG sigma", 0.3, 2.0, 0.9, 0.1)
with b2:
    phi = st.slider("XDoG phi", 5.0, 30.0, 16.0, 1.0)
with b3:
    edges_low = st.slider("Edges low", 0, 200, 50, 5)
with b4:
    edges_high = st.slider("Edges high", 0, 300, 150, 5)

run = st.button("Gerar")

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

if run:
    if not uploaded:
        st.warning("Envie uma imagem primeiro.")
        st.stop()

    pil_img = Image.open(uploaded)
    img_bgr = pil_to_bgr(pil_img)

    # Processa
    if mode == "sketch":
        out = engine.fx_pencil_sketch(img_bgr, strength=sketch_strength)

    elif mode == "colorir_linhas_pro":
        out = engine.fx_colorir_linhas_pro(
            img_bgr,
            block_size=block_size,
            C=C,
            close_size=close_size,
            thickness=line_thickness,
            contrast=contrast,
        )

    elif mode == "xdog":
        out = engine.fx_xdog_lineart(
            img_bgr,
            sigma=sigma,
            k=1.6,
            tau=0.97,
            eps=0.02,
            phi=phi,
            close_size=close_size,
        )

    elif mode == "hq":
        out = engine.fx_hq(img_bgr, line_thickness=line_thickness, poster_levels=poster_levels)

    elif mode == "cartoon":
        out = engine.fx_cartoon(img_bgr)

    elif mode == "edges":
        out = engine.fx_edges(img_bgr, low=edges_low, high=edges_high)

    elif mode == "grayscale":
        out = engine.fx_grayscale(img_bgr)

    elif mode == "blur":
        out = engine.fx_blur(img_bgr, ksize=blur_ksize)

    else:
        st.error("Modo não suportado.")
        st.stop()

    out_pil = bgr_to_pil(out)

    c1, c2 = st.columns(2)
    with c1:
        st.image(pil_img, caption="Original", use_container_width=True)
    with c2:
        st.image(out_pil, caption=f"Resultado: {mode}", use_container_width=True)

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button(
        "Baixar PNG",
        data=buf.getvalue(),
        file_name=f"resultado_{mode}.png",
        mime="image/png",
    )
