# app.py
# Execute com: streamlit run app.py

import io
import hashlib
import inspect
import json
import logging
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import cv2

import foto_para_desenho as engine
from image_input import prepare_image

DEMO_PATH = Path(__file__).parent / "assets" / "demo_fotoimagem.jpg"
logger = logging.getLogger(__name__)


# =========================
# Config / UI base
# =========================
st.set_page_config(page_title="FotoImagem | Foto → Desenho", page_icon="🎨", layout="wide")
st.title("FotoImagem · Foto → Desenho")
st.caption("Experimente a demonstração ou envie uma imagem. Escolha entre 15 efeitos e baixe sua criação em PNG.")


# =========================
# Helpers
# =========================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def call_with_supported_kwargs(func, **kwargs):
    sig = inspect.signature(func)
    params = sig.parameters
    allowed = set(params.keys())
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return func(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return func(**filtered)


def get_func(name: str):
    return getattr(engine, name)


# =========================
# Nomes comerciais (UI) -> modos internos (engine)
# =========================
EFFECTS = {
    "✍️ Sketch Artístico (lápis realista)": {
        "mode": "sketch",
        "desc": "Desenho a lápis suave, bom para retratos.",
    }, 
    "🎨 Livro de Colorir PRO (linhas limpas e nítidas)": {
        "mode": "colorir_linhas_pro_v2",
        "desc": "Ideal para imprimir e colorir. Linhas mais limpas, menos ruído no rosto e na roupa.",
    },
    "🦸 Linhas HQ (quadrinhos / contorno marcante)": {
        "mode": "colorir_linhas_hq",
        "desc": "Estilo quadrinhos com contorno forte e visual dramático.",
    },
    "🖊️ Ink Noir (tinta nanquim)": {
        "mode": "ink",
        "desc": "Traços fortes tipo nanquim, ótimo para artes P&B.",
    },
    "⚡ Traço X-DoG (lineart expressivo)": {
        "mode": "xdog",
        "desc": "Lineart com contraste alto, estilo artístico/impactante.",
    },
    "😄 Cartoon Pop (desenho animado)": {
        "mode": "cartoon",
        "desc": "Visual de desenho animado simples e divertido.",
    },
    "🧼 Cartoon Clean (pele mais suave)": {
        "mode": "cartoon_clean",
        "desc": "Cartoon com suavização mais forte, reduz textura e ruído.",
    },
    "🌈 Aquarela (pintura suave)": {
        "mode": "watercolor",
        "desc": "Efeito aquarela com bordas leves e cores macias.",
    },
    "🎭 Poster Art (cores chapadas)": {
        "mode": "poster_art",
        "desc": "Posterização artística com contornos e poucos níveis de cor.",
    },
    "📸 Retrato Suave (skin smoothing)": {
        "mode": "soft_portrait",
        "desc": "Suaviza pele e deixa o retrato mais agradável.",
    },
    "🖤 Fine Art P&B (alto contraste)": {
        "mode": "bw_fineart",
        "desc": "Preto e branco refinado com contraste e textura controlada.",
    },
    "✨ Glow Dreamy (brilho etéreo)": {
        "mode": "glow_dreamy",
        "desc": "Brilho suave tipo sonho, ótimo para fotos noturnas.",
    },
    "🔎 Contornos (edges)": {
        "mode": "edges",
        "desc": "Apenas detecção de bordas (efeito rápido).",
    },
    "⚪ Preto & Branco (grayscale)": {
        "mode": "grayscale",
        "desc": "Converte para tons de cinza.",
    },
    "🌫️ Desfoque (blur)": {
        "mode": "blur",
        "desc": "Aplica blur (desfoque) para suavizar a foto.",
    },
}


# =========================
# Input
# =========================
source = st.radio("Imagem para transformar", ["Experimentar demonstração", "Enviar minha imagem"], horizontal=True)
image_data = None
if source == "Experimentar demonstração":
    image_data = DEMO_PATH.read_bytes()
    st.caption("Cena fictícia criada por IA para demonstração dos efeitos.")
else:
    uploaded = st.file_uploader("Escolha uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_data = uploaded.getvalue()
    st.caption("JPG ou PNG · até 10 MB e 16 megapixels · saída com até 1.600 pixels no maior lado.")
    st.caption("A imagem é enviada ao servidor para processamento durante a sessão. O aplicativo não a grava em disco nem a envia a APIs de imagem.")

pil_img = None
if image_data is not None:
    try:
        pil_img, resized = prepare_image(image_data)
        if resized:
            st.info(f"Imagem ajustada para {pil_img.width} × {pil_img.height} pixels para agilizar o processamento.")
    except ValueError as exc:
        st.error(str(exc))

effect_label = st.selectbox("Efeito", list(EFFECTS.keys()), index=0)
mode = EFFECTS[effect_label]["mode"]
st.caption(EFFECTS[effect_label]["desc"])

st.subheader("Ajustes do efeito")
params = {}


# =========================
# Sliders dinâmicos por modo
# =========================
if mode == "colorir_linhas_pro_v2":
    st.info("Melhor opção para imprimir e colorir. Ajuste em 3 passos: Linhas → Limpeza → Espessura.")
    preset = st.selectbox("Preset", ["Retrato (dia)", "Retrato (noite)", "Paisagem"], index=0)

    # Observação: mantemos valores como estavam; apenas removemos a UI de 'Remover fundo'.
    if preset == "Retrato (dia)":
        defaults = dict(
            edges_low=35, edges_high=120, min_area=140, close_size=3, thickness=2,
            skin_suppress=True, skin_strength=0.55,
            bilateral_d=9, bilateral_sigma_color=60, bilateral_sigma_space=60,
            thinning=True
        )
    elif preset == "Retrato (noite)":
        defaults = dict(
            edges_low=30, edges_high=110, min_area=170, close_size=4, thickness=2,
            skin_suppress=True, skin_strength=0.60,
            bilateral_d=9, bilateral_sigma_color=70, bilateral_sigma_space=70,
            thinning=True
        )
    else:
        defaults = dict(
            edges_low=40, edges_high=140, min_area=160, close_size=3, thickness=2,
            skin_suppress=False, skin_strength=0.0,
            bilateral_d=9, bilateral_sigma_color=60, bilateral_sigma_space=60,
            thinning=True
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        params["edges_low"] = st.slider("Bordas (low)", 0, 200, defaults["edges_low"], 5)
        params["edges_high"] = st.slider("Bordas (high)", 0, 300, defaults["edges_high"], 5)

    with c2:
        params["min_area"] = st.slider("Limpeza (min_area)", 0, 900, defaults["min_area"], 10)
        params["close_size"] = st.slider("Conectar falhas (close)", 1, 9, defaults["close_size"], 1)

    with c3:
        params["thickness"] = st.slider("Espessura final", 1, 6, defaults["thickness"], 1)
        params["thinning"] = st.checkbox("Afinar linhas (thinning)", value=defaults["thinning"])

    st.divider()

    # --- Sessão deslizante (scroll) dentro de um expander ---
    with st.expander("Ajustes avançados (opcional)", expanded=False):
        # Container com altura fixa para rolagem (sessão deslizante)
        adv = st.container(height=260)

        with adv:
            c4, c5 = st.columns(2)

            with c4:
                params["skin_suppress"] = st.checkbox(
                    "Reduzir detalhes no rosto/pele",
                    value=defaults["skin_suppress"]
                )
                params["skin_strength"] = st.slider(
                    "Força (pele)", 0.0, 1.0, defaults["skin_strength"], 0.05
                )

            with c5:
                params["bilateral_d"] = st.slider("Bilateral d", 3, 15, defaults["bilateral_d"], 2)
                params["bilateral_sigma_color"] = st.slider("Bilateral cor", 10, 150, defaults["bilateral_sigma_color"], 5)
                params["bilateral_sigma_space"] = st.slider("Bilateral espaço", 10, 150, defaults["bilateral_sigma_space"], 5)

            st.caption("Dica: se a foto estiver muito “texturizada”, aumente um pouco o Bilateral; se perder detalhes demais, reduza.")

elif mode == "colorir_linhas_hq":
    st.info("Linhas em estilo HQ (mais marcadas).")
    c1, c2, c3 = st.columns(3)
    with c1:
        params["line_thickness"] = st.slider("Espessura do contorno", 1, 6, 2, 1)
    with c2:
        params["poster_levels"] = st.slider("Posterização (níveis)", 2, 12, 6, 1)
    with c3:
        params["min_area"] = st.slider("Limpeza (min_area)", 0, 600, 120, 10)

elif mode == "xdog":
    c1, c2, c3 = st.columns(3)
    with c1:
        params["sigma"] = st.slider("Sigma", 0.3, 2.0, 0.9, 0.1)
    with c2:
        params["phi"] = st.slider("Phi (contraste das linhas)", 5.0, 30.0, 16.0, 1.0)
    with c3:
        params["close_size"] = st.slider("Conectar traços (close_size)", 1, 9, 2, 1)

elif mode == "ink":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        params["sigma"] = st.slider("Sigma", 0.3, 2.0, 0.9, 0.1)
    with c2:
        params["phi"] = st.slider("Phi", 5.0, 30.0, 18.0, 1.0)
    with c3:
        params["close_size"] = st.slider("Conectar traços (close_size)", 1, 9, 2, 1)
    with c4:
        params["thickness"] = st.slider("Espessura da tinta", 1, 6, 2, 1)

elif mode == "sketch":
    params["sketch_strength"] = st.slider("Força do sketch", 0.0, 1.0, 0.6, 0.05)

elif mode == "pencil_color":
    c1, c2, c3 = st.columns(3)
    with c1:
        params["sketch_strength"] = st.slider("Força das linhas", 0.0, 1.0, 0.6, 0.05)
    with c2:
        params["color_soft"] = st.slider("Suavização da cor", 3, 31, 9, 2)
    with c3:
        params["saturation"] = st.slider("Saturação", 0.0, 1.5, 0.6, 0.05)

elif mode == "watercolor":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        params["smooth"] = st.slider("Suavização", 3, 31, 11, 2)
    with c2:
        params["edge_low"] = st.slider("Bordas low", 0, 200, 40, 5)
    with c3:
        params["edge_high"] = st.slider("Bordas high", 0, 300, 120, 5)
    with c4:
        params["edge_weight"] = st.slider("Força das bordas", 0.0, 1.0, 0.35, 0.05)

elif mode == "hq":
    c1, c2 = st.columns(2)
    with c1:
        params["line_thickness"] = st.slider("Espessura do contorno", 1, 6, 2, 1)
    with c2:
        params["poster_levels"] = st.slider("Níveis de cor", 2, 12, 6, 1)

elif mode == "poster_art":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        params["levels"] = st.slider("Níveis de cor", 2, 12, 5, 1)
    with c2:
        params["edge_low"] = st.slider("Bordas low", 0, 200, 50, 5)
    with c3:
        params["edge_high"] = st.slider("Bordas high", 0, 300, 150, 5)
    with c4:
        params["edge_thickness"] = st.slider("Espessura do contorno", 1, 6, 2, 1)

elif mode == "cartoon":
    st.info("Cartoon simples (sem ajustes).")

elif mode == "cartoon_clean":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        params["spatial_radius"] = st.slider("Suavização espacial", 1, 40, 10, 1)
    with c2:
        params["color_radius"] = st.slider("Suavização de cor", 1, 80, 25, 1)
    with c3:
        params["edge_low"] = st.slider("Bordas low", 0, 200, 50, 5)
    with c4:
        params["edge_high"] = st.slider("Bordas high", 0, 300, 150, 5)

elif mode == "soft_portrait":
    c1, c2 = st.columns(2)
    with c1:
        params["smooth"] = st.slider("Suavização", 3, 31, 9, 2)
    with c2:
        params["sharp"] = st.slider("Nitidez (leve)", 0.0, 0.8, 0.25, 0.05)

elif mode == "bw_fineart":
    c1, c2, c3 = st.columns(3)
    with c1:
        params["clahe_clip"] = st.slider("CLAHE clip", 0.5, 5.0, 2.0, 0.1)
    with c2:
        params["clahe_grid"] = st.slider("CLAHE grid", 4, 16, 8, 1)
    with c3:
        params["contrast"] = st.slider("Contraste", 0.8, 1.6, 1.1, 0.05)

elif mode == "glow_dreamy":
    c1, c2 = st.columns(2)
    with c1:
        params["glow_strength"] = st.slider("Força do glow", 0.0, 1.5, 0.6, 0.05)
    with c2:
        params["blur_sigma"] = st.slider("Sigma do blur", 0.5, 20.0, 6.0, 0.5)

elif mode == "edges":
    c1, c2 = st.columns(2)
    with c1:
        params["low"] = st.slider("Low", 0, 200, 50, 5)
    with c2:
        params["high"] = st.slider("High", 0, 300, 150, 5)

elif mode == "grayscale":
    st.info("Sem ajustes para este efeito.")

elif mode == "blur":
    params["ksize"] = st.slider("Kernel (ksize)", 1, 51, 9, 2)


# =========================
# Run
# =========================
current_key = (
    source,
    hashlib.sha256(image_data).hexdigest() if image_data is not None else None,
    mode,
    json.dumps(params, sort_keys=True),
)
if st.session_state.get("selection_key") != current_key:
    st.session_state.pop("result", None)
    st.session_state["selection_key"] = current_key

run = st.button("Gerar", type="primary", disabled=pil_img is None)


# =========================
# Processamento
# =========================
if run:
    img_bgr = pil_to_bgr(pil_img)

    def _process():
        # Livro de colorir PRO V2
        if mode == "colorir_linhas_pro_v2":
            fx = get_func("fx_colorir_linhas_pro_v2")

            # Remoção definitiva do recurso lento:
            # Forçamos use_grabcut=False, sem expor na UI.
            safe_params = dict(params)
            safe_params.pop("use_grabcut", None)
            safe_params.pop("grabcut_iters", None)
            safe_params.pop("fg_margin", None)

            return call_with_supported_kwargs(
                fx,
                img_bgr=img_bgr,
                use_grabcut=False,
                **safe_params
            )

        # Linhas HQ (tenta função dedicada; fallback em fx_hq)
        if mode == "colorir_linhas_hq":
            if hasattr(engine, "fx_colorir_linhas_hq"):
                fx = get_func("fx_colorir_linhas_hq")
                return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)
            fx = get_func("fx_hq")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "xdog":
            fx = get_func("fx_xdog_lineart")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "ink":
            fx = get_func("fx_ink")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "sketch":
            fx = get_func("fx_pencil_sketch")
            strength = params.get("sketch_strength", 0.6)
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, strength=strength)

        if mode == "pencil_color":
            fx = get_func("fx_pencil_color")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "watercolor":
            fx = get_func("fx_watercolor")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "hq":
            fx = get_func("fx_hq")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "poster_art":
            fx = get_func("fx_poster_art")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "cartoon":
            fx = get_func("fx_cartoon")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr)

        if mode == "cartoon_clean":
            fx = get_func("fx_cartoon_clean")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "soft_portrait":
            fx = get_func("fx_soft_portrait")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "bw_fineart":
            fx = get_func("fx_bw_fineart")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "glow_dreamy":
            fx = get_func("fx_glow_dreamy")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "edges":
            fx = get_func("fx_edges")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        if mode == "grayscale":
            fx = get_func("fx_grayscale")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr)

        if mode == "blur":
            fx = get_func("fx_blur")
            return call_with_supported_kwargs(fx, img_bgr=img_bgr, **params)

        st.error("Modo não suportado.")
        st.stop()

    try:
        started = time.perf_counter()
        with st.spinner("Transformando sua imagem…"):
            out = _process()
            out_pil = bgr_to_pil(out)
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
        st.session_state["result"] = {
            "image": out_pil,
            "png": buf.getvalue(),
            "seconds": time.perf_counter() - started,
        }
    except Exception:
        logger.exception("Falha ao aplicar o efeito %s", mode)
        st.error("Não foi possível gerar este efeito. Tente outra imagem ou ajuste os controles.")
        st.stop()

result = st.session_state.get("result")
if result is not None:
    st.success(f"Pronto! Gerado em {result['seconds']:.2f} s.")
    left, right = st.columns(2)
    with left:
        st.image(pil_img, caption="Imagem original", width="stretch")
    with right:
        st.image(result["image"], caption=f"Resultado: {effect_label}", width="stretch")

    st.download_button(
        "Baixar PNG",
        data=result["png"],
        file_name=f"resultado_{mode}.png",
        mime="image/png",
        on_click="ignore",
    )
elif pil_img is not None:
    st.image(pil_img, caption="Imagem pronta para transformar — clique em Gerar.", width=420)

with st.expander("Sobre o FotoImagem"):
    st.write("Os efeitos usam processamento de imagem com OpenCV. A demonstração foi criada por IA; os filtros não usam modelos generativos.")
    st.write("O resultado depende da iluminação e dos detalhes da imagem. Experimente ajustar os controles para encontrar o estilo desejado.")
