import cv2
import numpy as np


def _odd(n: int, min_v: int = 3) -> int:
    n = int(n)
    if n < min_v:
        n = min_v
    if n % 2 == 0:
        n += 1
    return n


# =========================
# Básicos
# =========================

def fx_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def fx_blur(img_bgr: np.ndarray, ksize: int = 9) -> np.ndarray:
    k = _odd(ksize, min_v=1)
    return cv2.GaussianBlur(img_bgr, (k, k), 0)


def fx_edges(img_bgr: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, int(low), int(high))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# =========================
# Sketch (lápis PB)
# =========================

def fx_pencil_sketch(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray

    k = int(9 + strength * 30)
    k = _odd(k, min_v=3)

    blur = cv2.GaussianBlur(inv, (k, k), 0)
    denom = np.clip(255 - blur, 1, 255)

    sketch = (gray.astype(np.float32) * 255.0) / denom.astype(np.float32)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# =========================
# Colorir linhas PRO
# =========================

def fx_colorir_linhas_pro(
    img_bgr: np.ndarray,
    block_size: int = 19,
    C: int = 7,
    close_size: int = 2,
    thickness: int = 2,
    contrast: float = 1.6,
) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # contraste + denoise preservando bordas
    gray = cv2.convertScaleAbs(gray, alpha=float(contrast), beta=(1.0 - float(contrast)) * 128)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)

    bs = _odd(block_size, min_v=3)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs,
        int(C)
    )

    # conecta falhas
    cs = max(1, int(close_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cs, cs))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # engrossa linhas pretas (dilata o inverso)
    t = max(1, int(thickness))
    if t > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
        inv = 255 - bw
        inv = cv2.dilate(inv, k2, iterations=1)
        bw = 255 - inv

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# =========================
# XDoG Line Art
# =========================

def fx_xdog_lineart(
    img_bgr: np.ndarray,
    sigma: float = 0.9,
    k: float = 1.6,
    tau: float = 0.97,
    eps: float = 0.02,
    phi: float = 16.0,
    close_size: int = 2,
) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def gaussian_blur_f(img, s):
        ks = _odd(int(6 * s + 1), min_v=3)
        return cv2.GaussianBlur(img, (ks, ks), float(s))

    g1 = gaussian_blur_f(gray, float(sigma))
    g2 = gaussian_blur_f(gray, float(sigma) * float(k))
    dog = g1 - float(tau) * g2

    xdog = np.where(dog >= float(eps), 1.0, 1.0 + np.tanh(float(phi) * (dog - float(eps))))
    xdog = np.clip(xdog, 0.0, 1.0)

    out = (1.0 - xdog) * 255.0
    out = out.astype(np.uint8)

    cs = max(1, int(close_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cs, cs))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

    out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


# =========================
# Ink (nanquim / arte-final)
# =========================

def fx_ink(
    img_bgr: np.ndarray,
    sigma: float = 0.9,
    phi: float = 18.0,
    close_size: int = 2,
    thickness: int = 2,
) -> np.ndarray:
    """
    Arte-final preta no branco: XDoG + reforço + engrossamento.
    """
    ink = fx_xdog_lineart(img_bgr, sigma=sigma, phi=phi, close_size=close_size)
    bw = cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY)

    # engrossa ainda mais se quiser
    t = max(1, int(thickness))
    if t > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
        inv = 255 - bw
        inv = cv2.dilate(inv, k2, iterations=1)
        bw = 255 - inv

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# =========================
# Pencil Color (lápis colorido)
# =========================

def fx_pencil_color(
    img_bgr: np.ndarray,
    sketch_strength: float = 0.6,
    color_soft: int = 9,
    saturation: float = 0.6,
) -> np.ndarray:
    """
    Mistura sketch PB com cor suavizada e menos saturada.
    """
    # base color suave
    k = _odd(color_soft, min_v=3)
    color = cv2.bilateralFilter(img_bgr, d=k, sigmaColor=60, sigmaSpace=60)

    # reduz saturação (HSV)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(np.clip(saturation, 0.0, 1.5))
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # sketch para bordas
    sk = fx_pencil_sketch(img_bgr, strength=sketch_strength)
    sk_gray = cv2.cvtColor(sk, cv2.COLOR_BGR2GRAY)

    # máscara de borda (quanto mais escuro no sketch, mais “desenha”)
    edge = cv2.normalize(255 - sk_gray, None, 0, 255, cv2.NORM_MINMAX)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    edge_f = (edge.astype(np.float32) / 255.0)[..., None]

    # aplica “lápis” escurecendo levemente onde tem linha
    out = color.astype(np.float32) * (1.0 - 0.45 * edge_f)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# Watercolor (aquarela)
# =========================

def fx_watercolor(
    img_bgr: np.ndarray,
    smooth: int = 11,
    edge_low: int = 40,
    edge_high: int = 120,
    edge_weight: float = 0.35,
) -> np.ndarray:
    """
    Aquarela simples: suaviza + adiciona bordas suaves.
    """
    s = _odd(smooth, min_v=3)
    base = img_bgr.copy()
    for _ in range(2):
        base = cv2.bilateralFilter(base, d=s, sigmaColor=90, sigmaSpace=90)

    # bordas suaves
    edges = fx_edges(img_bgr, low=edge_low, high=edge_high)
    e = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    e = cv2.GaussianBlur(e, (5, 5), 0)
    e_inv = 255 - e  # linhas ficam escuras (0)

    ew = float(np.clip(edge_weight, 0.0, 1.0))
    out = cv2.addWeighted(base, 1.0, cv2.cvtColor(e_inv, cv2.COLOR_GRAY2BGR), ew, 0)
    return out


# =========================
# HQ (poster + linhas)
# =========================

def fx_hq(img_bgr: np.ndarray, line_thickness: int = 2, poster_levels: int = 6) -> np.ndarray:
    levels = max(2, int(poster_levels))
    step = max(1, 256 // levels)
    poster = (img_bgr // step) * step
    poster = poster.astype(np.uint8)

    # contorno preto
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    t = max(1, int(line_thickness))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    edges = cv2.dilate(edges, kernel, iterations=1)

    mask = edges > 0
    out = poster.copy()
    out[mask] = (0, 0, 0)
    return out


# =========================
# Poster Art (pop-art simples)
# =========================

def fx_poster_art(
    img_bgr: np.ndarray,
    levels: int = 5,
    edge_low: int = 50,
    edge_high: int = 150,
    edge_thickness: int = 2,
) -> np.ndarray:
    """
    Pôster: forte redução de cores + contorno.
    """
    lv = max(2, int(levels))
    step = max(1, 256 // lv)
    poster = (img_bgr // step) * step
    poster = poster.astype(np.uint8)

    # bordas
    e = fx_edges(img_bgr, low=edge_low, high=edge_high)
    eg = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
    t = max(1, int(edge_thickness))
    if t > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
        eg = cv2.dilate(eg, kernel, iterations=1)
    mask = eg > 0

    out = poster.copy()
    out[mask] = (0, 0, 0)
    return out


# =========================
# Cartoon
# =========================

def fx_cartoon(img_bgr: np.ndarray, edge_strength: int = 9, color_smooth: int = 9) -> np.ndarray:
    edge_strength = _odd(edge_strength, min_v=3)
    color_smooth = _odd(color_smooth, min_v=3)

    color = img_bgr.copy()
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=color_smooth, sigmaColor=75, sigmaSpace=75)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_strength, 2
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_bgr)


def fx_cartoon_clean(
    img_bgr: np.ndarray,
    spatial_radius: int = 10,
    color_radius: int = 25,
    edge_low: int = 50,
    edge_high: int = 150,
) -> np.ndarray:
    """
    Cartoon mais limpo: mean shift + bordas suaves.
    """
    sr = int(np.clip(spatial_radius, 1, 40))
    cr = int(np.clip(color_radius, 1, 80))
    base = cv2.pyrMeanShiftFiltering(img_bgr, sp=sr, sr=cr)

    # bordas
    edges = fx_edges(img_bgr, low=edge_low, high=edge_high)
    eg = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    eg = cv2.GaussianBlur(eg, (3, 3), 0)
    mask = eg > 0

    out = base.copy()
    out[mask] = (0, 0, 0)
    return out


# =========================
# Soft Portrait (leve)
# =========================

def fx_soft_portrait(
    img_bgr: np.ndarray,
    smooth: int = 9,
    sharp: float = 0.25,
) -> np.ndarray:
    """
    Suaviza pele/ruído e mantém nitidez geral.
    (Não é "embelezamento agressivo", é um softening leve.)
    """
    s = _odd(smooth, min_v=3)
    smooth_img = cv2.bilateralFilter(img_bgr, d=s, sigmaColor=80, sigmaSpace=80)

    # unsharp leve
    blur = cv2.GaussianBlur(smooth_img, (0, 0), 1.2)
    out = cv2.addWeighted(smooth_img, 1.0 + float(sharp), blur, -float(sharp), 0)
    return out


# =========================
# B&W Fine Art (PB artístico)
# =========================

def fx_bw_fineart(
    img_bgr: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    contrast: float = 1.1,
) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE (contraste local)
    grid = int(np.clip(clahe_grid, 4, 16))
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(grid, grid))
    g = clahe.apply(gray)

    # contraste global leve
    g = cv2.convertScaleAbs(g, alpha=float(contrast), beta=0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# =========================
# Glow / Dreamy
# =========================

def fx_glow_dreamy(
    img_bgr: np.ndarray,
    glow_strength: float = 0.6,
    blur_sigma: float = 6.0,
) -> np.ndarray:
    """
    Glow: blur + screen/add blend leve.
    """
    gs = float(np.clip(glow_strength, 0.0, 1.5))
    sigma = float(np.clip(blur_sigma, 0.5, 20.0))

    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(img_bgr, 1.0, blur, gs, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def _auto_canny_thresholds(gray_u8: np.ndarray, sigma: float = 0.33):
    # thresholds automáticos baseados na mediana (bem robusto pra fotos reais)
    v = np.median(gray_u8)
    low = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return low, high

def _remove_small_components(bin_u8: np.ndarray, min_area: int = 60) -> np.ndarray:
    """
    Remove componentes pequenos em uma imagem binária (255=foreground).
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bin_u8 > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(bin_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= int(min_area):
            out[labels == i] = 255
    return out




def _remove_small_components_u8(bin_u8: np.ndarray, min_area: int = 120) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bin_u8 > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(bin_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[labels == i] = 255
    return out


def _grabcut_fg_mask_fast(img_bgr: np.ndarray, margin: int = 18, iters: int = 3) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = int(np.clip(margin, 1, min(h, w) // 3))
    rect = (m, m, max(1, w - 2 * m), max(1, h - 2 * m))

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, int(iters), cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    return fg


def fx_colorir_linhas_pro_v2(
    img_bgr: np.ndarray,
    # Detecção de linhas
    edges_low: int = 35,
    edges_high: int = 120,

    # Limpeza / conexão
    min_area: int = 140,
    close_size: int = 3,

    # Espessura final (livro de colorir costuma ser 1–2)
    thickness: int = 2,

    # Anti-ruído (remove textura, melhora rosto)
    bilateral_d: int = 9,
    bilateral_sigma_color: int = 60,
    bilateral_sigma_space: int = 60,

    # Reduz detalhes (pele/rosto) sem IA
    skin_suppress: bool = True,
    skin_strength: float = 0.55,   # 0..1 (quanto reduz as linhas dentro de pele)

    # Fundo (opcional)
    use_grabcut: bool = False,
    grabcut_iters: int = 3,
    fg_margin: int = 18,

    # Afinar linhas (se disponível: opencv-contrib)
    thinning: bool = True,
):
    """
    PRO V2 (Livro de Colorir):
    - Bilateral (preserva bordas, remove textura)
    - Canny + fechamento + remove componentes pequenos
    - (Opcional) reduz linhas em áreas prováveis de pele (HSV)
    - (Opcional) remove fundo via GrabCut
    - (Opcional) thinning para linhas finas (cv2.ximgproc)
    Retorna BGR com fundo branco e linhas pretas.
    """

    h, w = img_bgr.shape[:2]

    # 1) Máscara de foreground (opcional, para reduzir fundo)
    if use_grabcut:
        fg = _grabcut_fg_mask_fast(img_bgr, margin=fg_margin, iters=grabcut_iters)
        fg_ratio = float(np.count_nonzero(fg)) / float(h * w)
        if fg_ratio < 0.06:
            fg = np.full((h, w), 255, dtype=np.uint8)
        fg_dil = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    else:
        fg_dil = None

    # 2) Anti-ruído sem destruir bordas
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(
        gray,
        d=int(np.clip(bilateral_d, 3, 15)),
        sigmaColor=float(np.clip(bilateral_sigma_color, 10, 150)),
        sigmaSpace=float(np.clip(bilateral_sigma_space, 10, 150)),
    )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3) Bordas
    low = int(np.clip(edges_low, 0, 255))
    high = int(np.clip(edges_high, 0, 255))
    if high <= low:
        high = min(255, low + 40)

    edges = cv2.Canny(gray, low, high)

    # 4) Se for usar FG, corta as bordas ao redor do sujeito
    if fg_dil is not None:
        edges = cv2.bitwise_and(edges, fg_dil)

    # 5) Redução simples de detalhes no rosto/pele (HSV)
    if skin_suppress:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Faixas de pele (heurística). Funciona razoavelmente em fotos comuns.
        # Ajustáveis depois se você quiser.
        lower1 = np.array([0, 30, 40], dtype=np.uint8)
        upper1 = np.array([25, 180, 255], dtype=np.uint8)
        lower2 = np.array([160, 30, 40], dtype=np.uint8)
        upper2 = np.array([180, 180, 255], dtype=np.uint8)

        skin = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        skin = cv2.medianBlur(skin, 5)
        skin = cv2.morphologyEx(
            skin, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )

        # Reduz bordas dentro da pele: edges = edges * (1 - k*skin)
        k = float(np.clip(skin_strength, 0.0, 1.0))
        if k > 0:
            skin_mask = (skin > 0).astype(np.uint8)
            # remove uma parte das bordas em pele
            edges = cv2.bitwise_and(edges, cv2.bitwise_not((skin_mask * int(255 * k)).astype(np.uint8)))

    # 6) Conectar falhas e remover sujeira
    cs = int(np.clip(close_size, 1, 9))
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cs, cs)),
        iterations=1
    )
    edges = _remove_small_components_u8(edges, min_area=int(max(0, min_area)))

    # 7) Afinar linhas (se disponível)
    if thinning and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        edges = cv2.ximgproc.thinning(edges)

    # 8) Espessura final (após thinning)
    t = int(np.clip(thickness, 1, 8))
    if t > 1:
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)), iterations=1)

    # 9) Fundo branco e linhas pretas
    out = 255 - edges
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

