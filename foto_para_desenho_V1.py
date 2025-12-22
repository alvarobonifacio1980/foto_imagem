import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


# =========================================================
# Utilidades
# =========================================================

def resize_to_fit(img: np.ndarray, max_width: int = 1600, max_height: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width and h <= max_height:
        return img
    scale = min(max_width / w, max_height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Não consegui ler a imagem: {path}")
    return img


def ensure_out_path(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# Estilos / Filtros
# =========================================================

def fx_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def fx_blur(img_bgr: np.ndarray, ksize: int = 9) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)


def fx_edges(img_bgr: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def fx_pencil_sketch(img_bgr: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Pencil sketch PB (color dodge):
      sketch = gray / (255 - blur(inv(gray)))
    """
    strength = float(np.clip(strength, 0.0, 1.0))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray

    k = int(9 + strength * 30)
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(inv, (k, k), 0)

    denom = np.clip(255 - blur, 1, 255)
    sketch = (gray.astype(np.float32) * 255.0) / denom.astype(np.float32)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def fx_sketch_clean(
    img_bgr: np.ndarray,
    sketch_strength: float = 0.6,
    clean_level: float = 0.25,
    edges_low: int = 40,
    edges_high: int = 140,
    edge_weight: float = 0.8,
) -> np.ndarray:
    """
    Sketch mais limpo:
      - gera sketch base (PB)
      - "limpa" preenchimento: empurra tons claros para branco (threshold suave)
      - extrai bordas (Canny) e mistura para reforçar contorno

    Parâmetros:
      clean_level: 0..1 (quanto limpar). Maior => mais branco (menos preenchimento).
      edge_weight: 0..1 (força das bordas). Maior => contorno mais presente.
    """
    clean_level = float(np.clip(clean_level, 0.0, 1.0))
    edge_weight = float(np.clip(edge_weight, 0.0, 1.0))

    # 1) sketch base
    sketch = fx_pencil_sketch(img_bgr, strength=sketch_strength)
    s = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)

    # 2) limpar preenchimento:
    #    aumenta contraste para jogar cinzas para branco, mantendo escuros
    #    alpha maior => mais "limpo"
    alpha = 1.0 + clean_level * 2.2     # 1.0 .. 3.2
    beta = clean_level * 35.0           # 0 .. 35
    s_clean = cv2.convertScaleAbs(s, alpha=alpha, beta=beta)

    # leve blur para suavizar granulação
    s_clean = cv2.GaussianBlur(s_clean, (3, 3), 0)

    # 3) bordas
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    e = cv2.Canny(gray, int(edges_low), int(edges_high))

    # transforma borda em "linhas pretas no branco"
    e_inv = 255 - e  # 255 no fundo, 0 nas bordas

    # 4) mistura: escurece onde tem borda
    #    min() preserva brancos e desenha preto onde tem linha
    combined = cv2.min(s_clean, cv2.addWeighted(s_clean, 1.0 - edge_weight, e_inv, edge_weight, 0))

    # opcional: binarização leve pra ficar mais "livro de colorir"
    # combined = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Conversor de fotos em desenho (sketch clean incluído)")

    p.add_argument("--input", required=True)
    p.add_argument("--output", default="")
    p.add_argument(
        "--mode",
        required=True,
        choices=["sketch", "sketch_clean", "edges", "grayscale", "blur"],
    )

    # parâmetros comuns
    p.add_argument("--show", action="store_true")

    # sketch
    p.add_argument("--sketch_strength", type=float, default=0.6)

    # sketch_clean
    p.add_argument("--clean_level", type=float, default=0.25)
    p.add_argument("--edge_weight", type=float, default=0.8)
    p.add_argument("--edges_low", type=int, default=40)
    p.add_argument("--edges_high", type=int, default=140)

    # blur
    p.add_argument("--blur_ksize", type=int, default=9)

    return p.parse_args()


def build_default_output(input_path: Path, mode: str) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    return out_dir / f"{input_path.stem}_{mode}.png"


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Entrada não encontrada: {in_path}")

    img = read_image(in_path)

    if args.mode == "sketch":
        out = fx_pencil_sketch(img, strength=args.sketch_strength)

    elif args.mode == "sketch_clean":
        out = fx_sketch_clean(
            img,
            sketch_strength=args.sketch_strength,
            clean_level=args.clean_level,
            edges_low=args.edges_low,
            edges_high=args.edges_high,
            edge_weight=args.edge_weight,
        )

    elif args.mode == "edges":
        out = fx_edges(img, low=args.edges_low, high=args.edges_high)

    elif args.mode == "grayscale":
        out = fx_grayscale(img)

    elif args.mode == "blur":
        out = fx_blur(img, ksize=args.blur_ksize)

    else:
        raise ValueError("Modo inválido")

    out_path = Path(args.output) if args.output else build_default_output(in_path, args.mode)
    ensure_out_path(out_path)

    if not cv2.imwrite(str(out_path), out):
        raise RuntimeError(f"Falha ao salvar: {out_path}")

    print(f"[OK] Salvo em: {out_path.resolve()}")

    if args.show:
        preview = resize_to_fit(out, 1600, 900)
        cv2.imshow(f"Preview - {args.mode}", preview)
        print("Pressione ESC para fechar.")
        while True:
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Modo teste. Exemplos:")
        print('python foto_para_desenho.py --input "minha_foto.jpg" --mode sketch --sketch_strength 0.6 --show')
        print('python foto_para_desenho.py --input "minha_foto.jpg" --mode sketch_clean --sketch_strength 0.6 --clean_level 0.35 --edge_weight 0.9 --show')
    else:
        main()
