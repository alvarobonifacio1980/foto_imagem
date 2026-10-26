"""Decode uploads and keep image processing within predictable memory limits."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_INPUT_PIXELS = 16_000_000
MAX_PROCESSING_EDGE = 1600


def prepare_image(data: bytes) -> tuple[Image.Image, bool]:
    """Return an oriented RGB image and whether it was resized."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Escolha um arquivo de até 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Use uma imagem JPG ou PNG.")
            if source.width * source.height > MAX_INPUT_PIXELS:
                raise ValueError("Escolha uma imagem com até 16 megapixels.")
            if min(source.size) < 16:
                raise ValueError("A imagem deve ter pelo menos 16 pixels em cada lado.")
            # Resize before making RGB/alpha copies, then apply phone orientation.
            resized = max(source.size) > MAX_PROCESSING_EDGE
            source.thumbnail((MAX_PROCESSING_EDGE, MAX_PROCESSING_EDGE), Image.Resampling.LANCZOS)
            oriented = ImageOps.exif_transpose(source)
            if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = oriented.convert("RGB")
            return image, resized
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Não foi possível abrir a imagem. Envie um JPG ou PNG válido.") from exc
