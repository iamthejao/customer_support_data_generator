"""Figure images in the forms each renderer needs.

python-docx cannot embed SVG, so SVG figures are rasterised to PNG with the
same Typst compiler that renders the PDFs (no extra converter, system fonts
ignored so the bytes do not depend on the machine).
"""

from __future__ import annotations

import typst

from csfd.documents.ir import Asset

# Width of the rasterised image and its resolution: an A4 text column at print quality.
_PNG_WIDTH_CM = 16
_PNG_PPI = 200

_EXTENSIONS = {"image/svg+xml": "svg", "image/png": "png", "image/jpeg": "jpg"}


def extension(asset: Asset) -> str:
    return _EXTENSIONS[asset.media_type]


def svg_to_png(svg: bytes) -> bytes:
    """Rasterise an SVG to PNG, deterministically."""
    source = (
        "#set page(width: auto, height: auto, margin: 0pt)\n"
        f'#image(bytes(sys.inputs.svg), format: "svg", width: {_PNG_WIDTH_CM}cm)\n'
    )
    compiler = typst.Compiler(ignore_system_fonts=True, sys_inputs={"svg": svg.decode("utf-8")})
    out = compiler.compile(source.encode("utf-8"), format="png", ppi=_PNG_PPI)
    assert isinstance(out, bytes)
    return out


def raster_bytes(asset: Asset) -> tuple[bytes, str]:
    """Image bytes and extension a DOCX can embed (SVG becomes PNG)."""
    if asset.media_type == "image/svg+xml":
        return svg_to_png(asset.content), "png"
    return asset.content, extension(asset)
