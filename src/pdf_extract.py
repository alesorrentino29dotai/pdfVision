from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import fitz  # PyMuPDF
from PIL import Image


@dataclass(frozen=True)
class TextLine:
    page_index: int
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 (page coords)


@dataclass(frozen=True)
class ImageRegion:
    """One embedded figure on a page (reading order: top-to-bottom, left-to-right)."""

    page_index: int
    index_on_page: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PageExtract:
    page_index: int
    width: float
    height: float
    lines: list[TextLine]
    has_images: bool
    image_count: int
    images: list[ImageRegion]


def _dedupe_image_bboxes(raw: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    seen: set[tuple[float, float, float, float]] = set()
    out: list[tuple[float, float, float, float]] = []
    for bb in sorted(raw, key=lambda b: (b[1], b[0])):
        key = tuple(round(x, 2) for x in bb)
        if key in seen:
            continue
        seen.add(key)
        out.append(bb)
    return out


def extract_text_lines(doc: fitz.Document, page_index: int) -> PageExtract:
    page = doc.load_page(page_index)
    w, h = float(page.rect.width), float(page.rect.height)

    text_dict = page.get_text("dict")
    lines: list[TextLine] = []
    image_blocks = 0
    image_bboxes: list[tuple[float, float, float, float]] = []

    for block in text_dict.get("blocks", []):
        # type==0 is text, type==1 is image (we handle rendering separately)
        if block.get("type") == 1:
            image_blocks += 1
            bb = block.get("bbox")
            if bb and len(bb) == 4:
                image_bboxes.append((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])))
            continue
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
            lines.append(TextLine(page_index=page_index, text=text, bbox=bbox))  # type: ignore[arg-type]

    # Stable sort: top-to-bottom then left-to-right.
    lines.sort(key=lambda l: (round(l.bbox[1], 1), round(l.bbox[0], 1)))
    # Alternative signal: embedded images list (some PDFs may not expose image blocks).
    embedded_images = len(page.get_images(full=True))
    image_count = max(image_blocks, embedded_images)
    bboxes = _dedupe_image_bboxes(image_bboxes)
    images = [
        ImageRegion(page_index=page_index, index_on_page=i, bbox=bb)
        for i, bb in enumerate(bboxes)
    ]
    return PageExtract(
        page_index=page_index,
        width=w,
        height=h,
        lines=lines,
        has_images=image_count > 0,
        image_count=image_count,
        images=images,
    )


def iter_pages_text_lines(pdf_path: str | Path) -> Iterator[PageExtract]:
    pdf_path = Path(pdf_path)
    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            yield extract_text_lines(doc, i)


def render_page_png(
    doc: fitz.Document,
    page_index: int,
    dpi: int = 150,
) -> Image.Image:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_region_png(
    doc: fitz.Document,
    page_index: int,
    bbox: tuple[float, float, float, float],
    dpi: int = 150,
) -> Image.Image:
    """Render a PDF page rectangle (figure) at the given DPI."""
    page = doc.load_page(page_index)
    clip = fitz.Rect(bbox)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def export_image_region_pngs(
    pdf_path: str | Path,
    out_dir: str | Path,
    extracts: Iterable[PageExtract],
    dpi: int = 150,
) -> list[Path]:
    """Write one PNG per extracted image region: page_NNN_img_MM.png"""
    pdf_path = Path(pdf_path)
    out_dir = ensure_dir(out_dir)
    out_paths: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for ex in extracts:
            for im in ex.images:
                img = render_region_png(doc, ex.page_index, im.bbox, dpi=dpi)
                out_path = out_dir / f"page_{ex.page_index + 1:03d}_img_{im.index_on_page + 1:02d}.png"
                img.save(out_path)
                out_paths.append(out_path)
    return out_paths


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_page_renders(
    pdf_path: str | Path,
    out_dir: str | Path,
    pages: Optional[Iterable[int]] = None,
    dpi: int = 150,
) -> list[Path]:
    pdf_path = Path(pdf_path)
    out_dir = ensure_dir(out_dir)
    out_paths: list[Path] = []

    with fitz.open(pdf_path) as doc:
        page_indices = list(pages) if pages is not None else list(range(doc.page_count))
        for i in page_indices:
            img = render_page_png(doc, i, dpi=dpi)
            out_path = out_dir / f"page_{i+1:03d}.png"
            img.save(out_path)
            out_paths.append(out_path)
    return out_paths

