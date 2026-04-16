from __future__ import annotations

from dataclasses import replace

from .pdf_extract import ImageRegion, PageExtract
from .question_segment import Question


def _prompt_top_y(q: Question) -> float | None:
    if q.prompt_lines:
        return min(ln.bbox[1] for ln in q.prompt_lines)
    if q.options:
        return min(opt.line.bbox[1] for opt in q.options)
    return None


def link_questions_to_images(questions: list[Question], extracts: list[PageExtract]) -> list[Question]:
    """
    Assign each question to the figure it most likely refers to (same page).

    Heuristic: prefer the last figure whose bottom edge lies above the top of the question text;
    if none, a single figure on the page wins; otherwise pick the vertically nearest figure.
    """
    by_page: dict[int, PageExtract] = {e.page_index: e for e in extracts}
    out: list[Question] = []
    tol = 4.0

    for q in questions:
        ex = by_page.get(q.page_index)
        if not ex or not ex.images:
            out.append(replace(q, image_index=None))
            continue

        imgs = sorted(ex.images, key=lambda r: (r.bbox[1], r.bbox[0]))
        prompt_y0 = _prompt_top_y(q)
        if prompt_y0 is None:
            out.append(replace(q, image_index=0 if len(imgs) == 1 else None))
            continue

        above = [im for im in imgs if im.bbox[3] <= prompt_y0 + tol]
        if above:
            chosen = max(above, key=lambda im: im.bbox[3])
            out.append(replace(q, image_index=chosen.index_on_page))
        elif len(imgs) == 1:
            out.append(replace(q, image_index=0))
        else:

            def dist(im: ImageRegion) -> float:
                ic = (im.bbox[1] + im.bbox[3]) / 2.0
                return abs(ic - prompt_y0)

            chosen = min(imgs, key=dist)
            out.append(replace(q, image_index=chosen.index_on_page))

    return out
