from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional

import fitz  # PyMuPDF

from .question_segment import Question
from .solver.base import AnswerLabel, OpenSolveResult, SolveResult


AnnotateMode = Literal["highlight_correct", "strike_wrong"]


@dataclass(frozen=True)
class AnnotateStats:
    highlighted: int
    fallback_notes: int


def _option_rect(question: Question, answer: AnswerLabel) -> Optional[fitz.Rect]:
    for opt in question.options:
        if opt.label == answer:
            return fitz.Rect(opt.line.bbox)
    return None


def _clip_expand(rect: fitz.Rect, page: fitz.Page, pad: float = 2.0) -> fitz.Rect:
    r = fitz.Rect(rect)
    r.x0 -= pad
    r.y0 -= pad
    r.x1 += pad
    r.y1 += pad
    return r & page.rect


def _fallback_note(page: fitz.Page, q: Question, res: SolveResult, suffix: str = "") -> None:
    if q.prompt_lines:
        x0, y0, _, _ = q.prompt_lines[0].bbox
        pos = fitz.Point(x0, max(0, y0 - 10))
    else:
        pos = fitz.Point(36, 36)
    note = page.add_text_annot(pos, f"Correct: {res.answer} (conf {res.confidence:.2f}){suffix}")
    note.update()


def _draw_open_answer_inline(page: fitz.Page, q: Question, open_res: OpenSolveResult) -> bool:
    if q.prompt_lines:
        x0 = min(ln.bbox[0] for ln in q.prompt_lines)
        y1 = max(ln.bbox[3] for ln in q.prompt_lines)
    else:
        x0, y1 = 36.0, 50.0

    margin = 36.0
    start = fitz.Point(max(margin, x0), min(page.rect.height - 30, y1 + 8))
    box = fitz.Rect(start.x, start.y, page.rect.width - margin, min(page.rect.height - 6, start.y + 90))
    if box.height < 14 or box.width < 50:
        return False

    txt = f"Risposta: {open_res.answer_text}"
    spare = page.insert_textbox(
        box,
        txt,
        fontsize=9.4,
        fontname="helv",
        color=(0.05, 0.50, 0.05),
        align=fitz.TEXT_ALIGN_LEFT,
        overlay=True,
    )
    return spare >= -1.0


def _square_annot(
    page: fitz.Page,
    r: fitz.Rect,
    *,
    stroke: tuple[float, float, float],
    fill: tuple[float, float, float],
    border: float = 1.25,
    opacity: float = 0.92,
) -> bool:
    """Annotazione Square (add_rect_annot): layer sopra il contenuto pagina, ben visibile nei viewer."""
    if r.is_empty:
        return False
    a = page.add_rect_annot(r)
    a.set_colors(stroke=stroke, fill=fill)
    a.set_border(width=border)
    a.set_opacity(opacity)
    a.update()
    return True


def annotate_pdf(
    *,
    pdf_in: str | Path,
    pdf_out: str | Path,
    questions: list[Question],
    results: Mapping[str, SolveResult],  # qid -> result
    open_results: Optional[Mapping[str, OpenSolveResult]] = None,  # qid -> open answer
    mode: AnnotateMode = "highlight_correct",
) -> AnnotateStats:
    """
    Marca le opzioni MCQ con annotazioni PDF **Square** e scrive le risposte aperte inline.
    Le square annotations stanno nel layer annotazioni, sopra al testo.
    """
    pdf_in = Path(pdf_in)
    pdf_out = Path(pdf_out)

    highlighted = 0
    fallback_notes = 0

    doc = fitz.open(pdf_in)
    wrapped: set[int] = set()
    open_results = open_results or {}
    try:
        for q in questions:
            page = doc.load_page(q.page_index)
            if q.page_index not in wrapped:
                page.wrap_contents()
                wrapped.add(q.page_index)

            # Open-ended question rendering.
            open_res = open_results.get(q.qid)
            if open_res is not None:
                if not _draw_open_answer_inline(page, q, open_res):
                    pos = fitz.Point(36, max(20, page.rect.height - 40))
                    note = page.add_text_annot(pos, f"Risposta: {open_res.answer_text}")
                    note.update()
                    fallback_notes += 1
                continue

            # MCQ rendering.
            res = results.get(q.qid)
            if not res:
                continue

            rect = _option_rect(q, res.answer)
            if rect is not None and rect.is_valid:
                if mode == "highlight_correct":
                    r = _clip_expand(rect, page)
                    if r.is_empty:
                        _fallback_note(page, q, res, " [bbox]")
                        fallback_notes += 1
                    else:
                        ok = _square_annot(
                            page,
                            r,
                            stroke=(0.0, 0.55, 0.05),
                            fill=(0.55, 1.0, 0.45),
                            border=1.4,
                            opacity=0.45,
                        )
                        if ok:
                            highlighted += 1
                        else:
                            _fallback_note(page, q, res, " [annot]")
                            fallback_notes += 1
                else:
                    drew_any = False
                    for opt in q.options:
                        r0 = fitz.Rect(opt.line.bbox)
                        if not r0.is_valid:
                            continue
                        r = _clip_expand(r0, page)
                        if r.is_empty:
                            continue
                        if opt.label == res.answer:
                            if _square_annot(
                                page,
                                r,
                                stroke=(0.0, 0.55, 0.05),
                                fill=(0.55, 1.0, 0.45),
                                border=1.4,
                                opacity=0.45,
                            ):
                                drew_any = True
                        else:
                            if _square_annot(
                                page,
                                r,
                                stroke=(0.75, 0.12, 0.1),
                                fill=(1.0, 0.82, 0.8),
                                border=1.0,
                                opacity=0.88,
                            ):
                                drew_any = True
                            ym = (r.y0 + r.y1) * 0.5
                            page.draw_line(
                                fitz.Point(r.x0 + 0.5, ym),
                                fitz.Point(r.x1 - 0.5, ym),
                                color=(0.82, 0.05, 0.05),
                                width=1.35,
                                overlay=True,
                            )
                    if drew_any:
                        highlighted += 1
                    else:
                        _fallback_note(page, q, res, " [bbox]")
                        fallback_notes += 1
            else:
                _fallback_note(page, q, res)
                fallback_notes += 1

        doc.save(pdf_out, garbage=4, deflate=True)
    finally:
        doc.close()

    return AnnotateStats(highlighted=highlighted, fallback_notes=fallback_notes)
