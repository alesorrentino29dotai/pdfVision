from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .pdf_extract import PageExtract, TextLine


QUESTION_START_RE = re.compile(r"^(?P<num>\d{1,3})\.\s*(?P<rest>.*\S)?$")
LESSON_HEADER_RE = re.compile(r"^Lezione\s+\d+")


@dataclass(frozen=True)
class Option:
    label: str  # A, B, C, D
    text: str
    line: TextLine


@dataclass(frozen=True)
class Question:
    qid: str  # e.g. p004_q01
    page_index: int
    number_on_page: str  # e.g. 01, 15, 163 ...
    prompt: str
    prompt_lines: list[TextLine]
    options: list[Option]  # empty for open/non-mcq
    image_index: Optional[int] = None  # 0-based index on page; set by link_questions_to_images


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def segment_questions(
    pages: Iterable[PageExtract],
    *,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> list[Question]:
    out: list[Question] = []

    for page in pages:
        if start_page is not None and page.page_index < start_page:
            continue
        if end_page is not None and page.page_index > end_page:
            continue

        # Heuristic: skip index/frontmatter pages until we hit "Lezione" or first question.
        lines = [ln for ln in page.lines if _normalize(ln.text)]
        if not lines:
            continue

        # Some pages contain headers/footers; keep everything for now and just parse starts.
        q_starts: list[tuple[int, re.Match[str]]] = []
        for idx, ln in enumerate(lines):
            t = _normalize(ln.text)
            if LESSON_HEADER_RE.match(t):
                continue
            m = QUESTION_START_RE.match(t)
            if m:
                q_starts.append((idx, m))

        for s_i, (start_idx, m) in enumerate(q_starts):
            end_idx = q_starts[s_i + 1][0] if (s_i + 1) < len(q_starts) else len(lines)

            block = lines[start_idx:end_idx]
            if not block:
                continue

            number = m.group("num")
            rest = (m.group("rest") or "").strip()

            # Build prompt: first line after the number plus subsequent lines until options.
            # In this PDF, MCQ options are just 4 consecutive lines (no A/B markers).
            payload_lines: list[TextLine] = []
            if rest:
                payload_lines.append(TextLine(page_index=block[0].page_index, text=rest, bbox=block[0].bbox))
            payload_lines.extend(block[1:])

            payload_texts = [_normalize(ln.text) for ln in payload_lines if _normalize(ln.text)]

            # Identify MCQ: if we have >= 5 lines, treat first as prompt and next 4 as options.
            # If we have exactly 4 lines, prompt might be empty (rare) -> treat as non-mcq.
            options: list[Option] = []
            prompt_lines: list[TextLine] = []
            prompt = ""

            if len(payload_lines) >= 5:
                prompt_lines = [payload_lines[0]]
                prompt = _normalize(payload_lines[0].text)
                opt_lines = payload_lines[1:5]
                labels = ["A", "B", "C", "D"]
                options = [
                    Option(label=labels[i], text=_normalize(opt_lines[i].text), line=opt_lines[i])
                    for i in range(4)
                ]

                # If additional lines exist, append them to prompt (often wraps), unless they look like headers.
                extra = payload_lines[5:]
                if extra:
                    extra_text = " ".join(_normalize(x.text) for x in extra if _normalize(x.text))
                    if extra_text:
                        prompt = _normalize(f"{prompt} {extra_text}")
                        prompt_lines = prompt_lines + extra
            else:
                # Non-mcq or malformed; keep entire block as prompt.
                prompt = _normalize(" ".join(payload_texts))
                prompt_lines = payload_lines
                options = []

            qid = f"p{page.page_index+1:03d}_q{int(number):02d}"
            out.append(
                Question(
                    qid=qid,
                    page_index=page.page_index,
                    number_on_page=number,
                    prompt=prompt,
                    prompt_lines=prompt_lines,
                    options=options,
                )
            )

    return out

