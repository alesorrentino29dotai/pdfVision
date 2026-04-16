from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.annotate_pdf import annotate_pdf
from src.image_link import link_questions_to_images
from src.pdf_extract import PageExtract, export_image_region_pngs, export_page_renders, extract_text_lines
from src.question_segment import Question, segment_questions
from src.solver.api_solver import ApiSolver
from src.solver.base import SolveInput, SolveResult, Solver
from src.solver.gemini_solver import GeminiSolver
from src.solver.local_solver import LocalOllamaSolver


def _parse_pages(pages: Optional[str]) -> Optional[list[int]]:
    """
    Returns 0-based page indices.
    Accepts:
      - "1-3"
      - "5"
      - "1,3,5-7"
    """
    if not pages:
        return None
    out: set[int] = set()
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            for p in range(start, end + 1):
                out.add(p - 1)
        else:
            out.add(int(part) - 1)
    return sorted(out)


def _pages_label_1based(indices: Optional[list[int]]) -> str:
    if indices is None:
        return "all"
    return ",".join(str(i + 1) for i in indices)


def _build_solver(
    backend: str,
    model: str,
    api_key: Optional[str],
    ollama_host: Optional[str],
    gemini_key: Optional[str],
    verbose: bool,
) -> Solver:
    if backend == "api":
        return ApiSolver(model=model, api_key=api_key)
    if backend == "local":
        return LocalOllamaSolver(model=model, host=ollama_host)
    if backend == "gemini":
        return GeminiSolver(model=model, api_key=gemini_key, verbose=verbose)
    raise ValueError(f"Unknown backend: {backend}")


def run_from_answers_json(args: argparse.Namespace) -> int:
    """
    Riprende pdf_in + risposte da un answers.json e rigenera solo il PDF (niente inferenza).
    Serve a rifare l'annotazione dopo aver editato il JSON o cambiato --mode.
    """
    src = Path(args.from_answers)
    payload = json.loads(src.read_text(encoding="utf-8"))
    pdf_in = Path(args.pdf_in) if args.pdf_in else Path(payload["pdf_in"])
    if not pdf_in.is_file():
        raise SystemExit(f"PDF non trovato: {pdf_in}")

    pdf_out = Path(args.pdf_out)
    answers_json = Path(args.answers_json) if args.answers_json else pdf_out.with_suffix(".answers.json")
    page_indices = _parse_pages(args.pages if args.pages else payload.get("pages"))

    render_dir = Path(args.debug_dir) if args.debug_dir else (pdf_out.parent / "debug_renders")
    if page_indices is not None:
        export_page_renders(pdf_in, render_dir, pages=page_indices, dpi=args.dpi)
    else:
        export_page_renders(pdf_in, render_dir, pages=None, dpi=args.dpi)

    with fitz.open(pdf_in) as doc:
        extracts: list[PageExtract] = []
        indices = page_indices if page_indices is not None else list(range(doc.page_count))
        for i in indices:
            extracts.append(extract_text_lines(doc, i))

    export_image_region_pngs(pdf_in, render_dir, extracts, dpi=args.dpi)
    questions = link_questions_to_images(segment_questions(extracts), extracts)

    results: dict[str, SolveResult] = {}
    for qid, row in payload.get("answers", {}).items():
        try:
            results[qid] = SolveResult.model_validate(
                {
                    "answer": row["answer"],
                    "confidence": row["confidence"],
                    "rationale": row.get("rationale", ""),
                }
            )
        except Exception as e:
            raise SystemExit(f"Voce answers non valida per {qid!r}: {e}") from e

    mode = args.mode
    if args.verbose:
        print("[run] === from-answers -> PDF (niente inferenza) ===")
        print(f"[run] json: {src.resolve()}")
        print(f"[run] pdf_in: {pdf_in.resolve()}")
        print(f"[run] pages (1-based): {_pages_label_1based(page_indices)}")
        n_mcq = sum(1 for q in questions if len(q.options) == 4)
        matched = sum(1 for q in questions if len(q.options) == 4 and q.qid in results)
        print(f"[run] domande: {len(questions)} | MCQ: {n_mcq} | risposte JSON usate su MCQ: {matched}/{n_mcq}")
        print(f"[run] chiavi in answers: {len(payload.get('answers', {}))}")
        print(f"[run] mode annotazione: {mode}")

    stats = annotate_pdf(pdf_in=pdf_in, pdf_out=pdf_out, questions=questions, results=results, mode=mode)

    out_payload = {**payload, "pdf_out": str(pdf_out), "mode": mode, "stats": {"highlighted": stats.highlighted, "fallback_notes": stats.fallback_notes}}
    answers_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {pdf_out}")
    print(f"Wrote: {answers_json}")
    print(f"Highlighted: {stats.highlighted} (fallback notes: {stats.fallback_notes})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        epilog=(
            "Esempio debug una pagina:  py run.py --in quiz.pdf --out out.pdf --pages 5 --verbose\n"
            "PDF solo da JSON:          py run.py --from-answers quiz.answers.json --out out.pdf --verbose"
        ),
    )
    ap.add_argument("--in", dest="pdf_in", default=None, help="Input PDF path (optional se usi --from-answers)")
    ap.add_argument("--out", dest="pdf_out", required=True, help="Output annotated PDF path")
    ap.add_argument("--answers", dest="answers_json", default=None, help="answers.json output path")
    ap.add_argument(
        "--from-answers",
        dest="from_answers",
        default=None,
        metavar="PATH",
        help="Carica risposte da un answers.json e genera il PDF senza inferenza (richiede gli stessi segmenti sul PDF).",
    )
    ap.add_argument("--backend", choices=["api", "local", "gemini"], default="gemini")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Model name for selected backend")
    ap.add_argument("--api-key", default=None, help="API key (optional; else env)")
    ap.add_argument("--gemini-key", default=None, help="Gemini API key (optional; else GEMINI_API_KEY/GOOGLE_API_KEY env)")
    ap.add_argument("--ollama-host", default=None, help="Ollama host, e.g. http://localhost:11434")
    ap.add_argument("--pages", default=None, help='Pages 1-based, e.g. "1-3" or "1,4,7-9"')
    ap.add_argument("--dpi", type=int, default=150, help="DPI for page renders sent to solver")
    ap.add_argument(
        "--mode",
        choices=["highlight_correct", "strike_wrong"],
        default="highlight_correct",
        help="highlight_correct=solo risposta giusta in verde; strike_wrong=giusta in verde + sbagliate barrate.",
    )
    ap.add_argument("--debug-dir", default=None, help="Optional directory to store page renders")
    ap.add_argument("--skip-solve", action="store_true", help="Run extract/segment/render only (no solving/annotations).")
    ap.add_argument("--verbose", action="store_true", help="Print progress during inference/annotation.")
    ap.add_argument(
        "--image-policy",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to send visuals to the model: auto=figure crop when linked, else full page if the page has images.",
    )
    ap.add_argument(
        "--min-request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds to wait between solver requests (useful for free-tier rate limits).",
    )
    args = ap.parse_args()

    if args.from_answers:
        return run_from_answers_json(args)

    if not args.pdf_in:
        ap.error("specificare --in oppure usare --from-answers")

    pdf_in = Path(args.pdf_in)
    pdf_out = Path(args.pdf_out)
    answers_json = Path(args.answers_json) if args.answers_json else pdf_out.with_suffix(".answers.json")

    page_indices = _parse_pages(args.pages)

    # Sensible default for Gemini free tier (often 5 req/min/model).
    if args.backend == "gemini" and args.min_request_interval <= 0:
        args.min_request_interval = 13.0

    solver = (
        None
        if args.skip_solve
        else _build_solver(args.backend, args.model, args.api_key, args.ollama_host, args.gemini_key, args.verbose)
    )

    # Render pages to PNG (for questions with images); store in debug dir or temp-like folder next to output.
    render_dir = Path(args.debug_dir) if args.debug_dir else (pdf_out.parent / "debug_renders")
    if args.verbose:
        print("[run] === pdfVision pipeline (estrazione + inferenza) ===")
        print(f"[run] pdf_in:  {pdf_in.resolve()}")
        print(f"[run] pdf_out: {pdf_out.resolve()}")
        print(f"[run] answers: {answers_json.resolve()}")
        print(f"[run] pagine (1-based): {_pages_label_1based(page_indices)}")
        print(f"[run] render_dir: {render_dir.resolve()}")
        print(
            f"[run] dpi={args.dpi} image_policy={args.image_policy} mode={args.mode} "
            f"skip_solve={args.skip_solve}"
        )
        print(f"[run] backend={args.backend} model={args.model}")
    if page_indices is not None:
        export_page_renders(pdf_in, render_dir, pages=page_indices, dpi=args.dpi)
    else:
        export_page_renders(pdf_in, render_dir, pages=None, dpi=args.dpi)

    # Extract structured lines and segment questions.
    with fitz.open(pdf_in) as doc:
        extracts: list[PageExtract] = []
        indices = page_indices if page_indices is not None else list(range(doc.page_count))
        for i in indices:
            extracts.append(extract_text_lines(doc, i))

    export_image_region_pngs(pdf_in, render_dir, extracts, dpi=args.dpi)

    questions = link_questions_to_images(segment_questions(extracts), extracts)
    page_has_images = {p.page_index: p.has_images for p in extracts}
    page_image_count = {p.page_index: p.image_count for p in extracts}

    if args.verbose:
        print(f"[run] domande segmentate: {len(questions)} (MCQ 4 opzioni: {sum(1 for q in questions if len(q.options) == 4)})")

    def resolve_solve_image(q: Question) -> tuple[str | None, str]:
        """Pick PNG path for the solver: per-image crop when linked, else full page when policy allows."""
        page_png = render_dir / f"page_{q.page_index+1:03d}.png"
        crop_png: Path | None = None
        if q.image_index is not None:
            p = render_dir / f"page_{q.page_index+1:03d}_img_{q.image_index+1:02d}.png"
            if p.exists():
                crop_png = p

        if args.image_policy == "never":
            return None, "none"
        if crop_png is not None:
            return str(crop_png), "region"
        if not page_png.exists():
            return None, "none"
        if args.image_policy == "always":
            return str(page_png), "full_page"
        if args.image_policy == "auto" and page_has_images.get(q.page_index, False):
            return str(page_png), "full_page"
        return None, "none"

    results: dict[str, SolveResult] = {}
    if solver is not None:
        mcq_questions = [q for q in questions if len(q.options) == 4]
        if args.verbose:
            print(f"[run] total questions: {len(questions)} | MCQ: {len(mcq_questions)}")
            # Per-page summary (for the pages we processed)
            pages_sorted = sorted(page_has_images.keys())
            by_pi = {e.page_index: e for e in extracts}
            for pi in pages_sorted:
                ex = by_pi[pi]
                print(
                    f"[run] page {pi+1}: regions={len(ex.images)} "
                    f"(embedded_count={page_image_count.get(pi, 0)})"
                )
        last_call_t = 0.0
        for idx, q in enumerate(mcq_questions, start=1):
            if args.min_request_interval > 0:
                now = time.time()
                wait_s = (last_call_t + args.min_request_interval) - now
                if wait_s > 0:
                    if args.verbose:
                        print(f"[run] throttling: sleep {wait_s:.2f}s")
                    time.sleep(wait_s)

            vis_path, vis_kind = resolve_solve_image(q)
            inp = SolveInput(
                qid=q.qid,
                prompt=q.prompt,
                options=[opt.text for opt in q.options],
                page_image_png_path=vis_path,
            )
            if args.verbose:
                img_note = "no"
                if vis_path:
                    ii = f", figure={q.image_index+1}" if q.image_index is not None and vis_kind == "region" else ""
                    img_note = f"{vis_kind}{ii}"
                print(
                    f"[run] solve {idx}/{len(mcq_questions)}: {q.qid} "
                    f"(page {q.page_index+1}, visual={img_note})"
                )
                pr = q.prompt
                if len(pr) > 400:
                    pr = pr[:400] + "…"
                print(f"[run]   testo domanda: {pr!r}")
                for opt in q.options:
                    ot = opt.text
                    if len(ot) > 160:
                        ot = ot[:160] + "…"
                    print(f"[run]   {opt.label}) {ot!r}")
                if vis_path:
                    print(f"[run]   file immagine solver: {vis_path}")
                t0 = time.time()
            results[q.qid] = solver.solve(inp)
            if args.verbose:
                dt = time.time() - t0
                r = results[q.qid]
                print(f"[run] done {q.qid}: answer={r.answer} conf={r.confidence:.2f} ({dt:.2f}s)")
            last_call_t = time.time()

    # Annotate and write outputs.
    if results:
        stats = annotate_pdf(pdf_in=pdf_in, pdf_out=pdf_out, questions=questions, results=results, mode=args.mode)
    else:
        pdf_out.write_bytes(pdf_in.read_bytes())
        stats = type("Tmp", (), {"highlighted": 0, "fallback_notes": 0})()

    mcq_by_qid = {q.qid: q for q in questions if len(q.options) == 4}
    answers_out: dict[str, dict] = {}
    for qid, res in results.items():
        row = res.model_dump()
        qm = mcq_by_qid.get(qid)
        if qm is not None:
            _vp, vk = resolve_solve_image(qm)
            row["page"] = qm.page_index + 1
            row["image_index_on_page"] = qm.image_index
            row["visual_kind"] = vk
        answers_out[qid] = row

    payload = {
        "pdf_in": str(pdf_in),
        "pdf_out": str(pdf_out),
        "mode": args.mode,
        "backend": args.backend,
        "model": args.model,
        "pages": args.pages,
        "stats": {"highlighted": stats.highlighted, "fallback_notes": stats.fallback_notes},
        "question_count": len(questions),
        "mcq_count": sum(1 for q in questions if len(q.options) == 4),
        "answers": answers_out,
    }
    answers_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {pdf_out}")
    print(f"Wrote: {answers_json}")
    print(f"Highlighted: {stats.highlighted} (fallback notes: {stats.fallback_notes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

