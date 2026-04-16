from __future__ import annotations

import json
import re
from typing import Optional

from ollama import Client as OllamaClient
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import OpenSolveResult, SolveInput, SolveResult, Solver


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F]+")


def _sanitize_jsonish_text(s: str) -> str:
    """
    Make model output JSON-parseable by removing ASCII control chars.

    Some Ollama models occasionally emit unescaped newlines inside JSON strings or
    prepend/append extra text. Replacing control chars with spaces avoids hard failures.
    """
    return _CONTROL_CHARS_RE.sub(" ", s).strip()


def _extract_json_for_solve(raw: str) -> str:
    """
    Ollama models often wrap JSON in ```json ... ``` or add prose; strip to a single JSON object.
    """
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1]
    return s


def _parse_solve_result_from_text(raw: str) -> SolveResult:
    """
    Best-effort parser for Ollama outputs.

    Strategy:
    - strip fences / isolate {...}
    - remove control chars
    - try strict JSON
    - fallback: regex-extract answer/confidence, keep a short rationale
    """
    jsonish = _sanitize_jsonish_text(_extract_json_for_solve(raw))

    # Strict JSON parse first.
    try:
        obj = json.loads(jsonish)
        return SolveResult.model_validate(obj)
    except Exception:
        pass

    # Fallback: extract fields even if formatting is broken.
    m_ans = re.search(r'(?:"answer"\s*:\s*")(?P<a>[ABCD])"', jsonish, flags=re.IGNORECASE)
    ans = (m_ans.group("a").upper() if m_ans else "A")

    m_conf = re.search(r'(?:"confidence"\s*:\s*)(?P<c>\d+(?:\.\d+)?)', jsonish, flags=re.IGNORECASE)
    conf = float(m_conf.group("c")) if m_conf else 0.5
    conf = max(0.0, min(1.0, conf))

    # Try to keep something useful as rationale (trim aggressively).
    rationale = jsonish
    m_rat = re.search(r'(?:"rationale"\s*:\s*")(?P<r>.*?)(?:"\s*[,}])', jsonish, flags=re.IGNORECASE)
    if m_rat:
        rationale = m_rat.group("r")
    rationale = rationale.strip()
    if len(rationale) > 400:
        rationale = rationale[:400] + "…"

    return SolveResult(answer=ans, confidence=conf, rationale=rationale)


def _parse_open_result_from_text(raw: str) -> OpenSolveResult:
    jsonish = _sanitize_jsonish_text(_extract_json_for_solve(raw))
    try:
        obj = json.loads(jsonish)
        return OpenSolveResult.model_validate(obj)
    except Exception:
        pass

    m_ans = re.search(
        r'(?:"answer_text"\s*:\s*")(?P<a>.*?)(?:"\s*[,}])',
        jsonish,
        flags=re.IGNORECASE,
    )
    answer_text = m_ans.group("a").strip() if m_ans else jsonish
    if len(answer_text) > 500:
        answer_text = answer_text[:500] + "…"

    m_conf = re.search(r'(?:"confidence"\s*:\s*)(?P<c>\d+(?:\.\d+)?)', jsonish, flags=re.IGNORECASE)
    conf = float(m_conf.group("c")) if m_conf else 0.5
    conf = max(0.0, min(1.0, conf))

    m_rat = re.search(r'(?:"rationale"\s*:\s*")(?P<r>.*?)(?:"\s*[,}])', jsonish, flags=re.IGNORECASE)
    rationale = (m_rat.group("r").strip() if m_rat else answer_text)
    if len(rationale) > 400:
        rationale = rationale[:400] + "…"
    return OpenSolveResult(answer_text=answer_text, confidence=conf, rationale=rationale)


class LocalOllamaSolver(Solver):
    """
    Local solver via Ollama.

    Note: Vision support depends on the chosen model. If the model doesn't support images,
    we still send the text; for best results pick a vision-capable model.
    """

    def __init__(
        self,
        *,
        model: str = "llava",
        host: Optional[str] = None,
    ) -> None:
        self._client = OllamaClient(host=host) if host else OllamaClient()
        self._model = model

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(2))
    def solve(self, inp: SolveInput) -> SolveResult:
        opts = "\n".join([f"{chr(ord('A')+i)}) {t}" for i, t in enumerate(inp.options)])
        prompt = (
            f"Sei un tutor di {inp.subject}. "
            "Scegli UNA sola risposta corretta (A-D). Rispondi SOLO in JSON valido.\n\n"
            f"QID: {inp.qid}\n"
            f"Domanda:\n{inp.prompt}\n\n"
            f"Opzioni:\n{opts}\n\n"
            "Output JSON con chiavi: answer (A|B|C|D), confidence (0..1), rationale (breve)."
        )

        images = [inp.page_image_png_path] if inp.page_image_png_path else None
        resp = self._client.generate(model=self._model, prompt=prompt, images=images)
        txt = (resp.get("response") or "").strip()
        return _parse_solve_result_from_text(txt)

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(2))
    def solve_open(self, inp: SolveInput) -> OpenSolveResult:
        prompt = (
            f"Sei un tutor di {inp.subject}. "
            "Fornisci una risposta sintetica a una domanda aperta. Rispondi SOLO in JSON valido.\n\n"
            f"QID: {inp.qid}\n"
            f"Domanda aperta:\n{inp.prompt}\n\n"
            "Output JSON con chiavi: answer_text (breve), confidence (0..1), rationale (breve)."
        )
        images = [inp.page_image_png_path] if inp.page_image_png_path else None
        resp = self._client.generate(model=self._model, prompt=prompt, images=images)
        txt = (resp.get("response") or "").strip()
        return _parse_open_result_from_text(txt)

