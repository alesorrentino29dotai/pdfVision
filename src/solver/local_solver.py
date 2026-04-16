from __future__ import annotations

import re
from typing import Optional

from ollama import Client as OllamaClient
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SolveInput, SolveResult, Solver


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
            "Scegli UNA sola risposta corretta (A-D). Rispondi SOLO in JSON valido.\n\n"
            f"QID: {inp.qid}\n"
            f"Domanda:\n{inp.prompt}\n\n"
            f"Opzioni:\n{opts}\n\n"
            "Output JSON con chiavi: answer (A|B|C|D), confidence (0..1), rationale (breve)."
        )

        images = [inp.page_image_png_path] if inp.page_image_png_path else None
        resp = self._client.generate(model=self._model, prompt=prompt, images=images)
        txt = (resp.get("response") or "").strip()
        return SolveResult.model_validate_json(_extract_json_for_solve(txt))

