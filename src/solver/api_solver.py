from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import OpenSolveResult, SolveInput, SolveResult, Solver


def _encode_png_b64(png_path: str | Path) -> str:
    data = Path(png_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


class ApiSolver(Solver):
    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def solve(self, inp: SolveInput) -> SolveResult:
        opts = "\n".join([f"{chr(ord('A')+i)}) {t}" for i, t in enumerate(inp.options)])
        system = (
            f"Sei un tutor di {inp.subject}. Devi scegliere UNA sola risposta corretta (A-D) "
            "per una domanda a scelta multipla. Rispondi SOLO in JSON valido."
        )

        user_text = (
            f"QID: {inp.qid}\n"
            f"Domanda:\n{inp.prompt}\n\n"
            f"Opzioni:\n{opts}\n\n"
            "Output JSON con chiavi: answer (A|B|C|D), confidence (0..1), rationale (breve)."
        )

        content: list[dict] = [{"type": "text", "text": user_text}]
        if inp.page_image_png_path:
            b64 = _encode_png_b64(inp.page_image_png_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        txt = resp.choices[0].message.content or "{}"
        return SolveResult.model_validate_json(txt)

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def solve_open(self, inp: SolveInput) -> OpenSolveResult:
        system = (
            f"Sei un tutor di {inp.subject}. Devi rispondere in modo conciso a una domanda aperta. "
            "Rispondi SOLO in JSON valido."
        )
        user_text = (
            f"QID: {inp.qid}\n"
            f"Domanda aperta:\n{inp.prompt}\n\n"
            "Output JSON con chiavi: answer_text (breve), confidence (0..1), rationale (breve)."
        )
        content: list[dict] = [{"type": "text", "text": user_text}]
        if inp.page_image_png_path:
            b64 = _encode_png_b64(inp.page_image_png_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        txt = resp.choices[0].message.content or "{}"
        return OpenSolveResult.model_validate_json(txt)

