from __future__ import annotations

from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from .base import SolveInput, SolveResult, Solver


class GeminiSolver(Solver):
    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        verbose: bool = False,
        # NOTE: structured JSON output (response_mime_type / response_json_schema)
        # is available on beta endpoints; stable v1 may reject these fields.
        api_version: str = "v1beta",
    ) -> None:
        # If api_key is None, SDK will pick up GEMINI_API_KEY / GOOGLE_API_KEY from env.
        http_options = types.HttpOptions(api_version=api_version)
        if api_key:
            self._client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            self._client = genai.Client(http_options=http_options)
        self._model = model
        self._verbose = verbose

    def solve(self, inp: SolveInput) -> SolveResult:
        opts = "\n".join([f"{chr(ord('A')+i)}) {t}" for i, t in enumerate(inp.options)])

        prompt = (
            "Sei un tutor di elettrotecnica. Devi scegliere UNA sola risposta corretta (A-D) "
            "per una domanda a scelta multipla. Rispondi solo in JSON conforme allo schema.\n\n"
            f"QID: {inp.qid}\n"
            f"Domanda:\n{inp.prompt}\n\n"
            f"Opzioni:\n{opts}\n"
        )

        contents: list[types.Part | str] = [prompt]
        if inp.page_image_png_path:
            img_bytes = Path(inp.page_image_png_path).read_bytes()
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # The free tier is heavily rate limited (e.g. 5 req/min/model). Handle 429 by sleeping
        # for the suggested retryDelay when available, then retry a few times.
        import re
        import time

        last_err: Optional[Exception] = None
        for attempt in range(1, 9):
            try:
                t0 = time.time()
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_json_schema=SolveResult.model_json_schema(),
                    ),
                )
                dt = time.time() - t0
                if self._verbose:
                    print(f"[gemini] {inp.qid}: response in {dt:.2f}s")
                txt = (resp.text or "").strip()
                return SolveResult.model_validate_json(txt)
            except genai_errors.ClientError as e:
                last_err = e
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or " 429 " in msg or msg.startswith("429"):
                    # Example includes: "retryDelay': '36s'" or "Please retry in 36.3s"
                    m = re.search(r"retryDelay'?:\s*'(?P<s>\d+)s'", msg)
                    if m:
                        wait_s = int(m.group("s")) + 1
                        if self._verbose:
                            print(f"[gemini] {inp.qid}: 429 rate-limit, sleep {wait_s}s (attempt {attempt}/8)")
                        time.sleep(wait_s)
                        continue
                    m2 = re.search(r"retry in\s+(?P<s>\d+)(?:\.\d+)?s", msg, flags=re.IGNORECASE)
                    if m2:
                        wait_s = int(m2.group("s")) + 1
                        if self._verbose:
                            print(f"[gemini] {inp.qid}: 429 rate-limit, sleep {wait_s}s (attempt {attempt}/8)")
                        time.sleep(wait_s)
                        continue
                    if self._verbose:
                        print(f"[gemini] {inp.qid}: 429 rate-limit, sleep 15s (attempt {attempt}/8)")
                    time.sleep(15)
                    continue
                raise

        raise last_err or RuntimeError("Gemini solve failed")


