"""Optional timing script for Ollama (mirrors LocalOllamaSolver.generate options).

Environment:
  OLLAMA_BENCH_NUM_GPU — unset = default GPU offload; ``0`` = CPU only (helps on tight VRAM).
"""
from __future__ import annotations

import os
import time

from ollama import Client


def _gen_opts() -> dict:
    o: dict = {"num_predict": 384, "temperature": 0.1}
    raw = os.environ.get("OLLAMA_BENCH_NUM_GPU", "").strip()
    if raw != "":
        o["num_gpu"] = int(raw)
    return o


def run_generate(model: str, prompt: str) -> tuple[float, int]:
    c = Client()
    t0 = time.perf_counter()
    r = c.generate(model=model, prompt=prompt, options=_gen_opts())
    elapsed = time.perf_counter() - t0
    text = r.get("response") or ""
    return elapsed, len(text)


PROMPT = (
    "Scegli UNA sola risposta corretta (A-D). Rispondi SOLO in JSON valido.\n\n"
    "QID: q1\n"
    "Domanda: Qual è la capitale d'Italia?\n\n"
    "Opzioni:\n"
    "A) Torino\nB) Milano\nC) Roma\nD) Napoli\n\n"
    "Output JSON con chiavi: answer (A|B|C|D), confidence (0..1), rationale (breve)."
)


def main() -> None:
    models = ["qwen2.5:7b-instruct", "llava:7b"]
    for model in models:
        print(f"\n=== {model} ===")
        for run in (1, 2):
            sec, n = run_generate(model, PROMPT)
            print(f"  run {run}: {sec:.2f} s  (response chars: {n})")


if __name__ == "__main__":
    main()
