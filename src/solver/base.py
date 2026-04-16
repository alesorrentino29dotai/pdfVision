from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import BaseModel, Field


AnswerLabel = Literal["A", "B", "C", "D"]


class SolveResult(BaseModel):
    answer: AnswerLabel = Field(..., description="Correct option label (A-D).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0-1 confidence.")
    rationale: str = Field(..., description="Short explanation (1-3 sentences).")


class SolveInput(BaseModel):
    qid: str
    subject: str = "elettrotecnica"
    prompt: str
    options: list[str]  # A..D by order
    page_image_png_path: Optional[str] = None  # full page render, optional


class OpenSolveResult(BaseModel):
    answer_text: str = Field(..., description="Short textual answer for open-ended question.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0-1 confidence.")
    rationale: str = Field(..., description="Short explanation (1-3 sentences).")


class Solver(ABC):
    @abstractmethod
    def solve(self, inp: SolveInput) -> SolveResult:
        raise NotImplementedError

    @abstractmethod
    def solve_open(self, inp: SolveInput) -> OpenSolveResult:
        raise NotImplementedError

