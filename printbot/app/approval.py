"""Operator approval. Every physical action is preceded by an explicit approval."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Approver(Protocol):
    async def approve(self, question: str, details: str = "") -> bool: ...


class ConsoleApprover:
    """Blocks for a typed 'yes'. Anything else is a refusal."""

    async def approve(self, question: str, details: str = "") -> bool:
        prompt = f"\n{details}\n>>> {question} Type 'yes' to proceed: " if details else f">>> {question} [yes/no]: "
        answer = await asyncio.to_thread(input, prompt)
        return answer.strip().lower() == "yes"


class DenyAll:
    async def approve(self, question: str, details: str = "") -> bool:
        return False


class ScriptedApprover:
    """Test helper: answers from a list, default deny when exhausted."""

    def __init__(self, answers: list[bool]):
        self.answers = list(answers)
        self.asked: list[str] = []

    async def approve(self, question: str, details: str = "") -> bool:
        self.asked.append(question)
        return self.answers.pop(0) if self.answers else False
