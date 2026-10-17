"""Deterministic hallucination check: every score in an answer must appear in a tool result.

Scores are the facts a football agent is most tempted to invent, and they are easy to
extract with a regex, so this check runs on every eval case and in unit tests without
needing an LLM judge.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# "2-1", "2 x 1", "2×1". Lookarounds keep dates (2026-05-10) and times (21:30) out.
_SCORE = re.compile(r"(?<![\d\-/:])(\d{1,2})\s*(?:-|x|×)\s*(\d{1,2})(?![\d\-/:])")


def extract_scores(text: str) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in _SCORE.findall(text)}


@dataclass
class GroundingReport:
    checked: int
    ungrounded: list[tuple[int, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ungrounded


def check_scores(answer: str, tool_outputs: Iterable[str]) -> GroundingReport:
    """Flag scores in `answer` that no tool output contains (in either order)."""
    known: set[tuple[int, int]] = set()
    for output in tool_outputs:
        for a, b in extract_scores(output):
            known |= {(a, b), (b, a)}
    claimed = extract_scores(answer)
    return GroundingReport(
        checked=len(claimed),
        ungrounded=sorted(score for score in claimed if score not in known),
    )
