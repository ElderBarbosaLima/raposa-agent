"""Offline checks for the eval harness itself: cases are well-formed and scoring works."""

from pathlib import Path

import yaml

from evals.run import Verdict, evaluate
from raposa.agent import AgentReply, ToolCall
from raposa.personas import PERSONAS
from raposa.tools import TOOL_DEFINITIONS

CASES = yaml.safe_load((Path(__file__).parents[1] / "evals" / "cases.yaml").read_text())
TOOL_NAMES = {t["name"] for t in TOOL_DEFINITIONS}


def test_cases_are_well_formed() -> None:
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    for case in CASES:
        assert case["persona"] in PERSONAS
        assert set(case.get("expect_tools", [])) <= TOOL_NAMES
        assert case.get("expect_tools") or case.get("must_contain_any") or case.get("rubric")


def never_called(*_: str) -> Verdict:
    raise AssertionError("judge should not run for cases without a rubric")


def test_evaluate_passes_grounded_answer() -> None:
    case = {"question": "q", "expect_tools": ["get_recent_matches"], "must_contain_any": ["Galo"]}
    reply = AgentReply(
        "Empate de 2 x 2 com o Galo!",
        "end_turn",
        [ToolCall("get_recent_matches", {}, '{"score": "2-2"}')],
    )
    assert evaluate(case, reply, never_called) == []


def test_evaluate_reports_every_failure() -> None:
    case = {
        "question": "q",
        "expect_tools": ["get_head_to_head"],
        "must_contain_any": ["Flamengo"],
        "rubric": "r",
    }
    reply = AgentReply("Ganhamos de 3-0!", "end_turn", [])
    failures = evaluate(case, reply, lambda *_: Verdict(reasoning="nope", passed=False))
    assert len(failures) == 4
