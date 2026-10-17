"""Run the eval suite against the real model.

    uv run python -m evals.run                # all cases
    uv run python -m evals.run -k rival       # cases whose id contains "rival"

Costs a few cents per run (one agent conversation plus at most one judge call per case).
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import anthropic
import yaml
from pydantic import BaseModel

from raposa.agent import AgentReply, RaposaAgent
from raposa.config import get_settings
from raposa.data.football_data import load_snapshot
from raposa.data.store import MatchStore
from raposa.grounding import check_scores
from raposa.knowledge.retriever import load_default_retriever
from raposa.tools import Toolbox

ROOT = Path(__file__).parent
FIXTURE = ROOT.parent / "tests" / "fixtures" / "matches_2099.json"
TODAY = date(2099, 5, 20)

JUDGE_PROMPT = """\
You are grading one answer from a football chatbot about Cruzeiro.

Question: {question}

Answer:
<answer>
{answer}
</answer>

Criterion: {rubric}

Decide whether the answer meets the criterion. Be strict but fair."""


class Verdict(BaseModel):
    reasoning: str
    passed: bool


@dataclass
class CaseResult:
    id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    seconds: float = 0.0


def judge(
    client: anthropic.Anthropic, model: str, question: str, answer: str, rubric: str
) -> Verdict:
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(question=question, answer=answer, rubric=rubric),
            }
        ],
        output_format=Verdict,
    )
    if response.parsed_output is None:
        return Verdict(
            reasoning=f"judge returned no verdict ({response.stop_reason})", passed=False
        )
    return response.parsed_output


def evaluate(case: dict, reply: AgentReply, judge_fn) -> list[str]:
    failures = []
    called = {c.name for c in reply.tool_calls}
    for tool in case.get("expect_tools", []):
        if tool not in called:
            failures.append(f"expected tool {tool} to be called")

    expected = case.get("must_contain_any")
    if expected and not any(s.casefold() in reply.text.casefold() for s in expected):
        failures.append(f"answer contains none of {expected}")

    grounding = check_scores(reply.text, [c.output for c in reply.tool_calls])
    if not grounding.ok:
        failures.append(f"ungrounded scores: {grounding.ungrounded}")

    if rubric := case.get("rubric"):
        verdict = judge_fn(case["question"], reply.text, rubric)
        if not verdict.passed:
            failures.append(f"judge: {verdict.reasoning}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Raposa evals")
    parser.add_argument("-k", "--filter", default="", help="Only run cases whose id contains this")
    args = parser.parse_args(argv)

    cases = [c for c in yaml.safe_load((ROOT / "cases.yaml").read_text()) if args.filter in c["id"]]
    settings = get_settings()
    client = anthropic.Anthropic()
    toolbox = Toolbox(load_snapshot(FIXTURE, MatchStore()), load_default_retriever())
    agent = RaposaAgent(toolbox, client=client, settings=settings, today=TODAY)

    def judge_fn(question: str, answer: str, rubric: str) -> Verdict:
        return judge(client, settings.model, question, answer, rubric)

    results: list[CaseResult] = []
    for case in cases:
        started = time.monotonic()
        reply = agent.ask(case["question"], persona=case["persona"])
        failures = evaluate(case, reply, judge_fn)
        result = CaseResult(
            id=case["id"],
            passed=not failures,
            failures=failures,
            answer=reply.text,
            tools_called=[c.name for c in reply.tool_calls],
            seconds=round(time.monotonic() - started, 1),
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  {case['id']:<24} {result.seconds:>5}s  {', '.join(failures)}")

    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} passed with {settings.model} (effort={settings.effort})")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_file.write_text(
        json.dumps(
            {
                "model": settings.model,
                "effort": settings.effort,
                "passed": passed,
                "total": len(results),
                "cases": [asdict(r) for r in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {out_file.relative_to(ROOT.parent)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
