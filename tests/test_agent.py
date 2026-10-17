"""Agent loop tests with a scripted fake client: no network, no API key."""

from datetime import date
from types import SimpleNamespace as NS
from typing import Any

from raposa.agent import FALLBACK_BETA, MAX_ROUNDS_MESSAGE, REFUSAL_MESSAGE, RaposaAgent
from raposa.config import Settings
from raposa.tools import Toolbox


def text(value: str) -> NS:
    return NS(type="text", text=value)


def tool_use(id_: str, name: str, **input_: Any) -> NS:
    return NS(type="tool_use", id=id_, name=name, input=input_)


def response(stop_reason: str, *content: NS) -> NS:
    return NS(stop_reason=stop_reason, stop_details=None, content=list(content))


class FakeClient:
    """Returns scripted responses and records every request."""

    def __init__(self, *responses: NS) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs: Any) -> NS:
        # Snapshot messages: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def make_agent(toolbox: Toolbox, client: FakeClient, **settings: Any) -> RaposaAgent:
    return RaposaAgent(
        toolbox, client=client, settings=Settings(**settings), today=date(2099, 5, 20)
    )


def test_runs_tools_and_returns_final_text(toolbox: Toolbox) -> None:
    client = FakeClient(
        response(
            "tool_use", text("Deixa eu ver..."), tool_use("t1", "get_recent_matches", limit=1)
        ),
        response("end_turn", text("Empate em 2 x 2 no clássico!")),
    )
    reply = make_agent(toolbox, client).ask("Como foi o último jogo?")

    assert reply.text == "Empate em 2 x 2 no clássico!"
    assert reply.stop_reason == "end_turn"
    assert [c.name for c in reply.tool_calls] == ["get_recent_matches"]
    assert '"score": "2-2"' in reply.tool_calls[0].output

    tool_results = client.requests[1]["messages"][-1]["content"]
    assert tool_results[0]["tool_use_id"] == "t1"
    assert tool_results[0]["is_error"] is False


def test_parallel_tool_results_go_back_in_one_message(toolbox: Toolbox) -> None:
    client = FakeClient(
        response(
            "tool_use", tool_use("a", "get_recent_matches"), tool_use("b", "get_upcoming_matches")
        ),
        response("end_turn", text("ok")),
    )
    make_agent(toolbox, client).ask("Resumo da semana?")

    last = client.requests[1]["messages"][-1]
    assert last["role"] == "user"
    assert [r["tool_use_id"] for r in last["content"]] == ["a", "b"]


def test_tool_errors_are_reported_to_the_model(toolbox: Toolbox) -> None:
    client = FakeClient(
        response("tool_use", tool_use("t1", "get_season_record", season=1921)),
        response("end_turn", text("Não tenho dados dessa temporada.")),
    )
    reply = make_agent(toolbox, client).ask("Campanha de 1921?")

    assert reply.tool_calls[0].is_error
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert result["content"].startswith("Error:")


def test_request_shape(toolbox: Toolbox) -> None:
    client = FakeClient(response("end_turn", text("Oi!")))
    make_agent(toolbox, client, effort="low").ask("oi", persona="analyst")

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "low"}
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert "neutral" in request["system"][0]["text"]
    assert request["system"][1]["text"] == "Today's date is 2099-05-20."


def test_refusal_returns_safe_message(toolbox: Toolbox) -> None:
    client = FakeClient(response("refusal"))
    reply = make_agent(toolbox, client).ask("...")
    assert reply.text == REFUSAL_MESSAGE
    assert reply.stop_reason == "refusal"


def test_stops_after_max_tool_rounds(toolbox: Toolbox) -> None:
    looping = [response("tool_use", tool_use(f"t{i}", "get_recent_matches")) for i in range(2)]
    client = FakeClient(*looping)
    reply = make_agent(toolbox, client, max_tool_rounds=2).ask("loop")
    assert reply.text == MAX_ROUNDS_MESSAGE
    assert len(reply.tool_calls) == 2


def test_history_is_prepended(toolbox: Toolbox) -> None:
    client = FakeClient(response("end_turn", text("ok")))
    history = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "Fala!"}]
    make_agent(toolbox, client).ask("e aí?", history=history)
    assert client.requests[0]["messages"][:2] == history
