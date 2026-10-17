import json

import pytest

from raposa.data.store import MatchStore
from raposa.knowledge.retriever import load_default_retriever
from raposa.tools import TOOL_DEFINITIONS, Toolbox, ToolError


def run(toolbox: Toolbox, name: str, **args) -> dict:
    return json.loads(toolbox.run(name, args))


def test_every_definition_has_a_handler(toolbox: Toolbox) -> None:
    for definition in TOOL_DEFINITIONS:
        assert definition["input_schema"]["additionalProperties"] is False
        assert definition["name"] in toolbox._handlers


def test_recent_matches(toolbox: Toolbox) -> None:
    matches = run(toolbox, "get_recent_matches", limit=2)["matches"]
    assert [m["opponent"] for m in matches] == ["Atlético Mineiro", "São Paulo"]
    assert matches[0]["score"] == "2-2"
    assert matches[0]["result"] == "D"


def test_limit_is_clamped(toolbox: Toolbox) -> None:
    assert len(run(toolbox, "get_recent_matches", limit=500)["matches"]) == 7


def test_upcoming_matches_have_no_score(toolbox: Toolbox) -> None:
    matches = run(toolbox, "get_upcoming_matches")["matches"]
    assert [m["opponent"] for m in matches] == ["Flamengo", "Palmeiras"]
    assert all("score" not in m for m in matches)


def test_head_to_head(toolbox: Toolbox) -> None:
    result = run(toolbox, "get_head_to_head", opponent="Atletico")
    assert result["summary"] == {"played": 2, "wins": 1, "draws": 1, "losses": 0}
    assert result["seasons_covered"] == [2099]


def test_season_record(toolbox: Toolbox) -> None:
    record = run(toolbox, "get_season_record")
    assert record["season"] == 2099
    assert (record["wins"], record["draws"], record["losses"]) == (4, 2, 1)
    assert (record["goals_for"], record["goals_against"]) == (13, 8)
    assert record["points"] == 14
    assert record["form_last_5"] == "WLWWD"


def test_unknown_season_is_a_tool_error(toolbox: Toolbox) -> None:
    with pytest.raises(ToolError, match="not in the database"):
        toolbox.run("get_season_record", {"season": 1921})


def test_knowledge_search(toolbox: Toolbox) -> None:
    results = run(toolbox, "search_club_knowledge", query="Libertadores titles")["results"]
    assert any("1976" in r["text"] and "1997" in r["text"] for r in results)


def test_empty_store_explains_how_to_sync() -> None:
    toolbox = Toolbox(MatchStore(), load_default_retriever())
    with pytest.raises(ToolError, match="raposa-sync"):
        toolbox.run("get_recent_matches", {})


def test_unknown_tool(toolbox: Toolbox) -> None:
    with pytest.raises(ToolError, match="Unknown tool"):
        toolbox.run("get_weather", {})
