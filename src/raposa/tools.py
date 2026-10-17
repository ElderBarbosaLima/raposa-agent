"""Tools the agent can call. Every match fact in an answer must come from one of these."""

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from raposa.data.store import MatchStore
from raposa.knowledge.retriever import BM25Retriever

MAX_LIMIT = 20

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_recent_matches",
        "description": (
            "Return the club's most recent finished matches, newest first, with score, "
            "opponent, competition, venue and result (W/D/L). Use for questions about the "
            "last game, recent form or a recent result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "description": "How many matches to return (default 5).",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_upcoming_matches",
        "description": (
            "Return the club's next scheduled matches, soonest first, with date (UTC), "
            "opponent, competition and venue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "description": "How many matches to return (default 3).",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_head_to_head",
        "description": (
            "Return finished matches against a given opponent available in the database, "
            "plus a W/D/L summary. The opponent is matched by name, ignoring accents and "
            "case (e.g. 'atletico', 'Flamengo'). Only covers seasons that were synced, "
            "not the full all-time record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "opponent": {"type": "string", "description": "Opponent name or part of it."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "description": "Max matches to list (default 10).",
                },
            },
            "required": ["opponent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_season_record",
        "description": (
            "Return the club's record in a season: played, wins, draws, losses, goals for "
            "and against, points (3 per win) and current form. Defaults to the latest "
            "season in the database."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "season": {"type": "integer", "description": "Season year, e.g. 2026."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_club_knowledge",
        "description": (
            "Search the curated knowledge base about the club's history: founding, titles, "
            "idols, rivalry and stadium. Use for any historical question instead of relying "
            "on memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """A recoverable tool failure; its message is returned to the model as an error result."""


class Toolbox:
    def __init__(self, store: MatchStore, retriever: BM25Retriever) -> None:
        self._store = store
        self._retriever = retriever
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "get_recent_matches": self._recent,
            "get_upcoming_matches": self._upcoming,
            "get_head_to_head": self._head_to_head,
            "get_season_record": self._season_record,
            "search_club_knowledge": self._search_knowledge,
        }

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return TOOL_DEFINITIONS

    def run(self, name: str, args: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolError(f"Unknown tool: {name}")
        return json.dumps(handler(args), ensure_ascii=False)

    # -- handlers -------------------------------------------------------------

    def _recent(self, args: dict[str, Any]) -> dict[str, Any]:
        team_id = self._team_id()
        matches = self._store.finished(_limit(args, default=5))
        return {"matches": [m.to_view(team_id) for m in matches]}

    def _upcoming(self, args: dict[str, Any]) -> dict[str, Any]:
        team_id = self._team_id()
        matches = self._store.upcoming(_limit(args, default=3))
        return {"matches": [m.to_view(team_id) for m in matches]}

    def _head_to_head(self, args: dict[str, Any]) -> dict[str, Any]:
        team_id = self._team_id()
        opponent = str(args.get("opponent", "")).strip()
        if not opponent:
            raise ToolError("'opponent' must be a non-empty string")
        matches = self._store.against(opponent)
        results = Counter(m.result(team_id) for m in matches)
        return {
            "opponent_query": opponent,
            "seasons_covered": self._store.seasons(),
            "summary": {
                "played": len(matches),
                "wins": results["W"],
                "draws": results["D"],
                "losses": results["L"],
            },
            "matches": [m.to_view(team_id) for m in matches[: _limit(args, default=10)]],
        }

    def _season_record(self, args: dict[str, Any]) -> dict[str, Any]:
        team_id = self._team_id()
        seasons = self._store.seasons()
        if not seasons:
            raise ToolError("No seasons in the database yet")
        season = int(args.get("season") or seasons[0])
        if season not in seasons:
            raise ToolError(f"Season {season} is not in the database. Available: {seasons}")
        matches = self._store.finished_in_season(season)
        results = Counter(m.result(team_id) for m in matches)
        return {
            "season": season,
            "played": len(matches),
            "wins": results["W"],
            "draws": results["D"],
            "losses": results["L"],
            "goals_for": sum(m.goals_for(team_id) for m in matches),
            "goals_against": sum(m.goals_against(team_id) for m in matches),
            "points": 3 * results["W"] + results["D"],
            "form_last_5": "".join(m.result(team_id) for m in matches[-5:]),
            "competitions": sorted({m.competition for m in matches}),
        }

    def _search_knowledge(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("'query' must be a non-empty string")
        hits = self._retriever.search(query, k=3)
        return {
            "results": [
                {"source": c.source, "section": c.title, "text": c.text, "score": round(s, 2)}
                for c, s in hits
            ]
        }

    def _team_id(self) -> int:
        team_id = self._store.team_id
        if team_id is None:
            raise ToolError("No match data has been synced yet. Run `raposa-sync` first.")
        return team_id


def _limit(args: dict[str, Any], *, default: int) -> int:
    try:
        value = int(args.get("limit") or default)
    except (TypeError, ValueError) as exc:
        raise ToolError("'limit' must be an integer") from exc
    return max(1, min(value, MAX_LIMIT))
