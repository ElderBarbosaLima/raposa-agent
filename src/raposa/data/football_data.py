"""Thin client for the football-data.org v4 API (free tier covers the Brasileirão Série A)."""

import json
from pathlib import Path
from typing import Any

import httpx

from raposa.data.models import Match
from raposa.data.store import MatchStore

BASE_URL = "https://api.football-data.org/v4"


class FootballDataError(RuntimeError):
    pass


def parse_match(raw: dict[str, Any]) -> Match:
    full_time = (raw.get("score") or {}).get("fullTime") or {}
    season_start = (raw.get("season") or {}).get("startDate") or raw["utcDate"]
    return Match(
        id=raw["id"],
        utc_date=raw["utcDate"],
        status=raw["status"],
        competition=raw["competition"]["name"],
        season=int(season_start[:4]),
        matchday=raw.get("matchday"),
        home_id=raw["homeTeam"]["id"],
        home_name=raw["homeTeam"].get("shortName") or raw["homeTeam"]["name"],
        away_id=raw["awayTeam"]["id"],
        away_name=raw["awayTeam"].get("shortName") or raw["awayTeam"]["name"],
        home_goals=full_time.get("home"),
        away_goals=full_time.get("away"),
    )


def load_snapshot(path: str | Path, store: MatchStore) -> MatchStore:
    """Load a saved API payload ({"team": {...}, "matches": [...]}) into `store`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    team = payload["team"]
    store.set_team(team["id"], team.get("shortName") or team["name"])
    store.upsert_matches(parse_match(raw) for raw in payload["matches"])
    return store


class FootballDataClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        http: httpx.Client | None = None,
    ) -> None:
        self._http = http or httpx.Client(timeout=20.0)
        self._base_url = base_url
        self._headers = {"X-Auth-Token": api_key}

    def find_team(self, name: str = "Cruzeiro", competition: str = "BSA") -> tuple[int, str]:
        """Return (team_id, display_name) for the first team in `competition` matching `name`."""
        payload = self._get(f"/competitions/{competition}/teams")
        for team in payload.get("teams", []):
            if name.casefold() in team["name"].casefold():
                return team["id"], team.get("shortName") or team["name"]
        raise FootballDataError(f"No team matching {name!r} in competition {competition}")

    def team_matches(self, team_id: int, season: int) -> list[Match]:
        payload = self._get(f"/teams/{team_id}/matches", params={"season": season})
        return [parse_match(raw) for raw in payload.get("matches", [])]

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._http.get(self._base_url + path, headers=self._headers, params=params)
        if response.status_code == 429:
            raise FootballDataError("Rate limited by football-data.org (free tier: 10 req/min)")
        response.raise_for_status()
        return response.json()
