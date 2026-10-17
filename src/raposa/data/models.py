from dataclasses import dataclass
from typing import Any, Literal

Result = Literal["W", "D", "L"]

FINISHED = "FINISHED"
UPCOMING_STATUSES = ("SCHEDULED", "TIMED")


@dataclass(frozen=True)
class Match:
    """A single fixture, stored exactly as the data provider reported it."""

    id: int
    utc_date: str  # ISO 8601, e.g. "2026-05-10T21:30:00Z"
    status: str
    competition: str
    season: int
    matchday: int | None
    home_id: int
    home_name: str
    away_id: int
    away_name: str
    home_goals: int | None = None
    away_goals: int | None = None

    @property
    def is_finished(self) -> bool:
        return self.status == FINISHED and self.home_goals is not None

    def involves(self, team_id: int) -> bool:
        return team_id in (self.home_id, self.away_id)

    def opponent(self, team_id: int) -> str:
        return self.away_name if self.home_id == team_id else self.home_name

    def goals_for(self, team_id: int) -> int | None:
        return self.home_goals if self.home_id == team_id else self.away_goals

    def goals_against(self, team_id: int) -> int | None:
        return self.away_goals if self.home_id == team_id else self.home_goals

    def result(self, team_id: int) -> Result | None:
        if not self.is_finished:
            return None
        scored, conceded = self.goals_for(team_id), self.goals_against(team_id)
        if scored > conceded:
            return "W"
        return "D" if scored == conceded else "L"

    def to_view(self, team_id: int) -> dict[str, Any]:
        """Serialize the match from the point of view of `team_id` (the club we follow)."""
        view: dict[str, Any] = {
            "date_utc": self.utc_date,
            "competition": self.competition,
            "season": self.season,
            "matchday": self.matchday,
            "venue": "home" if self.home_id == team_id else "away",
            "opponent": self.opponent(team_id),
            "home_team": self.home_name,
            "away_team": self.away_name,
            "status": self.status,
        }
        if self.is_finished:
            view |= {
                "score": f"{self.home_goals}-{self.away_goals}",
                "goals_for": self.goals_for(team_id),
                "goals_against": self.goals_against(team_id),
                "result": self.result(team_id),
            }
        return view
