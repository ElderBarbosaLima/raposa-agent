import sqlite3
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from raposa.data.models import FINISHED, UPCOMING_STATUSES, Match

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY,
    utc_date    TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    competition TEXT    NOT NULL,
    season      INTEGER NOT NULL,
    matchday    INTEGER,
    home_id     INTEGER NOT NULL,
    home_name   TEXT    NOT NULL,
    away_id     INTEGER NOT NULL,
    away_name   TEXT    NOT NULL,
    home_goals  INTEGER,
    away_goals  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (utc_date);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_COLUMNS = (
    "id, utc_date, status, competition, season, matchday, "
    "home_id, home_name, away_id, away_name, home_goals, away_goals"
)


def normalize(text: str) -> str:
    """Lowercase and strip accents so 'Atlético' matches 'atletico'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


class MatchStore:
    """SQLite-backed store of the club's fixtures."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Read-mostly store shared by FastAPI/Streamlit worker threads.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    # -- writes ---------------------------------------------------------------

    def upsert_matches(self, matches: Iterable[Match]) -> int:
        rows = [
            (
                m.id,
                m.utc_date,
                m.status,
                m.competition,
                m.season,
                m.matchday,
                m.home_id,
                m.home_name,
                m.away_id,
                m.away_name,
                m.home_goals,
                m.away_goals,
            )
            for m in matches
        ]
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO matches ({_COLUMNS}) VALUES ({', '.join('?' * 12)})",
                rows,
            )
        return len(rows)

    def set_team(self, team_id: int, team_name: str) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [("team_id", str(team_id)), ("team_name", team_name)],
            )

    # -- reads ----------------------------------------------------------------

    @property
    def team_id(self) -> int | None:
        value = self._meta("team_id")
        return int(value) if value is not None else None

    @property
    def team_name(self) -> str | None:
        return self._meta("team_name")

    def finished(self, limit: int | None = None) -> list[Match]:
        """Finished matches, most recent first."""
        return self._query(
            "WHERE status = ? AND home_goals IS NOT NULL ORDER BY utc_date DESC",
            (FINISHED,),
            limit,
        )

    def upcoming(self, limit: int | None = None) -> list[Match]:
        """Scheduled matches, soonest first."""
        placeholders = ", ".join("?" * len(UPCOMING_STATUSES))
        return self._query(
            f"WHERE status IN ({placeholders}) ORDER BY utc_date ASC", UPCOMING_STATUSES, limit
        )

    def against(self, opponent: str, limit: int | None = None) -> list[Match]:
        """Finished matches against an opponent, matched by accent-insensitive substring."""
        team_id = self.team_id
        needle = normalize(opponent)
        found = [m for m in self.finished() if needle in normalize(m.opponent(team_id))]
        return found[:limit] if limit else found

    def seasons(self) -> list[int]:
        rows = self._conn.execute("SELECT DISTINCT season FROM matches ORDER BY season DESC")
        return [row[0] for row in rows]

    def finished_in_season(self, season: int) -> list[Match]:
        return self._query(
            "WHERE status = ? AND home_goals IS NOT NULL AND season = ? ORDER BY utc_date ASC",
            (FINISHED, season),
        )

    def close(self) -> None:
        self._conn.close()

    # -- helpers --------------------------------------------------------------

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _query(self, clause: str, params: tuple, limit: int | None = None) -> list[Match]:
        sql = f"SELECT {_COLUMNS} FROM matches {clause}"
        if limit:
            sql += " LIMIT ?"
            params = (*params, limit)
        return [Match(*row) for row in self._conn.execute(sql, params)]
