"""Sync the club's fixtures from football-data.org into the local SQLite store.

Usage:
    uv run raposa-sync --season 2026
"""

import argparse
import sys
from datetime import date

from raposa.config import get_settings
from raposa.data.football_data import FootballDataClient, FootballDataError
from raposa.data.store import MatchStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, action="append", help="Season year (repeatable)")
    parser.add_argument("--team", default="Cruzeiro", help="Team name to look up")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.football_data_api_key:
        print("FOOTBALL_DATA_API_KEY is not set. See .env.example.", file=sys.stderr)
        return 1

    client = FootballDataClient(settings.football_data_api_key)
    store = MatchStore(settings.db_path)
    try:
        team_id, team_name = client.find_team(args.team)
        store.set_team(team_id, team_name)
        for season in args.season or [date.today().year]:
            count = store.upsert_matches(client.team_matches(team_id, season))
            print(f"{team_name} {season}: {count} matches synced")
    except FootballDataError as exc:
        print(f"Sync failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
