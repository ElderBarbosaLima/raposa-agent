from raposa.data.store import MatchStore, normalize
from tests.conftest import CRUZEIRO_ID


def test_team_metadata(store: MatchStore) -> None:
    assert store.team_id == CRUZEIRO_ID
    assert store.team_name == "Cruzeiro"


def test_finished_is_newest_first(store: MatchStore) -> None:
    dates = [m.utc_date for m in store.finished()]
    assert len(dates) == 7
    assert dates == sorted(dates, reverse=True)


def test_upcoming_is_soonest_first_and_excludes_finished(store: MatchStore) -> None:
    upcoming = store.upcoming()
    assert [m.id for m in upcoming] == [9008, 9009]
    assert all(m.home_goals is None for m in upcoming)


def test_against_matches_without_accents(store: MatchStore) -> None:
    matches = store.against("atletico")
    assert {m.id for m in matches} == {9003, 9007}


def test_upsert_replaces_existing_rows(store: MatchStore) -> None:
    match = store.upcoming(1)[0]
    finished = match.__class__(
        **{**match.__dict__, "status": "FINISHED", "home_goals": 1, "away_goals": 0}
    )
    store.upsert_matches([finished])
    assert store.finished(1)[0].id == match.id
    assert match.id not in {m.id for m in store.upcoming()}


def test_result_from_club_perspective(store: MatchStore) -> None:
    away_win = next(m for m in store.finished() if m.id == 9006)  # São Paulo 0-1 Cruzeiro
    assert away_win.result(CRUZEIRO_ID) == "W"
    assert away_win.to_view(CRUZEIRO_ID)["venue"] == "away"
    assert away_win.to_view(CRUZEIRO_ID)["score"] == "0-1"


def test_normalize() -> None:
    assert normalize("Grêmio São Paulo ATLÉTICO") == "gremio sao paulo atletico"


def test_empty_store_has_no_team() -> None:
    assert MatchStore().team_id is None
