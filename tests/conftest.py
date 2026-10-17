from pathlib import Path

import pytest

from raposa.data.football_data import load_snapshot
from raposa.data.store import MatchStore
from raposa.knowledge.retriever import load_default_retriever
from raposa.tools import Toolbox

FIXTURE = Path(__file__).parent / "fixtures" / "matches_2099.json"
CRUZEIRO_ID = 1771


@pytest.fixture
def store() -> MatchStore:
    return load_snapshot(FIXTURE, MatchStore())


@pytest.fixture
def toolbox(store: MatchStore) -> Toolbox:
    return Toolbox(store, load_default_retriever())
