from functools import lru_cache

from raposa.agent import RaposaAgent
from raposa.config import get_settings
from raposa.data.store import MatchStore
from raposa.knowledge.retriever import load_default_retriever
from raposa.tools import Toolbox


@lru_cache
def build_agent() -> RaposaAgent:
    """Wire the production agent from settings. Cached: one store and client per process."""
    settings = get_settings()
    toolbox = Toolbox(MatchStore(settings.db_path), load_default_retriever())
    return RaposaAgent(toolbox, settings=settings)
