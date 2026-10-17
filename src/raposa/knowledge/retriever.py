"""A small, dependency-free BM25 retriever over the club's markdown knowledge base.

The corpus is tiny (a few dozen sections), so lexical retrieval is fast, transparent and
easy to test. Swapping in embeddings later only requires a new class with the same
`search` signature.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from raposa.data.store import normalize

DOCS_DIR = Path(__file__).parent / "docs"

_TOKEN = re.compile(r"\w+")
# fmt: off
_STOPWORDS = frozenset(
    "a an and are as at be by for from has in is it its of on or that the to was were with "
    "o os um uma de da do das dos e em no na nos nas que para por com qual quais quem "
    "quando como foi sao ser".split()
)
# fmt: on


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str
    document: str = ""  # the file's H1, e.g. "Major titles"

    @property
    def index_text(self) -> str:
        # Section titles are strong signals in a small corpus, so they count twice.
        return f"{self.document} {self.title} {self.title} {self.text}"


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    """Split a document on `## ` headings; each section becomes one chunk."""
    heading = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    document = heading.group(1).strip() if heading else ""
    chunks: list[Chunk] = []
    for section in re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]:
        title, _, body = section.partition("\n")
        chunks.append(Chunk(source, title.strip(), body.strip(), document))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self._k1, self._b = k1, b
        self._docs = [Counter(tokenize(c.index_text)) for c in chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / len(self._lengths)
        doc_freq = Counter(term for doc in self._docs for term in doc)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    @classmethod
    def from_directory(cls, path: Path = DOCS_DIR) -> "BM25Retriever":
        chunks = [
            chunk
            for file in sorted(path.glob("*.md"))
            for chunk in chunk_markdown(file.name, file.read_text(encoding="utf-8"))
        ]
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = [(chunk, self._score(terms, i)) for i, chunk in enumerate(self.chunks)]
        ranked = sorted((s for s in scored if s[1] > 0), key=lambda s: s[1], reverse=True)
        return ranked[:k]

    def _score(self, terms: list[str], index: int) -> float:
        doc, length = self._docs[index], self._lengths[index]
        score = 0.0
        for term in terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            norm = (
                tf
                * (self._k1 + 1)
                / (tf + self._k1 * (1 - self._b + self._b * length / self._avg_len))
            )
            score += self._idf[term] * norm
        return score


def load_default_retriever() -> BM25Retriever:
    return BM25Retriever.from_directory(DOCS_DIR)
