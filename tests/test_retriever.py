import pytest

from raposa.knowledge.retriever import BM25Retriever, Chunk, chunk_markdown, load_default_retriever


@pytest.fixture(scope="module")
def retriever() -> BM25Retriever:
    return load_default_retriever()


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("When did Cruzeiro win the Libertadores?", "Copa Libertadores"),
        ("Quantas Copas do Brasil o Cruzeiro tem?", "Copa do Brasil"),
        ("Why was Palestra Italia renamed?", "The name change to Cruzeiro"),
        ("tríplice coroa 2003", "The 2003 treble"),
        ("Mineirão stadium", "The Mineirão"),
        ("Ronaldo debut", "Ronaldo"),
    ],
)
def test_top_hit(retriever: BM25Retriever, query: str, expected_section: str) -> None:
    hits = retriever.search(query, k=3)
    assert hits, f"no hits for {query!r}"
    assert expected_section in [chunk.title for chunk, _ in hits]


def test_no_hits_for_unrelated_query(retriever: BM25Retriever) -> None:
    assert retriever.search("quantum chromodynamics lattice") == []


def test_chunk_markdown_splits_on_h2() -> None:
    chunks = chunk_markdown("doc.md", "# Title\n\n## One\nfirst\n\n## Two\nsecond\n")
    assert [(c.title, c.text) for c in chunks] == [("One", "first"), ("Two", "second")]


def test_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_scores_are_sorted() -> None:
    retriever = BM25Retriever(
        [Chunk("a", "fox", "fox fox fox"), Chunk("b", "fox", "fox den"), Chunk("c", "x", "y")]
    )
    scores = [score for _, score in retriever.search("fox")]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) == 2
