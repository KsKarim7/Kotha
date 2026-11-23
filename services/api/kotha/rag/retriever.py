import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import TupleRow

from kotha.kb.chunks import Chunk
from kotha.kb.ingest import embedding_text
from kotha.rag.bm25 import BM25Okapi
from kotha.rag.embedder import TextEmbedder
from kotha.text.lexicon import expand_tokens, spelling_groups
from kotha.text.normalize import normalize

RRF_K = 60
CANDIDATE_COUNT = 20
RESULT_COUNT = 4
SELECT_CHUNKS = "SELECT id, doc_id, lang, title, heading, text FROM kb_chunks ORDER BY id"
NEAREST_CHUNKS = """
    SELECT id FROM kb_chunks WHERE id = ANY(%s) ORDER BY embedding <=> %s, id LIMIT %s
"""
MARKDOWN_MARKUP = str.maketrans("`|<>", "    ")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]]) -> list[tuple[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def stored_chunk(row: TupleRow) -> Chunk:
    chunk_id, doc_id, lang, title, heading, text = row
    index = int(chunk_id.rpartition(".")[2])
    return Chunk(chunk_id, doc_id, lang, title, heading, index, text)


def search_tokens(text: str) -> list[str]:
    return normalize(text.translate(MARKDOWN_MARKUP)).split()


def table_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.lstrip().startswith("|")]


def scoring_units(chunk: Chunk) -> list[list[str]]:
    rows = (f"{chunk.title}\n{chunk.heading}\n{row}" for row in table_rows(chunk.text))
    row_tokens = [tokens for tokens in map(search_tokens, rows) if tokens]
    return [search_tokens(embedding_text(chunk)), *row_tokens]


class SparseRetriever:
    # a table row is also scored alone, so a code listed once in a long table outranks chunks
    # that only mention it
    def __init__(self, chunks: Sequence[Chunk], lexicon: Mapping[str, str]) -> None:
        units = [(chunk.id, tokens) for chunk in chunks for tokens in scoring_units(chunk)]
        self._unit_chunk_ids = [chunk_id for chunk_id, _ in units]
        self._bm25 = BM25Okapi([tokens for _, tokens in units])
        self._spellings = spelling_groups(lexicon)

    def ranked_ids(
        self, query: str, limit: int = CANDIDATE_COUNT, use_lexicon: bool = True
    ) -> list[str]:
        tokens = search_tokens(query)
        if use_lexicon:
            tokens = expand_tokens(tokens, self._spellings)
        best: dict[str, float] = {}
        for chunk_id, score in zip(self._unit_chunk_ids, self._bm25.scores(tokens), strict=True):
            if score > 0:
                best[chunk_id] = max(best.get(chunk_id, 0.0), score)
        return sorted(best, key=lambda chunk_id: (-best[chunk_id], chunk_id))[:limit]


class HybridRetriever:
    def __init__(
        self,
        connection: psycopg.Connection[TupleRow],
        embedder: TextEmbedder,
        lexicon: Mapping[str, str],
    ) -> None:
        register_vector(connection)
        with connection.transaction():
            rows = connection.execute(SELECT_CHUNKS).fetchall()
        chunks = [stored_chunk(row) for row in rows]
        if not chunks:
            raise ValueError("kb_chunks is empty: run python -m kotha.kb.ingest first")
        self._connection = connection
        self._query_lock = threading.Lock()
        self._embedder = embedder
        self._chunks = {chunk.id: chunk for chunk in chunks}
        self._sparse = SparseRetriever(chunks, lexicon)

    def dense_ids(self, query: str, limit: int = CANDIDATE_COUNT) -> list[str]:
        query_vector = self._embedder.embed([query])[0]
        parameters = (list(self._chunks), query_vector, limit)
        # psycopg does not hold its connection lock for a whole transaction block
        with self._query_lock, self._connection.transaction():
            rows = self._connection.execute(NEAREST_CHUNKS, parameters).fetchall()
        return [chunk_id for (chunk_id,) in rows]

    def sparse_ids(
        self, query: str, limit: int = CANDIDATE_COUNT, use_lexicon: bool = True
    ) -> list[str]:
        return self._sparse.ranked_ids(query, limit, use_lexicon)

    def search(
        self, query: str, limit: int = RESULT_COUNT, use_lexicon: bool = True
    ) -> list[RetrievedChunk]:
        if not search_tokens(query):
            return []
        rankings = (self.dense_ids(query), self.sparse_ids(query, use_lexicon=use_lexicon))
        return [
            RetrievedChunk(self._chunks[chunk_id], score)
            for chunk_id, score in reciprocal_rank_fusion(rankings)[:limit]
        ]
