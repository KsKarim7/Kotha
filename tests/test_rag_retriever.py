import os
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import TupleRow

from kotha.db.schema import EMBEDDING_DIMENSIONS, connect, prepare_database
from kotha.kb.chunks import Chunk, chunk_documents
from kotha.kb.documents import read_documents
from kotha.kb.ingest import store_chunks
from kotha.rag.retriever import (
    HybridRetriever,
    SparseRetriever,
    reciprocal_rank_fusion,
    search_tokens,
    stored_chunk,
)
from kotha.text.lexicon import LEXICON_PATH, read_lexicon

KB_FOLDER = Path(__file__).resolve().parent.parent / "data" / "kb"
DATABASE_URL = os.environ.get("DATABASE_URL", "")
needs_database = pytest.mark.skipif(
    not DATABASE_URL and not os.environ.get("CI"), reason="DATABASE_URL is not set (outside CI)"
)
LEXICON = {"ব্যালেন্স": "balance", "ব্যালান্স": "balance"}
WORD_AXES = {"balance": 0, "voucher": 1, "roaming": 2}
NO_KNOWN_WORD_AXIS = EMBEDDING_DIMENSIONS - 1
PLANE_POINTS = {"closest": (0.6, 0.1), "middle": (3.0, 1.0), "farthest": (0.9, 0.9)}
PLANE_QUERY_POINT = (1.0, 0.0)


def make_chunk(chunk_id: str, title: str, text: str) -> Chunk:
    doc_id, lang, index = chunk_id.split(".")
    return Chunk(chunk_id, doc_id, lang, title, "Section", int(index), text)


CHUNKS = [
    make_chunk("balance.en.0", "Balance", "Dial the code to see your balance."),
    make_chunk("balance.bn.0", "ব্যালেন্স", "কোড ডায়াল করে ব্যালেন্স দেখুন।"),
    make_chunk("recharge.en.0", "Recharge", "Use a voucher from any shop."),
    make_chunk("roaming.en.0", "Roaming", "Turn on roaming before you travel."),
    make_chunk("refunds.en.0", "Refunds", "Money returns to the main account."),
    make_chunk("hours.en.0", "Hours", "Service centres open at ten."),
]
BULK_CHUNKS = [make_chunk(f"bulk.en.{index}", f"Bulk{index}", "shared") for index in range(25)]


class WordAxisEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
        for row, text in enumerate(texts):
            axes = [axis for word, axis in WORD_AXES.items() if word in text.lower()]
            vectors[row, axes or [NO_KNOWN_WORD_AXIS]] = 1.0
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class PlaneEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
        for row, text in enumerate(texts):
            points = [point for word, point in PLANE_POINTS.items() if word in text]
            vectors[row, :2] = points[0] if points else PLANE_QUERY_POINT
        return vectors


class WrongSizeEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


@pytest.fixture
def empty_table() -> Iterator[psycopg.Connection[TupleRow]]:
    schema = f"test_retriever_{uuid.uuid4().hex[:12]}"
    with connect(DATABASE_URL) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(f"CREATE SCHEMA {schema}")
        connection.execute(f"SET search_path TO {schema}, public")
        prepare_database(connection)
        connection.commit()
        try:
            yield connection
        finally:
            connection.rollback()
            connection.execute(f"DROP SCHEMA {schema} CASCADE")
            connection.commit()


@pytest.fixture
def filled_table(
    empty_table: psycopg.Connection[TupleRow],
) -> psycopg.Connection[TupleRow]:
    store_chunks(empty_table, CHUNKS, WordAxisEmbedder())
    return empty_table


def test_fusion_adds_reciprocal_ranks_with_k_60() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]]))
    assert fused == pytest.approx(
        {"a": 1 / 61 + 1 / 62, "b": 1 / 62, "c": 1 / 63 + 1 / 61},
    )


def test_fusion_sorts_by_score_then_id() -> None:
    fused = reciprocal_rank_fusion([["c"], ["b"], ["d", "a"], ["a"]])
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c", "d"]


def test_search_tokens_drop_markdown_markup() -> None:
    assert search_tokens("| Send `BAL` to `7470` |") == ["send", "bal", "to", "7470"]
    assert search_tokens("`NT-30GB`") == ["nt", "30gb"]
    assert search_tokens("`*747*5*<voucher number>#`") == ["747", "5", "voucher", "number"]


def test_markup_in_a_query_does_not_change_the_ranking() -> None:
    sparse = SparseRetriever(CHUNKS, LEXICON)
    assert sparse.ranked_ids("`balance`") == sparse.ranked_ids("balance") != []


def test_fusion_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_a_stored_row_becomes_a_chunk_with_its_index() -> None:
    row = ("balance.en.12", "balance", "en", "Balance", "Checking", "Text.")
    assert stored_chunk(row) == Chunk(
        "balance.en.12", "balance", "en", "Balance", "Checking", 12, "Text."
    )


def test_the_lexicon_lets_a_latin_query_find_a_bangla_chunk() -> None:
    sparse = SparseRetriever(CHUNKS, LEXICON)
    assert "balance.bn.0" not in sparse.ranked_ids("balance", use_lexicon=False)
    assert "balance.bn.0" in sparse.ranked_ids("balance")


def test_the_lexicon_lets_another_bangla_spelling_find_both_languages() -> None:
    sparse = SparseRetriever(CHUNKS, LEXICON)
    assert sparse.ranked_ids("ব্যালান্স", use_lexicon=False) == []
    assert set(sparse.ranked_ids("ব্যালান্স")) == {"balance.en.0", "balance.bn.0"}


def test_sparse_search_ignores_chunks_without_a_query_word_and_respects_the_limit() -> None:
    sparse = SparseRetriever(CHUNKS, LEXICON)
    assert sparse.ranked_ids("travel") == ["roaming.en.0"]
    assert sparse.ranked_ids("nothing matches this") == []
    assert len(sparse.ranked_ids("balance", limit=1)) == 1


def test_sparse_search_returns_twenty_candidates_by_default() -> None:
    assert len(SparseRetriever(BULK_CHUNKS, LEXICON).ranked_ids("shared")) == 20


def test_sparse_search_reads_the_title_and_heading() -> None:
    sparse = SparseRetriever(CHUNKS, LEXICON)
    assert sparse.ranked_ids("hours") == ["hours.en.0"]


def test_sparse_search_on_the_real_knowledge_base() -> None:
    chunks = chunk_documents(read_documents(KB_FOLDER))
    sparse = SparseRetriever(chunks, read_lexicon(LEXICON_PATH))
    assert sparse.ranked_ids("roaming")[0].startswith("roaming.")
    assert sparse.ranked_ids("ব্যালান্স", use_lexicon=False) == []
    assert sparse.ranked_ids("ব্যালান্স")[0].startswith("balance-check.")


@pytest.mark.parametrize("code", ["BAL", "7470", "01700000009"])
def test_a_code_written_in_backticks_can_be_found(code: str) -> None:
    chunks = chunk_documents(read_documents(KB_FOLDER))
    text_by_id = {chunk.id: chunk.text for chunk in chunks}
    sparse = SparseRetriever(chunks, read_lexicon(LEXICON_PATH))
    assert any(f"`{code}`" in text_by_id[chunk_id] for chunk_id in sparse.ranked_ids(code)[:4])


@needs_database
def test_dense_search_orders_by_cosine_distance_not_length_or_dot_product(
    empty_table: psycopg.Connection[TupleRow],
) -> None:
    chunks = [make_chunk(f"{word}.en.0", "Point", word) for word in PLANE_POINTS]
    store_chunks(empty_table, chunks, PlaneEmbedder())
    retriever = HybridRetriever(empty_table, PlaneEmbedder(), LEXICON)
    assert retriever.dense_ids("query") == ["closest.en.0", "middle.en.0", "farthest.en.0"]
    assert retriever.dense_ids("query", limit=1) == ["closest.en.0"]


@needs_database
def test_search_fuses_twenty_candidates_from_each_side(
    empty_table: psycopg.Connection[TupleRow],
) -> None:
    store_chunks(empty_table, BULK_CHUNKS, WordAxisEmbedder())
    retriever = HybridRetriever(empty_table, WordAxisEmbedder(), LEXICON)
    assert len(retriever.dense_ids("shared")) == 20
    assert len(retriever.search("shared", limit=len(BULK_CHUNKS))) == 20


@needs_database
def test_search_fuses_both_rankings_and_returns_the_chunk_text(
    filled_table: psycopg.Connection[TupleRow],
) -> None:
    retriever = HybridRetriever(filled_table, WordAxisEmbedder(), LEXICON)
    results = retriever.search("balance", use_lexicon=False)
    assert len(results) == 4
    assert results[0].chunk == CHUNKS[0]
    assert results[0].score == pytest.approx(2 / 61)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


@needs_database
def test_the_lexicon_only_changes_the_sparse_side_of_search(
    filled_table: psycopg.Connection[TupleRow],
) -> None:
    retriever = HybridRetriever(filled_table, WordAxisEmbedder(), LEXICON)
    plain = {result.chunk.id: result.score for result in retriever.search("balance", 6, False)}
    expanded = {result.chunk.id: result.score for result in retriever.search("balance", 6)}
    assert expanded["balance.bn.0"] > plain["balance.bn.0"]
    assert expanded["roaming.en.0"] == pytest.approx(plain["roaming.en.0"])


@needs_database
@pytest.mark.parametrize("query", ["", "   ", "?!", "`", "| <>"])
def test_a_query_without_words_returns_nothing_and_embeds_nothing(
    filled_table: psycopg.Connection[TupleRow], query: str
) -> None:
    embedder = WordAxisEmbedder()
    assert HybridRetriever(filled_table, embedder, LEXICON).search(query) == []
    assert embedder.calls == 0


@needs_database
def test_a_chunk_ingested_after_startup_is_ignored_until_restart(
    filled_table: psycopg.Connection[TupleRow],
) -> None:
    retriever = HybridRetriever(filled_table, WordAxisEmbedder(), LEXICON)
    late_chunk = make_chunk("late.en.0", "Roaming", "More roaming words.")
    store_chunks(filled_table, [*CHUNKS, late_chunk], WordAxisEmbedder())
    assert retriever.dense_ids("roaming")[0] == "roaming.en.0"
    assert "late.en.0" not in retriever.dense_ids("roaming")
    assert "late.en.0" not in [
        result.chunk.id for result in retriever.search("roaming", limit=len(CHUNKS) + 1)
    ]


@needs_database
def test_a_failed_query_does_not_break_later_searches(
    filled_table: psycopg.Connection[TupleRow],
) -> None:
    working = HybridRetriever(filled_table, WordAxisEmbedder(), LEXICON)
    broken = HybridRetriever(filled_table, WrongSizeEmbedder(), LEXICON)
    with pytest.raises(psycopg.errors.DataException):
        broken.dense_ids("balance")
    assert working.search("roaming")[0].chunk.id == "roaming.en.0"


@needs_database
def test_concurrent_searches_can_share_the_connection(
    filled_table: psycopg.Connection[TupleRow],
) -> None:
    retriever = HybridRetriever(filled_table, WordAxisEmbedder(), LEXICON)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rankings = list(pool.map(retriever.dense_ids, ["roaming"] * 40))
    assert all(ranking[0] == "roaming.en.0" for ranking in rankings)
    assert filled_table.info.transaction_status == TransactionStatus.IDLE


@needs_database
def test_searching_leaves_no_transaction_open(
    filled_table: psycopg.Connection[TupleRow],
) -> None:
    retriever = HybridRetriever(filled_table, WordAxisEmbedder(), LEXICON)
    assert filled_table.info.transaction_status == TransactionStatus.IDLE
    retriever.search("balance")
    assert filled_table.info.transaction_status == TransactionStatus.IDLE


@needs_database
def test_an_empty_table_is_refused_with_a_clear_message(
    empty_table: psycopg.Connection[TupleRow],
) -> None:
    with pytest.raises(ValueError, match="kotha.kb.ingest"):
        HybridRetriever(empty_table, WordAxisEmbedder(), LEXICON)
