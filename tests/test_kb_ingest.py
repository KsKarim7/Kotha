import hashlib
import os
import sys
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psycopg
import pytest
from huggingface_hub import try_to_load_from_cache
from psycopg.rows import TupleRow

from kotha.db.schema import EMBEDDING_DIMENSIONS, connect
from kotha.kb.chunks import Chunk, chunk_documents
from kotha.kb.documents import read_documents
from kotha.kb.ingest import embedding_text, main, store_chunks
from kotha.rag.embedder import MODEL_ID, MODEL_REVISION, Embedder

KB_FOLDER = Path(__file__).resolve().parent.parent / "data" / "kb"
DATABASE_URL = os.environ.get("DATABASE_URL", "")
needs_database = pytest.mark.skipif(
    not DATABASE_URL and not os.environ.get("CI"), reason="DATABASE_URL is not set (outside CI)"
)
model_is_cached = isinstance(
    try_to_load_from_cache(MODEL_ID, "pytorch_model.bin", revision=MODEL_REVISION), str
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.stack([self.vector(text) for text in texts])
        return vectors.astype(np.float32)

    @staticmethod
    def vector(text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(EMBEDDING_DIMENSIONS)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def schema_connection() -> Iterator[psycopg.Connection[TupleRow]]:
    schema = f"test_ingest_{uuid.uuid4().hex[:12]}"
    with connect(DATABASE_URL) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(f"CREATE SCHEMA {schema}")
        connection.execute(f"SET search_path TO {schema}, public")
        connection.commit()
        try:
            yield connection
        finally:
            connection.rollback()
            connection.execute(f"DROP SCHEMA {schema} CASCADE")
            connection.commit()


def stored_rows(connection: psycopg.Connection[TupleRow]) -> list[tuple[str, str, str]]:
    return connection.execute("SELECT id, heading, text FROM kb_chunks ORDER BY id").fetchall()


def test_the_embedded_text_starts_with_the_title_and_heading() -> None:
    chunk = Chunk("plans.en.0", "plans", "en", "Plans", "Night pack", 0, "Body text.")
    assert embedding_text(chunk) == "Plans\nNight pack\n\nBody text."


@needs_database
def test_ingesting_twice_stores_every_chunk_once(
    schema_connection: psycopg.Connection[TupleRow],
) -> None:
    chunks = chunk_documents(read_documents(KB_FOLDER))
    embedder = FakeEmbedder()
    first = store_chunks(schema_connection, chunks, embedder)
    second = store_chunks(schema_connection, chunks, embedder)
    rows = stored_rows(schema_connection)
    assert (first.stored, first.removed, second.stored, second.removed) == (
        len(chunks),
        0,
        len(chunks),
        0,
    )
    assert rows == sorted((chunk.id, chunk.heading, chunk.text) for chunk in chunks)
    assert embedder.calls == 2


@needs_database
def test_chunks_that_left_the_knowledge_base_are_removed(
    schema_connection: psycopg.Connection[TupleRow],
) -> None:
    chunks = chunk_documents(read_documents(KB_FOLDER))
    store_chunks(schema_connection, chunks, FakeEmbedder())
    kept = [chunk for chunk in chunks if chunk.doc_id != "roaming"]
    report = store_chunks(schema_connection, kept, FakeEmbedder())
    assert report.removed == len(chunks) - len(kept) > 0
    assert [row[0] for row in stored_rows(schema_connection)] == sorted(c.id for c in kept)


@needs_database
def test_stored_embeddings_are_the_embedder_output(
    schema_connection: psycopg.Connection[TupleRow],
) -> None:
    chunk = chunk_documents(read_documents(KB_FOLDER))[0]
    store_chunks(schema_connection, [chunk], FakeEmbedder())
    [(stored,)] = schema_connection.execute("SELECT embedding FROM kb_chunks").fetchall()
    expected = FakeEmbedder.vector(embedding_text(chunk)).astype(np.float32)
    assert np.allclose(stored.to_numpy(), expected)


@needs_database
def test_a_changed_chunk_replaces_the_stored_columns_and_embedding(
    schema_connection: psycopg.Connection[TupleRow],
) -> None:
    chunk = chunk_documents(read_documents(KB_FOLDER))[0]
    changed = replace(chunk, title="New title", heading="New heading", text="New text.")
    store_chunks(schema_connection, [chunk], FakeEmbedder())
    report = store_chunks(schema_connection, [changed], FakeEmbedder())
    [(title, heading, text, stored)] = schema_connection.execute(
        "SELECT title, heading, text, embedding FROM kb_chunks"
    ).fetchall()
    expected = FakeEmbedder.vector(embedding_text(changed)).astype(np.float32)
    assert (report.stored, report.removed) == (1, 0)
    assert (title, heading, text) == ("New title", "New heading", "New text.")
    assert np.allclose(stored.to_numpy(), expected)


def test_main_refuses_to_run_without_a_database_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit):
        main([])
    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_main_refuses_an_empty_folder_instead_of_emptying_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    with pytest.raises(SystemExit):
        main(["--kb-dir", str(tmp_path)])
    assert "no knowledge-base documents" in capsys.readouterr().err


def test_the_embedder_loads_the_pinned_model_offline_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, dict[str, object], str | None]] = []

    def record_load(model_id: str, **options: object) -> object:
        loads.append((model_id, options, os.environ.get("DISABLE_SAFETENSORS_CONVERSION")))
        return object()

    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "false")
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=record_load)
    )
    Embedder()
    expected_options = {"revision": MODEL_REVISION, "device": "cpu", "local_files_only": True}
    assert loads == [(MODEL_ID, expected_options, "true")]


@pytest.mark.skipif(not model_is_cached, reason="bge-m3 is not in the local Hugging Face cache")
def test_the_real_model_gives_normalised_cross_lingual_vectors() -> None:
    vectors = Embedder().embed(["ইন্টারনেট প্যাকেজ", "internet package", "আজ আকাশে অনেক মেঘ করেছে"])
    assert vectors.shape == (3, EMBEDDING_DIMENSIONS)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
