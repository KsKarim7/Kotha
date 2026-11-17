import math
import os
from collections.abc import Iterator

import numpy as np
import psycopg
import pytest
from psycopg.rows import TupleRow

from kotha.db.schema import EMBEDDING_DIMENSIONS, connect, prepare_database

DATABASE_URL = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL and not os.environ.get("CI"), reason="DATABASE_URL is not set (outside CI)"
)

INSERT = (
    "INSERT INTO kb_chunks (id, doc_id, lang, title, heading, text, embedding) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s)"
)
Connection = psycopg.Connection[TupleRow]


@pytest.fixture
def connection() -> Iterator[Connection]:
    with connect(DATABASE_URL) as database:
        prepare_database(database)
        yield database
        database.rollback()


def unit_vector(position: int) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
    vector[position] = 1.0
    return vector


def insert_chunk(connection: Connection, chunk_id: str, embedding: np.ndarray) -> None:
    doc_id, lang, _ = chunk_id.split(".")
    connection.execute(INSERT, (chunk_id, doc_id, lang, "Title", "Heading", "Text.", embedding))


def test_prepare_database_is_idempotent_and_creates_the_schema(connection: Connection) -> None:
    prepare_database(connection)
    columns = connection.execute(
        "SELECT attname, format_type(atttypid, atttypmod), attnotnull FROM pg_attribute "
        "WHERE attrelid = 'kb_chunks'::regclass AND attnum > 0 AND NOT attisdropped "
        "ORDER BY attnum"
    ).fetchall()
    assert columns == [
        ("id", "text", True),
        ("doc_id", "text", True),
        ("lang", "text", True),
        ("title", "text", True),
        ("heading", "text", True),
        ("text", "text", True),
        ("embedding", f"vector({EMBEDDING_DIMENSIONS})", True),
    ]
    indexes = connection.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname = current_schema() AND tablename = 'kb_chunks'"
    ).fetchall()
    assert indexes == [("kb_chunks_pkey",)]


def test_connections_have_a_statement_timeout(connection: Connection) -> None:
    assert connection.execute("SHOW statement_timeout").fetchone() == ("10s",)


def test_cosine_search_returns_the_nearest_chunk_first(connection: Connection) -> None:
    connection.execute("DELETE FROM kb_chunks")
    insert_chunk(connection, "plans.en.0", unit_vector(0))
    insert_chunk(connection, "plans.en.1", unit_vector(1))
    query = unit_vector(1) + 0.1 * unit_vector(0)
    rows = connection.execute(
        "SELECT id, embedding <=> %s FROM kb_chunks ORDER BY embedding <=> %s",
        (query, query),
    ).fetchall()
    norm = math.sqrt(1.01)
    assert rows == [
        ("plans.en.1", pytest.approx(1 - 1 / norm, abs=1e-5)),
        ("plans.en.0", pytest.approx(1 - 0.1 / norm, abs=1e-5)),
    ]


def test_an_embedding_of_the_wrong_dimension_is_refused(connection: Connection) -> None:
    with pytest.raises(psycopg.errors.DataException, match="expected 1024 dimensions"):
        insert_chunk(connection, "plans.en.0", np.ones(3, dtype=np.float32))


def test_an_unknown_language_is_refused(connection: Connection) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_chunk(connection, "plans.fr.0", unit_vector(0))
