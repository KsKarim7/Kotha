import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import TupleRow

EMBEDDING_DIMENSIONS = 1024
CONNECT_TIMEOUT_S = 5
STATEMENT_TIMEOUT_MS = 10_000

SCHEMA = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    f"""
    CREATE TABLE IF NOT EXISTS kb_chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        lang TEXT NOT NULL CHECK (lang IN ('bn', 'en')),
        title TEXT NOT NULL,
        heading TEXT NOT NULL,
        text TEXT NOT NULL,
        embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
    )
    """,
)


def connect(database_url: str) -> psycopg.Connection[TupleRow]:
    return psycopg.connect(
        database_url,
        connect_timeout=CONNECT_TIMEOUT_S,
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )


def prepare_database(connection: psycopg.Connection[TupleRow]) -> None:
    for statement in SCHEMA:
        connection.execute(statement)
    register_vector(connection)
