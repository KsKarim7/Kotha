import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from kotha.db.schema import connect, prepare_database
from kotha.kb.chunks import Chunk, chunk_documents
from kotha.kb.documents import read_documents
from kotha.rag.embedder import Embedder, TextEmbedder

KB_FOLDER = Path(__file__).resolve().parents[4] / "data" / "kb"
UPSERT = """
    INSERT INTO kb_chunks (id, doc_id, lang, title, heading, text, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        doc_id = EXCLUDED.doc_id, lang = EXCLUDED.lang, title = EXCLUDED.title,
        heading = EXCLUDED.heading, text = EXCLUDED.text, embedding = EXCLUDED.embedding
"""


@dataclass(frozen=True)
class IngestReport:
    stored: int
    removed: int


def embedding_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.heading}\n\n{chunk.text}"


def store_chunks(
    connection: psycopg.Connection[TupleRow], chunks: Sequence[Chunk], embedder: TextEmbedder
) -> IngestReport:
    vectors = embedder.embed([embedding_text(chunk) for chunk in chunks])
    prepare_database(connection)
    with connection.cursor() as cursor:
        cursor.executemany(
            UPSERT,
            [
                (chunk.id, chunk.doc_id, chunk.lang, chunk.title, chunk.heading, chunk.text, vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        cursor.execute(
            "DELETE FROM kb_chunks WHERE NOT (id = ANY(%s))", ([chunk.id for chunk in chunks],)
        )
        removed = cursor.rowcount
    connection.commit()
    return IngestReport(stored=len(chunks), removed=removed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk, embed and store the knowledge base.")
    parser.add_argument("--kb-dir", type=Path, default=KB_FOLDER)
    args = parser.parse_args(argv)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is not set")
    chunks = chunk_documents(read_documents(args.kb_dir))
    if not chunks:
        parser.error(f"no knowledge-base documents in {args.kb_dir}")
    with connect(database_url) as connection:
        report = store_chunks(connection, chunks, Embedder())
    print(f"stored {report.stored} chunks, removed {report.removed} stale chunks")


if __name__ == "__main__":
    main()
