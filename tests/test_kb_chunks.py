from pathlib import Path

from kotha.kb.chunks import MAX_WORDS, Chunk, chunk_document, chunk_documents, word_count
from kotha.kb.documents import NOTICE, KbDocument, read_documents

KB_FOLDER = Path(__file__).resolve().parent.parent / "data" / "kb"
MIN_WORDS = 150


def document(body: str) -> KbDocument:
    return KbDocument("plans", "en", "Plans", body, Path("plans.en.md"))


def words(count: int, word: str = "word") -> str:
    return " ".join([word] * count)


def test_each_heading_becomes_a_chunk_with_its_metadata() -> None:
    chunks = chunk_document(document("# Plans\n\n## First\n\nAlpha text.\n\n## Second\n\nBeta."))
    assert chunks == [
        Chunk("plans.en.0", "plans", "en", "Plans", "First", 0, "Alpha text."),
        Chunk("plans.en.1", "plans", "en", "Plans", "Second", 1, "Beta."),
    ]


def test_text_before_the_first_heading_is_headed_by_the_title() -> None:
    chunks = chunk_document(document("# Plans\n\nIntro words.\n\n## First\n\nAlpha."))
    assert [(chunk.heading, chunk.text) for chunk in chunks] == [
        ("Plans", "Intro words."),
        ("First", "Alpha."),
    ]


def test_a_long_section_splits_at_blank_lines_within_the_limit() -> None:
    body = "## Long\n\n" + "\n\n".join(words(120, f"w{n}") for n in range(4))
    chunks = chunk_document(document(body))
    assert [word_count(chunk.text) for chunk in chunks] == [240, 240]
    assert {chunk.heading for chunk in chunks} == {"Long"}
    assert [chunk.id for chunk in chunks] == ["plans.en.0", "plans.en.1"]


def test_a_table_is_never_split_even_when_it_is_long() -> None:
    rows = "\n".join(f"| NT-{n}GB | {words(10)} |" for n in range(40))
    table = f"| Code | Notes |\n| --- | --- |\n{rows}"
    chunks = chunk_document(document(f"## Packs\n\n{words(20)}\n\n{table}\n\n{words(20)}"))
    tables = [chunk.text for chunk in chunks if "| Code | Notes |" in chunk.text]
    assert len(tables) == 1
    assert tables[0].endswith(f"| NT-39GB | {words(10)} |")
    assert word_count(table) > MAX_WORDS


def test_chunk_ids_are_deterministic() -> None:
    body = "## A\n\nOne.\n\n## B\n\nTwo."
    assert chunk_document(document(body)) == chunk_document(document(body))


def test_the_real_knowledge_base_chunks_stay_within_the_limits() -> None:
    documents = read_documents(KB_FOLDER)
    chunks = chunk_documents(documents)
    sections = sum(document.body.count("\n## ") for document in documents)
    assert len(chunks) == sections
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(word_count(chunk.text) <= MAX_WORDS for chunk in chunks)
    short = [chunk for chunk in chunks if word_count(chunk.text) < MIN_WORDS]
    assert all(chunk.lang == "bn" for chunk in short)
    assert all(chunk.heading for chunk in chunks)
    assert {chunk.lang for chunk in chunks} == {"bn", "en"}
    assert NOTICE not in "".join(chunk.text for chunk in chunks)
