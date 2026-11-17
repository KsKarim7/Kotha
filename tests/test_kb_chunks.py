from pathlib import Path

import pytest

from kotha.kb.chunks import MAX_WORDS, Chunk, chunk_document, chunk_documents, word_count
from kotha.kb.documents import KbDocument, read_documents

KB_FOLDER = Path(__file__).resolve().parent.parent / "data" / "kb"
MIN_WORDS = 150
BANGLA_FLOOR = 120


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


def test_only_the_first_title_line_is_dropped() -> None:
    chunks = chunk_document(document("# Plans\n\n## First\n\nAlpha.\n\n# Other\n\nBeta."))
    assert [chunk.text for chunk in chunks] == ["Alpha.\n\n# Other\n\nBeta."]


@pytest.mark.parametrize("heading_line", ["## ", "##"])
def test_an_empty_heading_falls_back_to_the_title(heading_line: str) -> None:
    chunks = chunk_document(document(f"Intro.\n\n{heading_line}\n\nAlpha."))
    assert [(chunk.heading, chunk.text) for chunk in chunks] == [
        ("Plans", "Intro."),
        ("Plans", "Alpha."),
    ]


@pytest.mark.parametrize(
    ("sizes", "expected"),
    [
        ([100, 100, 100, 100, 10], [200, 210]),
        ([100] * 7, [200, 300, 200]),
        ([100] * 6 + [50], [200, 200, 250]),
    ],
)
def test_a_long_section_splits_into_balanced_chunks(sizes: list[int], expected: list[int]) -> None:
    blocks = [words(size, f"w{n}") for n, size in enumerate(sizes)]
    chunks = chunk_document(document("## Long\n\n" + "\n\n".join(blocks)))
    assert [word_count(chunk.text) for chunk in chunks] == expected


def test_split_chunks_keep_the_heading_and_are_numbered_in_order() -> None:
    blocks = [words(100, f"w{n}") for n in range(4)] + [words(10, "tail")]
    chunks = chunk_document(document("## Long\n\n" + "\n\n".join(blocks)))
    assert [(chunk.id, chunk.heading) for chunk in chunks] == [
        ("plans.en.0", "Long"),
        ("plans.en.1", "Long"),
    ]


def test_blank_lines_with_spaces_still_separate_blocks() -> None:
    chunks = chunk_document(document(f"## Long\n\n{words(200)}\n   \n{words(200)}"))
    assert [word_count(chunk.text) for chunk in chunks] == [200, 200]


def test_a_table_is_never_split_even_when_it_is_long() -> None:
    rows = "\n".join(f"| NT-{n}GB | {words(10)} |" for n in range(40))
    table = f"| Code | Notes |\n| --- | --- |\n{rows}"
    chunks = chunk_document(document(f"## Packs\n\n{words(20)}\n\n{table}\n\n{words(20)}"))
    tables = [chunk.text for chunk in chunks if "| Code | Notes |" in chunk.text]
    assert len(tables) == 1
    assert tables[0].endswith(f"| NT-39GB | {words(10)} |")
    assert word_count(table) > MAX_WORDS


def test_table_markup_does_not_count_as_words() -> None:
    assert word_count("| Code | Data |\n| --- | --- |\n| NT-1GB | 1 GB |") == 5


def without_headings(body: str) -> str:
    lines = [line for line in body.splitlines()[1:] if not line.startswith("## ")]
    return " ".join(" ".join(lines).split())


def test_the_real_knowledge_base_chunks_keep_all_content() -> None:
    for kb_document in read_documents(KB_FOLDER):
        chunks = chunk_document(kb_document)
        joined = " ".join(" ".join(chunk.text for chunk in chunks).split())
        assert joined == without_headings(kb_document.body), kb_document.path.name


def test_the_real_knowledge_base_chunks_stay_within_the_limits() -> None:
    documents = read_documents(KB_FOLDER)
    chunks = chunk_documents(documents)
    sections = sum(
        line.startswith("## ") for document in documents for line in document.body.splitlines()
    )
    assert len(chunks) == sections
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert [chunk.id for chunk in chunks if word_count(chunk.text) > MAX_WORDS] == []
    too_short = [
        chunk.id
        for chunk in chunks
        if word_count(chunk.text) < MIN_WORDS
        and (chunk.lang != "bn" or word_count(chunk.text) < BANGLA_FLOOR)
    ]
    assert too_short == []
