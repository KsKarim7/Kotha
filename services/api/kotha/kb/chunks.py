from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from kotha.kb.documents import KbDocument

MAX_WORDS = 300


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    lang: str
    title: str
    heading: str
    index: int
    text: str


def word_count(text: str) -> int:
    return len(text.split())


def split_sections(body: str, title: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [(title, [])]
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif not line.startswith("# "):
            sections[-1][1].append(line)
    return [
        (heading, "\n".join(lines).strip()) for heading, lines in sections if "".join(lines).strip()
    ]


def split_blocks(text: str) -> list[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def pack_blocks(blocks: Sequence[str], max_words: int = MAX_WORDS) -> list[str]:
    pieces: list[list[str]] = []
    for block in blocks:
        current = pieces[-1] if pieces else None
        if current and word_count("\n\n".join([*current, block])) <= max_words:
            current.append(block)
        else:
            pieces.append([block])
    return ["\n\n".join(piece) for piece in pieces]


def chunk_document(document: KbDocument) -> list[Chunk]:
    texts = [
        (heading, piece)
        for heading, section in split_sections(document.body, document.title)
        for piece in pack_blocks(split_blocks(section))
    ]
    return [
        Chunk(
            id=f"{document.doc_id}.{document.lang}.{index}",
            doc_id=document.doc_id,
            lang=document.lang,
            title=document.title,
            heading=heading,
            index=index,
            text=text,
        )
        for index, (heading, text) in enumerate(texts)
    ]


def chunk_documents(documents: Iterable[KbDocument]) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document)]
