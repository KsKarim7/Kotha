import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from kotha.kb.documents import KbDocument

MAX_WORDS = 300
BLANK_LINE = re.compile(r"\n[ \t]*\n")


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
    return sum(1 for token in text.split() if any(char.isalnum() for char in token))


def split_sections(body: str, title: str) -> list[tuple[str, str]]:
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    sections: list[tuple[str, list[str]]] = [(title, [])]
    for line in lines:
        if line.startswith("## ") or line == "##":
            sections.append((line[2:].strip() or title, []))
        else:
            sections[-1][1].append(line)
    return [
        (heading, "\n".join(section_lines).strip())
        for heading, section_lines in sections
        if "".join(section_lines).strip()
    ]


def split_blocks(text: str) -> list[str]:
    return [block.strip() for block in BLANK_LINE.split(text) if block.strip()]


def pack_blocks(blocks: Sequence[str]) -> list[str]:
    counts = [word_count(block) for block in blocks]
    remaining = sum(counts)
    pieces: list[list[str]] = []
    words = 0
    target = 0.0
    for block, count in zip(blocks, counts, strict=True):
        too_long = words + count > MAX_WORDS
        overshoots = abs(words + count - target) > abs(words - target)
        if not pieces or too_long or overshoots:
            target = remaining / max(1, math.ceil(remaining / MAX_WORDS))
            pieces.append([])
            words = 0
        pieces[-1].append(block)
        words += count
        remaining -= count
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
