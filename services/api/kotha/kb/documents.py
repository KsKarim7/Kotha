import re
from dataclasses import dataclass
from pathlib import Path

NOTICE = "Fictional content for the Kotha demo. Nodi Telecom is not a real operator."
LANGS = ("bn", "en")
FRONT_MATTER_KEYS = ("doc_id", "lang", "title", "notice")
DOC_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class KbDocument:
    doc_id: str
    lang: str
    title: str
    body: str
    path: Path


def split_front_matter(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.split("\n")
    if lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError(f"{path}: the front matter must start and end with a '---' line")
    end = lines.index("---", 1)
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(": ")
        if not separator:
            raise ValueError(f"{path}: front matter line {line!r} is not 'key: value'")
        fields[key] = value
    return fields, "\n".join(lines[end + 1 :]).strip()


def read_document(path: Path) -> KbDocument:
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: not valid UTF-8 ({error})") from error
    fields, body = split_front_matter(text, path)
    missing = [key for key in FRONT_MATTER_KEYS if key not in fields]
    extra = sorted(set(fields) - set(FRONT_MATTER_KEYS))
    if missing or extra:
        raise ValueError(f"{path}: front matter keys missing {missing}, unexpected {extra}")
    doc_id, lang = fields["doc_id"], fields["lang"]
    if lang not in LANGS:
        raise ValueError(f"{path}: unknown lang {lang!r}, expected one of {', '.join(LANGS)}")
    if not DOC_ID_PATTERN.fullmatch(doc_id):
        raise ValueError(f"{path}: doc_id {doc_id!r} is not lowercase kebab-case")
    if path.name != f"{doc_id}.{lang}.md":
        raise ValueError(
            f"{path}: the file name does not match doc_id {doc_id!r} and lang {lang!r}"
        )
    if fields["notice"] != NOTICE:
        raise ValueError(f"{path}: the notice must be exactly {NOTICE!r}")
    if not body:
        raise ValueError(f"{path}: the body is empty")
    return KbDocument(doc_id, lang, fields["title"], body, path)


def read_documents(folder: Path) -> list[KbDocument]:
    return [read_document(path) for path in sorted(folder.glob("*.*.md"))]
