from pathlib import Path

import pytest

from kotha.kb.documents import NOTICE, KbDocument, read_document, read_documents

BODY = "# Title\n\n## Section\n\nSome text.\n"


def front_matter(**changes: str | None) -> dict[str, str]:
    fields = {"doc_id": "plans", "lang": "en", "title": "Plans", "notice": NOTICE} | changes
    return {key: value for key, value in fields.items() if value is not None}


def write_document(
    folder: Path, name: str = "plans.en.md", fields: dict[str, str] | None = None, body: str = BODY
) -> Path:
    lines = ["---", *(f"{key}: {value}" for key, value in (fields or front_matter()).items())]
    path = folder / name
    path.write_text("\n".join([*lines, "---", body]), encoding="utf-8")
    return path


def test_read_document_returns_the_fields_and_the_body(tmp_path: Path) -> None:
    path = write_document(tmp_path)
    assert read_document(path) == KbDocument("plans", "en", "Plans", BODY.strip(), path)


@pytest.mark.parametrize(
    ("name", "fields", "body", "message"),
    [
        ("plans.en.md", front_matter(notice=None), BODY, "missing"),
        ("plans.en.md", front_matter(price="x"), BODY, "unexpected"),
        ("other.en.md", front_matter(), BODY, "does not match"),
        ("plans.bn.md", front_matter(), BODY, "does not match"),
        ("plans.fr.md", front_matter(lang="fr"), BODY, "unknown lang"),
        ("Plans.en.md", front_matter(doc_id="Plans"), BODY, "kebab-case"),
        ("plans.en.md", front_matter(notice="Real content."), BODY, "notice"),
        ("plans.en.md", front_matter(), "\n  \n", "body is empty"),
    ],
)
def test_read_document_refuses_a_broken_file(
    tmp_path: Path, name: str, fields: dict[str, str], body: str, message: str
) -> None:
    path = write_document(tmp_path, name, fields, body)
    with pytest.raises(ValueError, match=message) as refused:
        read_document(path)
    assert str(path) in str(refused.value)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("doc_id: plans\n\n# Title\n", "must start and end"),
        ("---\ndoc_id plans\n---\nbody\n", "is not 'key: value'"),
    ],
)
def test_read_document_refuses_broken_front_matter(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "plans.en.md"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_document(path)


def test_read_document_refuses_invalid_utf8(tmp_path: Path) -> None:
    path = write_document(tmp_path)
    path.write_bytes(path.read_bytes() + b"\xff")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_document(path)


def test_read_documents_sorts_by_path_and_skips_the_readme(tmp_path: Path) -> None:
    write_document(tmp_path, "plans.en.md")
    write_document(tmp_path, "plans.bn.md", front_matter(lang="bn", title="প্ল্যান"))
    (tmp_path / "README.md").write_text("# Knowledge base\n", encoding="utf-8")
    names = [document.path.name for document in read_documents(tmp_path)]
    assert names == ["plans.bn.md", "plans.en.md"]
