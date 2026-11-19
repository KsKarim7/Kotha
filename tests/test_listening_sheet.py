import re
from pathlib import Path

import pytest

from kotha.tts.speakable import VOICE_CHARACTERS
from listening_sheet import (
    REPO_ROOT,
    SENTENCES_PATH,
    Clip,
    clip_plan,
    main,
    read_sentences,
    voice_input,
    write_key,
    write_rating_sheet,
)

MONEY = re.compile(r"টাকা|৳|\btaka\b|\btk\b|\bbdt\b|price|দাম|খরচ", re.IGNORECASE)


def test_committed_sheet_has_20_sentences_without_prices() -> None:
    sentences = read_sentences(SENTENCES_PATH)
    assert len(sentences) == 20
    assert {sentence.lang for sentence in sentences} == {"bn", "en"}
    assert {"data-size", "code", "english-term"} <= {sentence.category for sentence in sentences}
    assert not [sentence.id for sentence in sentences if MONEY.search(sentence.text)]


def test_every_clip_after_normalisation_stays_inside_the_voice_vocabulary() -> None:
    for clip in clip_plan(read_sentences(SENTENCES_PATH)):
        if clip.version == "after":
            assert set(voice_input(clip)) <= VOICE_CHARACTERS[clip.sentence.lang]
        else:
            assert voice_input(clip) == clip.sentence.text


def test_clip_plan_is_shuffled_the_same_way_every_time_with_both_versions() -> None:
    sentences = read_sentences(SENTENCES_PATH)
    plan = clip_plan(sentences)
    assert plan == clip_plan(sentences)
    assert [clip.id for clip in plan] == [f"clip-{number:02d}" for number in range(1, 41)]
    versions = {(clip.sentence.id, clip.version) for clip in plan}
    assert versions == {(s.id, v) for s in sentences for v in ("before", "after")}
    unshuffled = [(s.id, v) for s in sentences for v in ("before", "after")]
    assert [(clip.sentence.id, clip.version) for clip in plan] != unshuffled


def test_rating_sheet_hides_the_version_and_the_key_names_it(tmp_path: Path) -> None:
    plan: list[Clip] = clip_plan(read_sentences(SENTENCES_PATH))
    write_rating_sheet(tmp_path / "ratings.csv", plan)
    write_key(tmp_path / "key.csv", plan)
    sheet = (tmp_path / "ratings.csv").read_text(encoding="utf-8").splitlines()
    key = (tmp_path / "key.csv").read_text(encoding="utf-8").splitlines()
    assert sheet[0] == "clip,lang,intended_text,listener_1,listener_2,listener_3"
    assert "before" not in "".join(sheet) and "after" not in "".join(sheet)
    assert key[0] == "clip,sentence_id,category,version,voice_input"
    assert len(sheet) == len(key) == 41


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("id,lang,text\nx,bn,হ্যাঁ\n", "missing columns: category"),
        ("id,lang,category,text\nx,fr,plain,Oui\n", "unknown lang 'fr'"),
        ("id,lang,category,text\nx,en,plain, \n", "must not be empty"),
        ("id,lang,category,text\nx,en,plain,Yes\nx,en,plain,No\n", "repeated sentence ids"),
    ],
)
def test_malformed_sentence_files_are_refused(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "sentences.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_sentences(path)


def test_main_refuses_an_output_dir_inside_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = REPO_ROOT / "eval" / "listening-out"
    monkeypatch.setattr("sys.argv", ["listening_sheet.py", "--output-dir", str(output_dir)])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    assert not output_dir.exists()
