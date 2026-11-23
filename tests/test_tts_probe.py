import csv
import unicodedata
from pathlib import Path

import pytest

from kotha.dialogue.routing import Language
from tts_probe import (
    REPO_ROOT,
    SENTENCES,
    VOICES,
    ProbeResult,
    ProbeSentence,
    character_label,
    dropped_characters,
    listening_sentences,
    main,
    real_time_factors,
    write_timings,
)

BANGLA_VOCAB = {"প", "্", "য", "া", "ক", " ", "3", "0", "6"}
BANGLA_BLANK = "6"


def test_dropped_characters_lists_each_missing_character_once() -> None:
    assert dropped_characters("প্যাক NT-30।।", BANGLA_VOCAB, BANGLA_BLANK) == ("-", "n", "t", "।")


def test_dropped_characters_compares_lowercased_text() -> None:
    assert dropped_characters("Ab", {"a", "b", "k"}, "k") == ()


def test_dropped_characters_is_empty_for_text_inside_the_vocab() -> None:
    assert dropped_characters("প্যাক 30", BANGLA_VOCAB, BANGLA_BLANK) == ()


def test_dropped_characters_counts_the_blank_token_although_it_is_in_the_vocab() -> None:
    assert dropped_characters("প্যাক 6", BANGLA_VOCAB, BANGLA_BLANK) == ("6",)
    assert dropped_characters("Pack", {"p", "a", "c", "k"}, "k") == ("k",)


def test_precomposed_yya_is_outside_a_vocab_that_has_only_ya_and_nukta() -> None:
    assert dropped_characters("য়", {"য", "়"}, BANGLA_BLANK) == ("য়",)


def test_character_label_names_the_code_point() -> None:
    assert character_label("।") == "U+0964 DEVANAGARI DANDA"
    assert character_label("#") == "U+0023 NUMBER SIGN"


def test_probe_sentences_have_unique_ids_and_known_voices() -> None:
    ids = [sentence.id for sentence in SENTENCES]
    assert len(ids) == len(set(ids))
    assert {sentence.lang for sentence in SENTENCES} == set(VOICES)
    assert all(sentence.id.startswith(f"{sentence.lang}-") for sentence in SENTENCES)


def test_only_the_precomposed_sentence_is_not_in_nfc() -> None:
    not_nfc = [s.id for s in SENTENCES if unicodedata.normalize("NFC", s.text) != s.text]
    assert not_nfc == ["bn-precomposed-ya"]
    precomposed = next(s for s in SENTENCES if s.id == "bn-precomposed-ya")
    assert "য়" in precomposed.text


def test_voices_are_pinned_to_full_commit_hashes() -> None:
    for _, revision in VOICES.values():
        assert len(revision) == 40
        int(revision, 16)


def test_the_listening_sentences_are_timed_as_the_app_would_speak_them() -> None:
    sentences = listening_sentences()
    assert len(sentences) == 20
    assert {sentence.lang for sentence in sentences} == {"bn", "en"}
    assert all(not any(char.isdigit() for char in sentence.text) for sentence in sentences)


def probe_result(
    sentence_id: str, lang: Language, audio_s: float, synthesis_s: float
) -> ProbeResult:
    return ProbeResult(ProbeSentence(sentence_id, lang, "x"), "x", (), audio_s, synthesis_s, None)


def test_real_time_factors_are_summarised_per_voice_and_written_per_sentence(
    tmp_path: Path,
) -> None:
    results = [
        probe_result("b1", "bn", 2.0, 1.0),
        probe_result("b2", "bn", 2.0, 3.0),
        probe_result("e1", "en", 4.0, 1.0),
        probe_result("e2", "en", 0.0, 1.0),
    ]
    metrics = real_time_factors(results)
    assert metrics["rtf.bn.p50"] == pytest.approx(1.0)
    assert metrics["rtf.bn.sentences"] == 2
    assert (metrics["rtf.en.p50"], metrics["rtf.en.sentences"]) == (0.25, 1)
    write_timings(tmp_path / "timings.csv", results)
    with (tmp_path / "timings.csv").open(encoding="utf-8") as written:
        rows = list(csv.DictReader(written))
    assert [row["id"] for row in rows] == ["b1", "b2", "e1", "e2"]
    assert float(rows[1]["rtf"]) == 1.5 and rows[3]["rtf"] == ""


def test_main_refuses_an_output_dir_inside_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = REPO_ROOT / "eval" / "probe-out"
    monkeypatch.setattr("sys.argv", ["tts_probe.py", "--output-dir", str(output_dir)])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    assert not output_dir.exists()
