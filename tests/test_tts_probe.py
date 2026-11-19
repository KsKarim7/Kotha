import pytest

from tts_probe import SENTENCES, VOICES, character_label, dropped_characters, main

BANGLA_VOCAB = {"প", "্", "য", "া", "ক", " ", "3", "0"}


def test_dropped_characters_lists_each_missing_character_once() -> None:
    assert dropped_characters("প্যাক NT-30।।", BANGLA_VOCAB) == ("-", "n", "t", "।")


def test_dropped_characters_compares_lowercased_text() -> None:
    assert dropped_characters("Ab", {"a", "b"}) == ()


def test_dropped_characters_is_empty_for_text_inside_the_vocab() -> None:
    assert dropped_characters("প্যাক 30", BANGLA_VOCAB) == ()


def test_precomposed_yya_is_outside_a_vocab_that_has_only_ya_and_nukta() -> None:
    assert dropped_characters("য়", {"য", "়"}) == ("য়",)


def test_character_label_names_the_code_point() -> None:
    assert character_label("।") == "U+0964 DEVANAGARI DANDA"
    assert character_label("#") == "U+0023 NUMBER SIGN"


def test_probe_sentences_have_unique_ids_and_known_voices() -> None:
    ids = [sentence.id for sentence in SENTENCES]
    assert len(ids) == len(set(ids))
    assert {sentence.lang for sentence in SENTENCES} == set(VOICES)
    assert all(sentence.id.startswith(f"{sentence.lang}-") for sentence in SENTENCES)


def test_voices_are_pinned_to_full_commit_hashes() -> None:
    for _, revision in VOICES.values():
        assert len(revision) == 40
        int(revision, 16)


def test_main_refuses_an_output_dir_inside_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["tts_probe.py", "--output-dir", "eval/probe-out"])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
