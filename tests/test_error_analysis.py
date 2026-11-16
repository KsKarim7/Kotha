import pytest

from error_analysis import (
    Substitution,
    categorize,
    category_counts,
    substitution_counts,
    top_substitutions,
)

LEXICON = {"মোবাইল": "mobile", "ব্যালেন্স": "balance"}


def test_a_substitution_is_counted() -> None:
    assert substitution_counts([("আমার নম্বর কত", "আমার নাম্বার কত")]) == {("নম্বর", "নাম্বার"): 1}


def test_repeated_pairs_add_up_across_utterances() -> None:
    pairs = [("check my balance", "check my balans"), ("balance koto", "balans koto")]
    assert substitution_counts(pairs) == {("balance", "balans"): 2}


def test_insertions_and_deletions_are_ignored() -> None:
    pairs = [("a b c", "a c"), ("x y", "x y z"), ("p q", "p q")]
    assert substitution_counts(pairs) == {}


def test_a_multi_word_substitute_chunk_is_paired_position_by_position() -> None:
    assert substitution_counts([("a b c d", "a e f d")]) == {("b", "e"): 1, ("c", "f"): 1}


def test_no_pairs_give_no_substitutions() -> None:
    assert substitution_counts([]) == {}
    assert top_substitutions([], LEXICON) == []


@pytest.mark.parametrize(
    ("ref_word", "hyp_word", "category"),
    [
        ("mobile", "মোবাইল", "loanword-script"),
        ("balance", "ব্যালেন্স", "loanword-script"),
        ("30", "তিরিশ", "number"),
        ("30gb", "30", "number"),
        ("mobile", "মোবাইলে", "script"),
        ("নম্বর", "number", "script"),
        ("নম্বর", "নাম্বার", "other"),
        ("balance", "balans", "other"),
        ("package", "packageটা", "other"),
    ],
)
def test_categorize_applies_the_first_matching_rule(
    ref_word: str, hyp_word: str, category: str
) -> None:
    assert categorize(ref_word, hyp_word, LEXICON) == category


def test_loanword_script_needs_the_matching_latin_key() -> None:
    assert categorize("balance", "মোবাইল", LEXICON) == "script"


def test_top_substitutions_sort_by_count_then_ref_then_hyp() -> None:
    pairs = [
        ("x mobile", "y মোবাইল"),
        ("x mobile", "y মোবাইল"),
        ("b", "c"),
        ("b", "a"),
        ("a", "z"),
    ]
    assert top_substitutions(pairs, LEXICON) == [
        Substitution("mobile", "মোবাইল", 2, "loanword-script"),
        Substitution("x", "y", 2, "other"),
        Substitution("a", "z", 1, "other"),
        Substitution("b", "a", 1, "other"),
        Substitution("b", "c", 1, "other"),
    ]


def test_top_substitutions_respects_the_limit() -> None:
    pairs = [(f"w{index}", f"v{index}") for index in range(25)]
    assert len(top_substitutions(pairs, LEXICON)) == 20
    assert [item.ref for item in top_substitutions(pairs, LEXICON, limit=3)] == [
        "w0",
        "w1",
        "w10",
    ]


def test_category_counts_list_every_category_in_a_fixed_order() -> None:
    substitutions = [
        Substitution("mobile", "মোবাইল", 3, "loanword-script"),
        Substitution("a", "b", 1, "other"),
        Substitution("c", "d", 1, "other"),
    ]
    assert category_counts(substitutions) == {
        "loanword-script": 1,
        "script": 0,
        "number": 0,
        "other": 2,
    }
