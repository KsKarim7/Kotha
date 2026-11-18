from kotha.text.lexicon import LEXICON_PATH, expand_tokens, read_lexicon, spelling_groups

LEXICON = {"ব্যালেন্স": "balance", "ব্যালান্স": "balance", "সিম": "sim"}
GROUPS = spelling_groups(LEXICON)


def test_every_spelling_points_to_its_whole_group() -> None:
    balance = frozenset({"balance", "ব্যালেন্স", "ব্যালান্স"})
    assert GROUPS["balance"] == GROUPS["ব্যালেন্স"] == GROUPS["ব্যালান্স"] == balance
    assert GROUPS["sim"] == frozenset({"sim", "সিম"})


def test_a_latin_word_expands_to_every_bangla_spelling() -> None:
    assert expand_tokens(["balance"], GROUPS) == ["balance", "ব্যালান্স", "ব্যালেন্স"]


def test_a_bangla_word_expands_to_latin_and_its_other_spellings() -> None:
    assert expand_tokens(["ব্যালান্স"], GROUPS) == ["ব্যালান্স", "balance", "ব্যালেন্স"]


def test_the_query_tokens_stay_first_and_keep_their_repeats() -> None:
    assert expand_tokens(["sim", "kothay", "sim"], GROUPS) == ["sim", "kothay", "sim", "সিম"]


def test_a_spelling_already_in_the_query_is_not_added_again() -> None:
    assert expand_tokens(["sim", "সিম"], GROUPS) == ["sim", "সিম"]


def test_words_outside_the_lexicon_are_left_alone() -> None:
    assert expand_tokens(["roaming"], GROUPS) == ["roaming"]
    assert expand_tokens([], GROUPS) == []


def test_the_committed_lexicon_expands_in_both_directions() -> None:
    groups = spelling_groups(read_lexicon(LEXICON_PATH))
    assert "ইন্টারনেট" in expand_tokens(["internet"], groups)
    assert "internet" in expand_tokens(["ইন্টারনেট"], groups)
