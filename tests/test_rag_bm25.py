import math

import pytest

from kotha.rag.bm25 import BM25Okapi

DOCUMENTS = [
    ["apple", "apple", "pear"],
    ["apple", "kiwi"],
    ["apple", "plum"],
    ["apple", "fig"],
    ["grape"],
]
LOG_3 = math.log(3)
MEAN_IDF = (-LOG_3 + 5 * LOG_3) / 6
APPLE_IDF_FLOOR = 0.25 * MEAN_IDF
AVERAGE_LENGTH = 10 / 5
NORM_LENGTH_3 = 1.5 * (1 - 0.75 + 0.75 * 3 / AVERAGE_LENGTH)
NORM_LENGTH_2 = 1.5 * (1 - 0.75 + 0.75 * 2 / AVERAGE_LENGTH)


def test_a_common_word_gets_the_epsilon_idf_floor() -> None:
    scores = BM25Okapi(DOCUMENTS).scores(["apple"])
    assert scores == pytest.approx(
        [
            APPLE_IDF_FLOOR * 2 * 2.5 / (2 + NORM_LENGTH_3),
            APPLE_IDF_FLOOR * 1 * 2.5 / (1 + NORM_LENGTH_2),
            APPLE_IDF_FLOOR * 1 * 2.5 / (1 + NORM_LENGTH_2),
            APPLE_IDF_FLOOR * 1 * 2.5 / (1 + NORM_LENGTH_2),
            0.0,
        ]
    )


def test_scores_add_up_over_the_query_words() -> None:
    scores = BM25Okapi(DOCUMENTS).scores(["apple", "pear"])
    apple = APPLE_IDF_FLOOR * 2 * 2.5 / (2 + NORM_LENGTH_3)
    pear = LOG_3 * 1 * 2.5 / (1 + NORM_LENGTH_3)
    assert scores[0] == pytest.approx(apple + pear)


def test_a_repeated_query_word_counts_each_time() -> None:
    index = BM25Okapi(DOCUMENTS)
    assert index.scores(["pear", "pear"])[0] == pytest.approx(2 * index.scores(["pear"])[0])


def test_an_unknown_word_scores_nothing() -> None:
    assert BM25Okapi(DOCUMENTS).scores(["mango"]) == [0.0] * len(DOCUMENTS)


def test_a_word_in_exactly_half_the_documents_has_zero_idf() -> None:
    index = BM25Okapi([["shared", "a"], ["shared", "b"], ["c"], ["d"]])
    assert index.scores(["shared"]) == pytest.approx([0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("documents", [[], [[]], [[], []]])
def test_a_corpus_without_tokens_is_refused(documents: list[list[str]]) -> None:
    with pytest.raises(ValueError, match="at least one document"):
        BM25Okapi(documents)
