import math
from collections import Counter, defaultdict
from collections.abc import Sequence

K1 = 1.5
B = 0.75
EPSILON = 0.25


def inverse_document_frequencies(
    postings: dict[str, list[tuple[int, int]]], document_count: int
) -> dict[str, float]:
    raw_idf = {
        token: math.log(document_count - len(entries) + 0.5) - math.log(len(entries) + 0.5)
        for token, entries in postings.items()
    }
    floor = EPSILON * sum(raw_idf.values()) / len(raw_idf)
    return {token: floor if idf < 0 else idf for token, idf in raw_idf.items()}


class BM25Okapi:
    def __init__(self, documents: Sequence[Sequence[str]]) -> None:
        if not any(documents):
            raise ValueError("BM25 needs at least one document with tokens")
        self._lengths = [len(document) for document in documents]
        self._average_length = sum(self._lengths) / len(documents)
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, document in enumerate(documents):
            for token, frequency in Counter(document).items():
                postings[token].append((index, frequency))
        self._postings = dict(postings)
        self._idf = inverse_document_frequencies(self._postings, len(documents))

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        totals = [0.0] * len(self._lengths)
        for token in query_tokens:
            idf = self._idf.get(token, 0.0)
            for index, frequency in self._postings.get(token, ()):
                relative_length = self._lengths[index] / self._average_length
                length_norm = K1 * (1 - B + B * relative_length)
                totals[index] += idf * frequency * (K1 + 1) / (frequency + length_norm)
        return totals
