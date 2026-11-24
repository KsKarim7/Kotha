import argparse
import csv
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTSET_PATH = REPO_ROOT / "data" / "testsets" / "rag_test.csv"
SPLIT_SEED = 20261124
SPLITS = ("dev", "test")


def assign_splits(rows: Sequence[dict[str, str]], seed: int = SPLIT_SEED) -> dict[str, str]:
    rng = random.Random(seed)
    by_document: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_document[row["doc_id"]].append(row)
    counts = {split: 0 for split in SPLITS}
    splits: dict[str, str] = {}
    for doc_id in sorted(by_document):
        questions = sorted(by_document[doc_id], key=lambda row: row["id"])
        rng.shuffle(questions)
        # languages alternate inside a document, so each split gets a share of each language
        questions.sort(key=lambda row: row["lang"])
        split = min(SPLITS, key=lambda name: (counts[name], rng.random()))
        for row in questions:
            splits[row["id"]] = split
            counts[split] += 1
            split = SPLITS[1 - SPLITS.index(split)]
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a fixed dev/test split column to rag_test.")
    parser.add_argument("--testset", type=Path, default=TESTSET_PATH)
    args = parser.parse_args()
    with args.testset.open(newline="", encoding="utf-8-sig") as testset_file:
        reader = csv.DictReader(testset_file)
        columns = [name for name in reader.fieldnames or [] if name != "split"]
        rows = list(reader)
    splits = assign_splits(rows)
    with args.testset.open("w", newline="", encoding="utf-8") as testset_file:
        writer = csv.DictWriter(testset_file, fieldnames=[*columns, "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({**row, "split": splits[row["id"]]} for row in rows)


if __name__ == "__main__":
    main()
