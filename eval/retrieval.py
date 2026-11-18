import argparse
import csv
import hashlib
import os
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from kotha.db.schema import connect
from kotha.kb.chunks import Chunk, chunk_documents
from kotha.kb.documents import read_documents
from kotha.kb.ingest import KB_FOLDER
from kotha.rag.embedder import MODEL_ID, MODEL_REVISION, Embedder
from kotha.rag.retriever import (
    CANDIDATE_COUNT,
    RRF_K,
    SELECT_CHUNKS,
    HybridRetriever,
    stored_chunk,
)
from kotha.text.lexicon import LEXICON_PATH, read_lexicon
from tracking import git_tags, log_run, open_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTSET_PATH = REPO_ROOT / "data" / "testsets" / "rag_test.csv"
EXPERIMENT_NAME = "kotha-retrieval"
TESTSET_COLUMNS = ("id", "lang", "doc_id", "question")
LANGS = ("bn", "en", "mix")
ALL_LANGS = "all"
HIT_CUTOFF = 3
RETRIEVER_NAMES = ("dense", "bm25", "hybrid", "hybrid_lexicon")
RESULT_COLUMNS = ("question_id", "lang", "doc_id", "retriever", "rank")

Ranker = Callable[[str], list[str]]


@dataclass(frozen=True)
class Question:
    id: str
    lang: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    retriever: str
    rank: int | None


@dataclass(frozen=True)
class GroupScore:
    retriever: str
    lang: str
    questions: int
    hit_at_3: float
    mrr: float


def read_testset(path: Path, known_doc_ids: Collection[str]) -> list[Question]:
    questions: list[Question] = []
    seen_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as testset_file:
        reader = csv.DictReader(testset_file)
        missing = [name for name in TESTSET_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"test set {path} is missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            if None in row or None in row.values():
                raise ValueError(
                    f"test set {path}, line {line}: expected {len(TESTSET_COLUMNS)} fields"
                )
            question = Question(row["id"], row["lang"], row["doc_id"], row["question"].strip())
            problem = question_problem(question, seen_ids, known_doc_ids)
            if problem:
                raise ValueError(f"test set {path}, line {line}: {problem}")
            seen_ids.add(question.id)
            questions.append(question)
    return questions


def question_problem(
    question: Question, seen_ids: Collection[str], known_doc_ids: Collection[str]
) -> str | None:
    if not question.id or question.id in seen_ids:
        return f"missing or repeated id {question.id!r}"
    if question.lang not in LANGS:
        return f"lang must be one of {', '.join(LANGS)}, got {question.lang!r}"
    if question.doc_id not in known_doc_ids:
        return f"unknown doc_id {question.doc_id!r}"
    if not question.text:
        return "empty question"
    return None


def database_matches_kb(
    connection: psycopg.Connection[TupleRow], kb_chunks: Collection[Chunk]
) -> bool:
    with connection.transaction():
        rows = connection.execute(SELECT_CHUNKS).fetchall()
    return {stored_chunk(row) for row in rows} == set(kb_chunks)


def chunk_doc_id(chunk_id: str) -> str:
    return chunk_id.split(".", 1)[0]


def first_relevant_rank(ranked_chunk_ids: Sequence[str], doc_id: str) -> int | None:
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_doc_id(chunk_id) == doc_id:
            return rank
    return None


def hybrid_ranker(retriever: HybridRetriever, use_lexicon: bool) -> Ranker:
    def ranked_ids(question: str) -> list[str]:
        results = retriever.search(question, limit=CANDIDATE_COUNT, use_lexicon=use_lexicon)
        return [result.chunk.id for result in results]

    return ranked_ids


def build_rankers(retriever: HybridRetriever) -> dict[str, Ranker]:
    return {
        "dense": retriever.dense_ids,
        "bm25": lambda question: retriever.sparse_ids(question, use_lexicon=False),
        "hybrid": hybrid_ranker(retriever, use_lexicon=False),
        "hybrid_lexicon": hybrid_ranker(retriever, use_lexicon=True),
    }


def evaluate(questions: Sequence[Question], rankers: Mapping[str, Ranker]) -> list[QuestionResult]:
    return [
        QuestionResult(question, name, first_relevant_rank(ranker(question.text), question.doc_id))
        for name, ranker in rankers.items()
        for question in questions
    ]


def group_score(retriever: str, lang: str, results: Sequence[QuestionResult]) -> GroupScore:
    ranks = [result.rank for result in results]
    hits = sum(1 for rank in ranks if rank is not None and rank <= HIT_CUTOFF)
    reciprocal_ranks = sum(1 / rank for rank in ranks if rank is not None)
    return GroupScore(retriever, lang, len(ranks), hits / len(ranks), reciprocal_ranks / len(ranks))


def score(results: Sequence[QuestionResult]) -> list[GroupScore]:
    scores: list[GroupScore] = []
    for retriever in RETRIEVER_NAMES:
        retriever_results = [result for result in results if result.retriever == retriever]
        for lang in (ALL_LANGS, *LANGS):
            group = [
                result
                for result in retriever_results
                if lang == ALL_LANGS or result.question.lang == lang
            ]
            if group:
                scores.append(group_score(retriever, lang, group))
    return scores


def group_metrics(scores: Sequence[GroupScore]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group in scores:
        metrics[f"{group.retriever}.{group.lang}.hit_at_3"] = group.hit_at_3
        metrics[f"{group.retriever}.{group.lang}.mrr"] = group.mrr
    return metrics


def run_params(
    testset: Path, testset_sha256: str, question_count: int, lexicon_size: int
) -> dict[str, str]:
    return {
        "testset": testset.name,
        "testset_sha256": testset_sha256,
        "questions": str(question_count),
        "embedding_model": MODEL_ID,
        "embedding_revision": MODEL_REVISION,
        "candidates_per_retriever": str(CANDIDATE_COUNT),
        "rrf_k": str(RRF_K),
        "hit_cutoff": str(HIT_CUTOFF),
        "lexicon": LEXICON_PATH.name,
        "lexicon_spellings": str(lexicon_size),
    }


def write_results(path: Path, results: Sequence[QuestionResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(RESULT_COLUMNS)
        writer.writerows(
            (
                result.question.id,
                result.question.lang,
                result.question.doc_id,
                result.retriever,
                "" if result.rank is None else result.rank,
            )
            for result in results
        )


def print_scores(scores: Sequence[GroupScore]) -> None:
    print("retriever\tlang\tn\thit@3\tmrr")
    for group in scores:
        print(
            f"{group.retriever}\t{group.lang}\t{group.questions}"
            f"\t{group.hit_at_3:.4f}\t{group.mrr:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score retrieval on the typed question set.")
    parser.add_argument("--testset", type=Path, default=TESTSET_PATH)
    parser.add_argument("--output", type=Path, required=True, help="per-question CSV")
    parser.add_argument("--mlflow-dir", type=Path, help="log the run to this MLflow store")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(REPO_ROOT):
        parser.error(f"--output {args.output} must be outside the repository")
    if args.mlflow_dir and args.mlflow_dir.resolve().is_relative_to(REPO_ROOT):
        parser.error(f"--mlflow-dir {args.mlflow_dir} must be outside the repository")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is not set")
    kb_chunks = chunk_documents(read_documents(KB_FOLDER))
    try:
        testset_sha256 = hashlib.sha256(args.testset.read_bytes()).hexdigest()
        questions = read_testset(args.testset, {chunk.doc_id for chunk in kb_chunks})
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not questions:
        parser.error(f"no questions in {args.testset}")
    tags: dict[str, str] = {}
    if args.mlflow_dir:
        try:
            tags = git_tags(REPO_ROOT)
            client, experiment = open_experiment(args.mlflow_dir, EXPERIMENT_NAME)
        except (subprocess.CalledProcessError, ValueError, OSError) as error:
            parser.error(f"cannot track this run: {error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lexicon = read_lexicon(LEXICON_PATH)
    with connect(database_url) as connection:
        if not database_matches_kb(connection, kb_chunks):
            parser.error("kb_chunks differs from data/kb: run python -m kotha.kb.ingest first")
        retriever = HybridRetriever(connection, Embedder(), lexicon)
        results = evaluate(questions, build_rankers(retriever))
    scores = score(results)
    print_scores(scores)
    write_results(output, results)
    if args.mlflow_dir:
        run_id = log_run(
            client,
            experiment,
            run_params(args.testset, testset_sha256, len(questions), len(lexicon)),
            group_metrics(scores),
            tags,
            [output],
        )
        print(f"mlflow run: {run_id}")


if __name__ == "__main__":
    main()
