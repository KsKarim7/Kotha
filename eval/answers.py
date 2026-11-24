import argparse
import csv
import hashlib
import os
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx2

from kotha.db.schema import connect
from kotha.dialogue.answer import AnswerPipeline, HandoffReason
from kotha.kb.chunks import chunk_documents
from kotha.kb.documents import read_documents
from kotha.kb.ingest import KB_FOLDER
from kotha.llm.gemini import ATTEMPT_MODELS, RATE_LIMITED, GeminiClient
from kotha.rag.embedder import Embedder
from kotha.rag.retriever import RESULT_COUNT, HybridRetriever
from kotha.text.lexicon import LEXICON_PATH, read_lexicon
from retrieval import (
    ALL_LANGS,
    LANGS,
    SPLITS,
    TESTSET_PATH,
    Question,
    database_matches_kb,
    read_testset,
    run_params,
)
from tracking import git_tags, log_run, open_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "kotha-answers"
# a few seconds between questions keeps a run under the free tier's per-minute limit
PAUSE_S = 4.0
RESULT_COLUMNS = (
    "question_id",
    "lang",
    "split",
    "gold_doc",
    "kind",
    "handoff_reason",
    "rejections",
    "model",
    "cited_docs",
    "retrieved_docs",
)


class RateLimitedError(Exception):
    pass


class RateLimitWatch(httpx2.HTTPTransport):
    # the client retries a 429 on another model, so only the transport sees every one
    def __init__(self) -> None:
        super().__init__()
        self.limited = 0

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        response = super().handle_request(request)
        if response.status_code == RATE_LIMITED:
            self.limited += 1
        return response


@dataclass(frozen=True)
class AnswerResult:
    question: Question
    kind: str
    handoff_reason: str
    rejections: tuple[str, ...]
    model: str
    cited_docs: tuple[str, ...]
    retrieved_docs: tuple[str, ...]

    @property
    def answered(self) -> bool:
        return self.kind == "answer" and bool(self.cited_docs)

    @property
    def gold_cited(self) -> bool:
        return self.question.doc_id in self.cited_docs


def ask(
    pipeline: AnswerPipeline,
    retriever: HybridRetriever,
    question: Question,
    watch: RateLimitWatch | None = None,
) -> AnswerResult:
    reply = pipeline.reply(question.text)
    if reply.handoff_reason == HandoffReason.LLM_RATE_LIMITED or (watch and watch.limited):
        raise RateLimitedError(question.id)
    retrieved = tuple(dict.fromkeys(r.chunk.doc_id for r in retriever.search(question.text)))
    return AnswerResult(
        question,
        reply.kind.value,
        reply.handoff_reason.value if reply.handoff_reason else "",
        tuple(rule.value for rule in reply.rejections),
        reply.model or "",
        tuple(dict.fromkeys(source.doc_id for source in reply.sources)),
        retrieved,
    )


def summary(results: Sequence[AnswerResult]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split in (ALL_LANGS, *SPLITS):
        for lang in (ALL_LANGS, *LANGS):
            group = [
                result
                for result in results
                if split in (ALL_LANGS, result.question.split)
                and lang in (ALL_LANGS, result.question.lang)
            ]
            if not group:
                continue
            prefix = f"{split}.{lang}"
            answered = [result for result in group if result.answered]
            metrics[f"{prefix}.questions"] = len(group)
            metrics[f"{prefix}.answered_rate"] = len(answered) / len(group)
            if answered:
                gold = sum(result.gold_cited for result in answered)
                metrics[f"{prefix}.gold_cited_rate"] = gold / len(answered)
            reasons = Counter(result.handoff_reason for result in group if result.handoff_reason)
            kinds = Counter(result.kind for result in group if result.kind != "handoff")
            models = Counter(result.model for result in answered)
            metrics |= {f"{prefix}.handoffs.{reason}": n for reason, n in reasons.items()}
            metrics |= {f"{prefix}.kinds.{kind}": n for kind, n in kinds.items()}
            metrics |= {f"{prefix}.answer_models.{model}": n for model, n in models.items()}
    return metrics


def write_results(path: Path, results: Sequence[AnswerResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(RESULT_COLUMNS)
        for result in results:
            question = result.question
            writer.writerow(
                (
                    question.id,
                    question.lang,
                    question.split,
                    question.doc_id,
                    result.kind,
                    result.handoff_reason,
                    ";".join(result.rejections),
                    result.model,
                    ";".join(result.cited_docs),
                    ";".join(result.retrieved_docs),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer the typed questions in text mode.")
    parser.add_argument("--split", choices=(*SPLITS, ALL_LANGS), required=True)
    parser.add_argument("--output", type=Path, required=True, help="per-question CSV")
    parser.add_argument("--mlflow-dir", type=Path)
    parser.add_argument("--pause-s", type=float, default=PAUSE_S)
    args = parser.parse_args()
    output: Path = args.output.resolve()
    for name, path in (("--output", output), ("--mlflow-dir", args.mlflow_dir)):
        if path and path.resolve().is_relative_to(REPO_ROOT):
            parser.error(f"{name} {path} must be outside the repository")
    if args.pause_s < 0:
        parser.error("--pause-s must not be negative")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is not set")
    kb_chunks = chunk_documents(read_documents(KB_FOLDER))
    questions = [
        question
        for question in read_testset(TESTSET_PATH, {chunk.doc_id for chunk in kb_chunks})
        if args.split in (ALL_LANGS, question.split)
    ]
    if not questions:
        parser.error(f"no {args.split} questions in {TESTSET_PATH}")
    output.unlink(missing_ok=True)
    try:
        tags = git_tags(REPO_ROOT)
        tracking = open_experiment(args.mlflow_dir, EXPERIMENT_NAME) if args.mlflow_dir else None
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        parser.error(f"cannot track this run: {error}")
    print(f"questions: {len(questions)}, up to {len(questions) * 6} Gemini requests")
    watch = RateLimitWatch()
    llm = GeminiClient(transport=watch)
    results: list[AnswerResult] = []
    try:
        with connect(database_url) as connection:
            if not database_matches_kb(connection, kb_chunks):
                parser.error("kb_chunks does not match data/kb: run python -m kotha.kb.ingest")
            lexicon = read_lexicon(LEXICON_PATH)
            retriever = HybridRetriever(connection, Embedder(), lexicon)
            pipeline = AnswerPipeline(retriever, llm)
            for number, question in enumerate(questions):
                if number:
                    time.sleep(args.pause_s)
                results.append(ask(pipeline, retriever, question, watch))
                print(f"{question.id}: {results[-1].kind} {results[-1].handoff_reason}".rstrip())
    except RateLimitedError as limited:
        message = f"rate limited at {limited}: this run is invalid and was not logged"
        raise SystemExit(message) from None
    finally:
        llm.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_results(output, results)
    metrics = summary(results)
    for name, value in sorted(metrics.items()):
        print(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
    if tracking is not None:
        client, experiment = tracking
        testset_sha256 = hashlib.sha256(TESTSET_PATH.read_bytes()).hexdigest()
        params = {
            **run_params(TESTSET_PATH, testset_sha256, len(questions), len(lexicon)),
            "split": args.split,
            "models": ",".join(dict.fromkeys(ATTEMPT_MODELS)),
            "result_count": str(RESULT_COUNT),
            "pause_s": str(args.pause_s),
            "input": "typed rag_test questions, text mode (no audio)",
        }
        run_id = log_run(client, experiment, params, metrics, tags, [output])
        print(f"mlflow run: {run_id}")


if __name__ == "__main__":
    main()
