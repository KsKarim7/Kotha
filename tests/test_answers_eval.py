import csv
from pathlib import Path

import httpx2
import pytest
from test_dialogue_answer import GOOD_EN, pipeline_for
from test_llm_client import answer, status

from answers import (
    AnswerResult,
    RateLimitedError,
    RateLimitWatch,
    ask,
    summary,
    write_results,
)
from kotha.dialogue.answer import AnswerPipeline
from kotha.llm.client import LLMRateLimitError
from kotha.llm.fake import FAKE_MODEL
from kotha.llm.gemini import GeminiClient
from retrieval import Question

SIM_QUESTION = Question("q1", "en", "sim-replacement", "what do I bring to replace my SIM", "dev")


def result(
    question_id: str, split: str, kind: str, cited: tuple[str, ...], **extra: str
) -> AnswerResult:
    question = Question(question_id, "en", "sim-replacement", "q", split)
    reason = extra.get("reason", "")
    return AnswerResult(question, kind, reason, (), extra.get("model", ""), cited, ())


def test_an_answer_records_its_model_cited_and_retrieved_documents() -> None:
    pipeline, _, retriever = pipeline_for([GOOD_EN])
    answered = ask(pipeline, retriever, SIM_QUESTION)
    assert (answered.kind, answered.model) == ("answer", FAKE_MODEL)
    assert answered.cited_docs == ("sim-replacement",)
    assert answered.retrieved_docs == ("sim-replacement", "data-rollover")
    assert answered.answered and answered.gold_cited


def test_a_handoff_records_its_reason_and_rules() -> None:
    pipeline, _, retriever = pipeline_for(["No citation here.", "Still none."])
    handed_off = ask(pipeline, retriever, SIM_QUESTION)
    assert (handed_off.kind, handed_off.handoff_reason) == ("handoff", "rejected_twice")
    assert handed_off.rejections == ("missing_citation", "missing_citation")
    assert not handed_off.answered


def test_a_rate_limited_question_stops_the_run() -> None:
    pipeline, _, retriever = pipeline_for([LLMRateLimitError("quota")])
    with pytest.raises(RateLimitedError, match="q1"):
        ask(pipeline, retriever, SIM_QUESTION)


def test_a_rate_limit_on_the_regeneration_stops_the_run() -> None:
    pipeline, _, retriever = pipeline_for(["No citation here.", LLMRateLimitError("quota")])
    with pytest.raises(RateLimitedError):
        ask(pipeline, retriever, SIM_QUESTION)


def test_a_429_hidden_by_a_successful_retry_still_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [status(429), answer(GOOD_EN)]
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        httpx2.HTTPTransport, "handle_request", lambda self, request: responses.pop(0)
    )
    watch = RateLimitWatch()
    llm = GeminiClient(transport=watch)
    _, _, retriever = pipeline_for([])
    pipeline = AnswerPipeline(retriever, llm)
    with pytest.raises(RateLimitedError):
        ask(pipeline, retriever, SIM_QUESTION, watch)
    assert watch.limited == 1
    llm.close()


def test_the_summary_counts_per_split_and_language() -> None:
    results = [
        result("a", "dev", "answer", ("sim-replacement",), model="m1"),
        result("b", "dev", "answer", ("refunds",), model="m2"),
        result("c", "dev", "handoff", (), reason="no_answer"),
        result("d", "test", "handoff", (), reason="llm_timeout"),
    ]
    metrics = summary(results)
    assert metrics["dev.all.answered_rate"] == pytest.approx(2 / 3)
    assert metrics["dev.en.gold_cited_rate"] == 0.5
    assert metrics["dev.all.handoffs.no_answer"] == 1
    assert metrics["dev.all.answer_models.m1"] == 1
    assert metrics["test.all.answered_rate"] == 0.0
    assert metrics["all.all.questions"] == 4
    assert "test.all.gold_cited_rate" not in metrics


def test_results_are_written_one_row_per_question(tmp_path: Path) -> None:
    output = tmp_path / "answers.csv"
    write_results(output, [result("a", "dev", "answer", ("x", "y"), model="m1")])
    with output.open(encoding="utf-8") as written:
        [row] = csv.DictReader(written)
    assert (row["split"], row["model"], row["cited_docs"]) == ("dev", "m1", "x;y")
