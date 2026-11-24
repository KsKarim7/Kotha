import json
import re
from pathlib import Path

import pytest

from kotha.dialogue.answer import AnswerPipeline, BotReply, ReplyKind, Source
from kotha.dialogue.routing import caller_language, is_account_request
from kotha.dialogue.scenarios import (
    SCENARIOS_PATH,
    Scenario,
    Turn,
    read_scenarios,
    run_scenarios,
    turn_failures,
)
from kotha.kb.chunks import chunk_documents
from kotha.kb.documents import read_documents
from kotha.kb.ingest import KB_FOLDER
from kotha.llm.client import Generation
from kotha.llm.fake import FAKE_MODEL, FakeLLMCall
from kotha.rag.retriever import RetrievedChunk, SparseRetriever
from kotha.text.lexicon import LEXICON_PATH, read_lexicon

KB_CHUNKS = {chunk.id: chunk for chunk in chunk_documents(read_documents(KB_FOLDER))}
PASSAGE_ID = re.compile(r"^\d+\. \[([^\]]+)\]$", re.MULTILINE)


class KbSparseRetriever:
    def __init__(self) -> None:
        self._sparse = SparseRetriever(list(KB_CHUNKS.values()), read_lexicon(LEXICON_PATH))

    def search(self, query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(KB_CHUNKS[chunk_id], 0.0)
            for chunk_id in self._sparse.ranked_ids(query, 4)
        ]


class CiteEveryPassageLLM:
    def __init__(self) -> None:
        self.calls: list[FakeLLMCall] = []

    def generate(self, system_prompt: str, user_prompt: str) -> Generation:
        self.calls.append(FakeLLMCall(system_prompt, user_prompt))
        citations = " ".join(f"[{chunk_id}]" for chunk_id in PASSAGE_ID.findall(user_prompt))
        if "Reply language: Bangla" in user_prompt:
            return Generation(f"নিয়মটি এখানে আছে {citations}।", FAKE_MODEL)
        return Generation(f"The rule is here {citations}.", FAKE_MODEL)


def answer_from(doc_id: str, lang: str = "en") -> BotReply:
    return BotReply(ReplyKind.ANSWER, lang, "text", (Source(f"{doc_id}.{lang}.0", doc_id, "T"),))


def test_the_committed_scenarios_cover_policy_questions_in_every_language() -> None:
    scenarios = read_scenarios(SCENARIOS_PATH)
    ids = {scenario.id for scenario in scenarios}
    for lang in ("bn", "en", "mix"):
        assert any(id.startswith("uc4-") and id.endswith(f"-{lang}") for id in ids), lang
    doc_ids = {chunk.doc_id for chunk in KB_CHUNKS.values()}
    turns = [turn for scenario in scenarios for turn in scenario.turns]
    assert {turn.doc_id for turn in turns if turn.doc_id} <= doc_ids


def test_the_committed_turns_agree_with_the_routing_rules() -> None:
    for scenario in read_scenarios(SCENARIOS_PATH):
        for turn in scenario.turns:
            expect_account = turn.kind == ReplyKind.ACCOUNT_UNAVAILABLE
            assert is_account_request(turn.caller) == expect_account, turn.caller
            assert caller_language(turn.caller) == turn.lang, turn.caller


def test_the_committed_scenarios_pass_with_sparse_retrieval_and_a_citing_llm() -> None:
    pipeline = AnswerPipeline(KbSparseRetriever(), CiteEveryPassageLLM())
    lines: list[str] = []
    assert run_scenarios(pipeline, read_scenarios(SCENARIOS_PATH), lines.append) == 0
    assert lines[-1].endswith("turns passed")


def test_a_failed_turn_is_counted_and_reported() -> None:
    pipeline = AnswerPipeline(KbSparseRetriever(), CiteEveryPassageLLM())
    scenario = Scenario(
        "wrong",
        (
            Turn("what is my balance", ReplyKind.ANSWER, "en", "balance-check"),
            Turn("what is my balance", ReplyKind.ACCOUNT_UNAVAILABLE, "en", None),
        ),
    )
    lines: list[str] = []
    assert run_scenarios(pipeline, [scenario], lines.append) == 1
    assert "FAIL wrong turn 1" in lines
    assert "pass wrong turn 2" in lines
    assert lines[-1] == "1 of 2 turns passed"


@pytest.mark.parametrize(
    ("turn", "reply", "failures"),
    [
        (Turn("q", ReplyKind.ANSWER, "en", "roaming"), answer_from("roaming"), []),
        (
            Turn("q", ReplyKind.ANSWER, "en", "roaming"),
            answer_from("refunds"),
            ["expected a citation of roaming"],
        ),
        (
            Turn("q", ReplyKind.ANSWER, "bn", "roaming"),
            answer_from("roaming"),
            ["expected lang bn, got en"],
        ),
        (
            Turn("q", ReplyKind.REPEAT, "en", None),
            answer_from("roaming"),
            ["expected repeat, got answer"],
        ),
    ],
)
def test_turn_failures_name_each_mismatch(turn: Turn, reply: BotReply, failures: list[str]) -> None:
    assert turn_failures(turn, reply) == failures


@pytest.mark.parametrize(
    ("turn", "message"),
    [
        ({"caller": "q", "kind": "answer", "lang": "en"}, "doc_id is required"),
        ({"caller": "q", "kind": "repeat", "lang": "en", "doc_id": "x"}, "doc_id is required"),
        ({"caller": "q", "kind": "repeat", "lang": "mix"}, "unknown lang"),
        ({"caller": "q", "kind": "shout", "lang": "en"}, "s turn 1: unknown kind 'shout'"),
        ({"caller": "q", "lang": "en"}, "s turn 1: missing kind"),
    ],
)
def test_a_malformed_scenario_file_is_refused(
    tmp_path: Path, turn: dict[str, str], message: str
) -> None:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": [{"id": "s", "turns": [turn]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_scenarios(path)


def test_duplicate_scenario_ids_are_refused(tmp_path: Path) -> None:
    scenario = {"id": "s", "turns": []}
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": [scenario, scenario]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        read_scenarios(path)
