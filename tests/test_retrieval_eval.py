import contextlib
import csv
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest

import retrieval
from kotha.kb.chunks import chunk_documents
from kotha.kb.documents import read_documents
from kotha.kb.ingest import KB_FOLDER, embedding_text
from kotha.rag.retriever import HybridRetriever, search_tokens
from kotha.text.lexicon import BENGALI_BLOCK
from retrieval import (
    REPO_ROOT,
    TESTSET_PATH,
    GroupScore,
    Question,
    QuestionResult,
    build_rankers,
    database_matches_kb,
    evaluate,
    first_relevant_rank,
    group_metrics,
    main,
    read_testset,
    score,
)
from split_testset import assign_splits
from tracking import open_store

PACKAGE_CODES = (
    "NT-1GB",
    "NT-5GB",
    "NT-15GB",
    "NT-30GB",
    "NT-60GB",
    "NT-NIGHT-5GB",
    "NV-100",
    "NV-300",
    "NC-10GB",
    "NC-25GB",
)
COPIED_PHRASE_WORDS = 5
KB_CHUNKS = chunk_documents(read_documents(KB_FOLDER))
KB_DOC_IDS = {chunk.doc_id for chunk in KB_CHUNKS}
KB_ROWS = [
    (chunk.id, chunk.doc_id, chunk.lang, chunk.title, chunk.heading, chunk.text)
    for chunk in KB_CHUNKS
]
QUESTIONS = read_testset(TESTSET_PATH, KB_DOC_IDS)
BALANCE_EN = Question("q1", "en", "balance", "how do I check")
ROAMING_BN = Question("q2", "bn", "roaming", "বিদেশে")


def word_runs(tokens: list[str]) -> set[tuple[str, ...]]:
    last_start = len(tokens) - COPIED_PHRASE_WORDS + 1
    return {tuple(tokens[start : start + COPIED_PHRASE_WORDS]) for start in range(last_start)}


def write_testset(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(["id,lang,doc_id,question", *rows]) + "\n", encoding="utf-8")
    return path


def result(question: Question, retriever: str, rank: int | None) -> QuestionResult:
    return QuestionResult(question, retriever, rank)


def test_the_committed_set_has_enough_questions_per_language() -> None:
    counts = Counter(question.lang for question in QUESTIONS)
    assert len(QUESTIONS) >= 60
    assert all(counts[lang] >= 15 for lang in ("bn", "en", "mix"))


def test_the_committed_split_is_the_seeded_one_and_balanced() -> None:
    with TESTSET_PATH.open(newline="", encoding="utf-8") as testset_file:
        rows = list(csv.DictReader(testset_file))
    assert {row["id"]: row["split"] for row in rows} == assign_splits(rows)
    splits = Counter(question.split for question in QUESTIONS)
    assert splits == {"dev": 36, "test": 36}
    for lang in ("bn", "en", "mix"):
        dev = sum(q.split == "dev" for q in QUESTIONS if q.lang == lang)
        assert 10 <= dev <= 14, lang
    for doc_id in KB_DOC_IDS:
        assert {q.split for q in QUESTIONS if q.doc_id == doc_id} == {"dev", "test"}, doc_id


def test_read_testset_refuses_an_unknown_split(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    path.write_text("id,lang,doc_id,question,split\nq1,en,roaming,abroad,train\n", "utf-8")
    with pytest.raises(ValueError, match="unknown split"):
        read_testset(path, {"roaming"})


def test_scores_and_metric_names_are_given_per_split_too() -> None:
    dev = Question("q1", "en", "balance", "x", "dev")
    test = Question("q2", "en", "balance", "y", "test")
    results = [QuestionResult(dev, "dense", 1), QuestionResult(test, "dense", None)]
    by_split = {(s.split, s.lang): s.hit_at_3 for s in score(results) if s.retriever == "dense"}
    assert by_split[("all", "all")] == 0.5
    assert (by_split[("dev", "en")], by_split[("test", "en")]) == (1.0, 0.0)
    metrics = group_metrics(score(results))
    assert metrics["dense.all.hit_at_3"] == 0.5 and metrics["dense.dev.en.hit_at_3"] == 1.0


def test_every_document_is_asked_about_in_every_language() -> None:
    for lang in ("bn", "en", "mix"):
        asked = {question.doc_id for question in QUESTIONS if question.lang == lang}
        assert asked == KB_DOC_IDS


def test_every_package_code_is_asked_about() -> None:
    assert all(any(code in question.text for question in QUESTIONS) for code in PACKAGE_CODES)


def test_each_question_is_written_in_its_language() -> None:
    for question in QUESTIONS:
        has_bangla = any(ord(char) in BENGALI_BLOCK for char in question.text)
        has_latin = any(char.isascii() and char.isalpha() for char in question.text)
        assert (has_bangla, has_latin) == {
            "bn": (True, False),
            "en": (False, True),
            "mix": (True, True),
        }[question.lang], question.id


def test_no_question_copies_a_phrase_from_the_knowledge_base() -> None:
    kb_runs: set[tuple[str, ...]] = set()
    for chunk in KB_CHUNKS:
        kb_runs |= word_runs(search_tokens(embedding_text(chunk)))
    copied = [q.id for q in QUESTIONS if word_runs(search_tokens(q.text)) & kb_runs]
    assert copied == []


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("q1,de,balance,hello", "lang must be one of"),
        ("q1,en,unknown,hello", "unknown doc_id"),
        ("q1,en,balance,  ", "empty question"),
        (",en,balance,hello", "missing or repeated id"),
        ("q1,en,balance", "expected 4 fields"),
        ("q1,en,balance,how much is left, and until when", "expected 4 fields"),
    ],
)
def test_read_testset_rejects_bad_rows(tmp_path: Path, row: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_testset(write_testset(tmp_path / "set.csv", [row]), {"balance"})


def test_read_testset_rejects_a_repeated_id(tmp_path: Path) -> None:
    rows = ["q1,en,balance,one", "q1,en,balance,two"]
    with pytest.raises(ValueError, match="line 3: missing or repeated id"):
        read_testset(write_testset(tmp_path / "set.csv", rows), {"balance"})


def test_read_testset_names_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    path.write_text("id,question\nq1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns: lang, doc_id"):
        read_testset(path, {"balance"})


def test_the_rank_is_the_first_chunk_from_the_right_document() -> None:
    ranked = ["roaming.en.0", "balance.bn.2", "balance.en.0"]
    assert first_relevant_rank(ranked, "balance") == 2
    assert first_relevant_rank(ranked, "refunds") is None
    assert first_relevant_rank([], "balance") is None


def test_a_document_id_must_match_exactly() -> None:
    assert first_relevant_rank(["sim-replacement.en.0"], "sim") is None


def test_scores_are_computed_by_hand_overall_and_per_language() -> None:
    other_en = Question("q3", "en", "refunds", "refund")
    results = [
        result(BALANCE_EN, "dense", 1),
        result(ROAMING_BN, "dense", 4),
        result(other_en, "dense", None),
    ]
    assert score(results) == [
        GroupScore("dense", "all", 3, pytest.approx(1 / 3), pytest.approx((1 + 1 / 4) / 3)),
        GroupScore("dense", "bn", 1, 0.0, 0.25),
        GroupScore("dense", "en", 2, 0.5, 0.5),
    ]


def test_rank_three_is_a_hit_and_rank_four_is_not() -> None:
    results = [result(BALANCE_EN, "bm25", 3), result(ROAMING_BN, "bm25", 4)]
    assert score(results)[0].hit_at_3 == 0.5


def test_metric_names_carry_retriever_and_language() -> None:
    metrics = group_metrics([GroupScore("hybrid", "mix", 2, 0.5, 0.75)])
    assert metrics == {"hybrid.mix.hit_at_3": 0.5, "hybrid.mix.mrr": 0.75}


def fake_retriever() -> HybridRetriever:
    retriever = create_autospec(HybridRetriever, instance=True)
    retriever.dense_ids.side_effect = lambda query, limit=20: ["balance.en.0"]
    retriever.sparse_ids.side_effect = lambda query, limit=20, use_lexicon=True: ["roaming.en.0"]

    def search(query: str, limit: int = 4, use_lexicon: bool = True) -> list[SimpleNamespace]:
        chunk_id = "balance.bn.0" if use_lexicon else "refunds.en.0"
        return [SimpleNamespace(chunk=SimpleNamespace(id=chunk_id))]

    retriever.search.side_effect = search
    return retriever


def test_each_retriever_is_called_with_its_own_settings() -> None:
    retriever = fake_retriever()
    rankers = build_rankers(retriever)
    assert {name: ranker("question") for name, ranker in rankers.items()} == {
        "dense": ["balance.en.0"],
        "bm25": ["roaming.en.0"],
        "hybrid": ["refunds.en.0"],
        "hybrid_lexicon": ["balance.bn.0"],
    }
    retriever.sparse_ids.assert_called_once_with("question", use_lexicon=False)
    retriever.search.assert_any_call("question", limit=20, use_lexicon=False)
    retriever.search.assert_any_call("question", limit=20, use_lexicon=True)


def test_evaluate_ranks_every_question_for_every_retriever() -> None:
    results = evaluate([BALANCE_EN], build_rankers(fake_retriever()))
    assert {item.retriever: item.rank for item in results} == {
        "dense": 1,
        "bm25": None,
        "hybrid": None,
        "hybrid_lexicon": 1,
    }


def connection_with_rows(rows: list[tuple[str, ...]]) -> MagicMock:
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = rows
    return connection


def use_stored_rows(monkeypatch: pytest.MonkeyPatch, rows: list[tuple[str, ...]]) -> None:
    connection = connection_with_rows(rows)
    monkeypatch.setattr(retrieval, "connect", lambda url: contextlib.nullcontext(connection))


def test_the_database_must_hold_exactly_the_current_chunks() -> None:
    assert database_matches_kb(connection_with_rows(KB_ROWS), KB_CHUNKS)
    assert not database_matches_kb(connection_with_rows(KB_ROWS[1:]), KB_CHUNKS)
    edited_row = (*KB_ROWS[0][:5], "Edited text.")
    assert not database_matches_kb(connection_with_rows([edited_row, *KB_ROWS[1:]]), KB_CHUNKS)


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    use_stored_rows(monkeypatch, KB_ROWS)
    monkeypatch.setattr(retrieval, "Embedder", lambda: "embedder")
    monkeypatch.setattr(retrieval, "HybridRetriever", lambda *args: fake_retriever())
    return ["--output", str(tmp_path / "out" / "retrieval.csv")]


def test_main_prints_scores_and_writes_one_row_per_question_and_retriever(
    cli: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(cli)
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "split\tretriever\tlang\tn\thit@3\tmrr"
    assert any(line.startswith(f"all\tdense\tall\t{len(QUESTIONS)}\t") for line in printed)
    with (tmp_path / "out" / "retrieval.csv").open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 4 * len(QUESTIONS)
    assert "question" not in rows[0]


def test_main_logs_one_run_to_its_own_experiment(
    cli: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setattr(retrieval, "git_tags", lambda repo: {"git_dirty": "false"})
    store = tmp_path / "mlflow"
    main([*cli, "--mlflow-dir", str(store)])
    run_id = capsys.readouterr().out.splitlines()[-1].removeprefix("mlflow run: ")
    client = open_store(store)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == "kotha-retrieval"
    assert run.data.params["questions"] == str(len(QUESTIONS))
    expected_digest = hashlib.sha256(TESTSET_PATH.read_bytes()).hexdigest()
    assert run.data.params["testset_sha256"] == expected_digest
    assert run.data.tags["git_dirty"] == "false"
    assert "hybrid_lexicon.all.hit_at_3" in run.data.metrics


@pytest.mark.parametrize("option", ["--output", "--mlflow-dir"])
def test_main_refuses_paths_inside_the_repository(
    cli: list[str], option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main([*cli, option, str(REPO_ROOT / "results")])
    assert f"{option} " in capsys.readouterr().err


def test_main_refuses_to_run_without_a_database_url(
    cli: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(SystemExit):
        main(cli)
    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_main_refuses_a_database_that_differs_from_the_knowledge_base(
    cli: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_stored_rows(monkeypatch, KB_ROWS[1:])
    with pytest.raises(SystemExit):
        main(cli)
    assert "kb_chunks differs from data/kb" in capsys.readouterr().err


def test_main_refuses_an_invalid_test_set(
    cli: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    testset = write_testset(tmp_path / "set.csv", ["q1,en,unknown,hello"])
    with pytest.raises(SystemExit):
        main([*cli, "--testset", str(testset)])
    assert "unknown doc_id" in capsys.readouterr().err
