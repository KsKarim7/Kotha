import getpass
import os
import subprocess
from pathlib import Path

import pytest

from tracking import (
    EXPERIMENT_NAME,
    experiment_id,
    git_tags,
    log_run,
    open_experiment,
    open_store,
)
from wer import REPO_ROOT

MLFLOW_SWITCHES = ("MLFLOW_DISABLE_TELEMETRY", "MLFLOW_LOGGING_LEVEL")


@pytest.fixture(autouse=True)
def restore_mlflow_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MLFLOW_SWITCHES:
        monkeypatch.delenv(name, raising=False)


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_open_store_switches_telemetry_off_and_quiets_the_logs(tmp_path: Path) -> None:
    open_store(tmp_path / "store")
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert os.environ["MLFLOW_LOGGING_LEVEL"] == "WARNING"


def test_open_store_forces_telemetry_off_but_keeps_a_chosen_log_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")
    monkeypatch.setenv("MLFLOW_LOGGING_LEVEL", "DEBUG")
    open_store(tmp_path / "store")
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert os.environ["MLFLOW_LOGGING_LEVEL"] == "DEBUG"


def test_open_store_uses_a_sqlite_file_in_the_given_folder(tmp_path: Path) -> None:
    client = open_store(tmp_path / "store")
    assert client.tracking_uri == f"sqlite:///{tmp_path / 'store' / 'mlflow.db'}"


def test_the_experiment_is_created_with_the_artifact_location(tmp_path: Path) -> None:
    store = tmp_path / "store"
    client = open_store(store)
    created = experiment_id(client, store)
    experiment = client.get_experiment(created)
    assert experiment.name == EXPERIMENT_NAME
    assert experiment.artifact_location == (store / "artifacts").as_uri()
    assert experiment_id(client, store) == created


def test_an_experiment_with_another_artifact_location_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "store"
    client = open_store(store)
    client.create_experiment(EXPERIMENT_NAME, artifact_location=(tmp_path / "other").as_uri())
    with pytest.raises(ValueError, match="use the --mlflow-dir it was created with"):
        experiment_id(client, store)


def test_open_experiment_passes_the_location_refusal_through(tmp_path: Path) -> None:
    store = tmp_path / "store"
    open_store(store).create_experiment(
        EXPERIMENT_NAME, artifact_location=(tmp_path / "other").as_uri()
    )
    with pytest.raises(ValueError, match="use the --mlflow-dir it was created with"):
        open_experiment(store)


def test_a_store_that_is_not_a_database_becomes_a_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "mlflow.db").write_bytes(b"not a sqlite database" * 64)
    monkeypatch.setattr("mlflow.store.db.utils.MAX_RETRY_COUNT", 1)
    with pytest.raises(ValueError, match="cannot use the MLflow store in .*file is not a database"):
        open_experiment(store)


def test_log_run_stores_params_metrics_tags_and_both_artifacts(tmp_path: Path) -> None:
    store = tmp_path / "store"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    utterances = outputs / "dev.csv"
    substitutions = outputs / "dev.substitutions.csv"
    utterances.write_text("id\n", encoding="utf-8")
    substitutions.write_text("condition\n", encoding="utf-8")
    params = {"model": "small", "split": "dev"}
    metrics = {"16k.all.wer": 0.25, "16k.mix.latency_p95_s": 1.5}
    tags = {"git_commit": "0" * 40, "git_dirty": "false"}

    client, experiment = open_experiment(store)
    run_id = log_run(client, experiment, params, metrics, tags, [utterances, substitutions])

    run = open_store(store).get_run(run_id)
    assert run.data.params == params
    assert run.data.metrics == metrics
    assert {name: run.data.tags[name] for name in tags} == tags
    assert sorted(item.path for item in client.list_artifacts(run_id)) == [
        "dev.csv",
        "dev.substitutions.csv",
    ]
    assert run.info.status == "FINISHED"
    private = [str(tmp_path), str(Path.home()), getpass.getuser()]
    logged = [*run.data.params.values(), *run.data.tags.values(), run.info.run_name]
    assert not [value for value in logged for text in private if text in value]


def test_git_tags_read_the_commit_and_whether_the_tree_is_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "first")
    commit = git(repo, "rev-parse", "HEAD")

    assert git_tags(repo) == {"git_commit": commit, "git_dirty": "false"}

    (repo / "untracked.txt").write_text("b\n", encoding="utf-8")
    assert git_tags(repo) == {"git_commit": commit, "git_dirty": "true"}


def test_git_tags_of_this_repository_name_its_head() -> None:
    tags = git_tags(REPO_ROOT)
    assert tags["git_commit"] == git(REPO_ROOT, "rev-parse", "HEAD")
    assert tags["git_dirty"] in {"true", "false"}


def test_git_tags_outside_a_repository_raise(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        git_tags(tmp_path)
