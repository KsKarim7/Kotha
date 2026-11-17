import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "kotha-asr"


def quiet_mlflow() -> None:
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")


def open_store(mlflow_dir: Path) -> "MlflowClient":
    quiet_mlflow()
    # MLflow starts telemetry and configures its loggers on import, so the switches come first
    from mlflow.tracking import MlflowClient

    store = mlflow_dir.resolve()
    store.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=f"sqlite:///{store / 'mlflow.db'}")


def experiment_id(client: "MlflowClient", mlflow_dir: Path) -> str:
    artifact_location = (mlflow_dir.resolve() / "artifacts").as_uri()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
    if experiment.artifact_location != artifact_location:
        raise ValueError(
            f"experiment {EXPERIMENT_NAME!r} stores artifacts at {experiment.artifact_location}, "
            f"not {artifact_location}; use the --mlflow-dir it was created with"
        )
    return experiment.experiment_id


def open_experiment(mlflow_dir: Path) -> tuple["MlflowClient", str]:
    quiet_mlflow()
    from mlflow.exceptions import MlflowException
    from sqlalchemy.exc import SQLAlchemyError

    try:
        client = open_store(mlflow_dir)
        return client, experiment_id(client, mlflow_dir)
    except (MlflowException, SQLAlchemyError) as error:
        raise ValueError(f"cannot use the MLflow store in {mlflow_dir}: {error}") from error


def run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def git_tags(repo: Path) -> dict[str, str]:
    dirty = bool(run_git(repo, "status", "--porcelain"))
    return {"git_commit": run_git(repo, "rev-parse", "HEAD"), "git_dirty": str(dirty).lower()}


def log_run(
    client: "MlflowClient",
    experiment: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    artifacts: Sequence[Path],
) -> str:
    run = client.create_run(experiment, tags=dict(tags))
    run_id = run.info.run_id
    for name, value in params.items():
        client.log_param(run_id, name, value)
    for name, value in metrics.items():
        client.log_metric(run_id, name, value)
    for artifact in artifacts:
        client.log_artifact(run_id, str(artifact))
    client.set_terminated(run_id)
    return run_id
