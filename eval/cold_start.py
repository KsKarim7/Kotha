import argparse
import os
import re
import time
from pathlib import Path

from latency import READY_TIMEOUT_SECONDS, launch_server, port_in_use, stop_server, wait_until_ready
from tracking import git_tags, log_run, open_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "kotha-cold-start"
STEP_LINE = re.compile(r"event=startup_step seconds=(\d+(?:\.\d+)?) stage=([A-Za-z][\w.-]*)")
EXPECTED_STEPS = (
    "asr_load",
    "llm_client",
    "embedder_load",
    "retriever_index",
    "tts_load_bn",
    "tts_load_en",
    "tts_prompts_bn",
    "tts_prompts_en",
    "vad_warm_up",
    "asr_warm_up_speech",
    "asr_warm_up",
    "retrieval_warm_up",
)


def startup_steps(server_log: str) -> dict[str, float]:
    steps = {stage: float(seconds) for seconds, stage in STEP_LINE.findall(server_log)}
    missing = [step for step in EXPECTED_STEPS if step not in steps]
    if missing:
        raise ValueError(f"the server log has no time for: {', '.join(missing)}")
    return steps


def breakdown(
    steps: dict[str, float], startup_seconds: float, cold_start_seconds: float
) -> dict[str, float]:
    return {
        **steps,
        "other_startup": startup_seconds - sum(steps.values()),
        "before_startup": cold_start_seconds - startup_seconds,
        "startup_seconds": startup_seconds,
        "cold_start_seconds": cold_start_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Time one cold start of the app, step by step.")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--mlflow-dir", type=Path)
    args = parser.parse_args()
    server_log: Path = args.server_log.resolve()
    if server_log.is_relative_to(REPO_ROOT):
        parser.error(f"--server-log {server_log} must be outside the repository")
    if port_in_use(args.port):
        parser.error(f"port {args.port} is already in use")
    server_log.parent.mkdir(parents=True, exist_ok=True)
    tags = git_tags(REPO_ROOT)

    launched = time.perf_counter()
    server = launch_server(args.port, server_log)
    try:
        url = f"http://127.0.0.1:{args.port}"
        readiness = wait_until_ready(url, READY_TIMEOUT_SECONDS, server)
        cold_start_seconds = time.perf_counter() - launched
    finally:
        stop_server(server)

    steps = startup_steps(server_log.read_text(encoding="utf-8"))
    parts = breakdown(steps, readiness.startup_seconds, cold_start_seconds)
    for name, seconds in parts.items():
        print(f"{name:<22}{seconds:>9.3f}")
    if args.mlflow_dir:
        client, experiment = open_experiment(args.mlflow_dir, EXPERIMENT_NAME)
        params = {
            "asr_model": os.environ.get("KOTHA_ASR_MODEL", "unset"),
            "asr_device": os.environ.get("KOTHA_ASR_DEVICE", "cpu"),
            "asr_compute_type": os.environ.get("KOTHA_ASR_COMPUTE_TYPE", "int8"),
        }
        run_id = log_run(client, experiment, params, parts, tags, [server_log])
        print(f"mlflow run: {run_id}")


if __name__ == "__main__":
    main()
