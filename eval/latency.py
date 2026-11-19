import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path

import numpy as np
from websockets.asyncio.client import ClientConnection, connect

from kotha.api.protocol import Readiness
from kotha.audio.convert import SAMPLE_RATE_HZ, float_to_pcm16
from kotha.dialogue.routing import caller_language
from kotha.kb.documents import read_documents
from kotha.kb.ingest import KB_FOLDER
from retrieval import TESTSET_PATH, Question, read_testset
from tracking import git_tags, log_run, open_experiment
from wer import latency_percentiles

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "kotha-latency"
FRAME_SAMPLES = SAMPLE_RATE_HZ // 50
LEAD_SILENCE_SECONDS = 0.3
# the end of speech is the last sample above 1% of the clip's peak, not the end of the TTS output
AUDIBLE_FRACTION = 0.01
TURN_TIMEOUT_SECONDS = 120.0
READY_TIMEOUT_SECONDS = 600.0
SERVER_STAGES = ("asr", "retrieval", "llm", "tts_first_audio")
STAGES = ("vad", *SERVER_STAGES, "ttfa", "ttfa_from_vad")
RESULT_COLUMNS = ("question_id", "lang", "kind", "split", *STAGES)


@dataclass
class TurnResult:
    question_id: str
    lang: str
    kind: str = ""
    split: bool = False
    stages: dict[str, float] = field(default_factory=dict)


def without_trailing_silence(clip: np.ndarray) -> np.ndarray:
    audible = np.flatnonzero(np.abs(clip) >= AUDIBLE_FRACTION * np.abs(clip).max())
    return clip[: audible[-1] + 1] if len(audible) else clip[:0]


def speech_frames(clip: np.ndarray) -> list[bytes]:
    lead = np.zeros(int(LEAD_SILENCE_SECONDS * SAMPLE_RATE_HZ), dtype=np.float32)
    pcm = float_to_pcm16(np.concatenate([lead, without_trailing_silence(clip)]))
    step = FRAME_SAMPLES * 2
    return [pcm[start : start + step] for start in range(0, len(pcm), step)]


def turn_stages(
    speech_end: float, vad_end: float, first_audio: float, timing: Mapping[str, float | None]
) -> dict[str, float]:
    stages = {
        "vad": vad_end - speech_end,
        "ttfa": first_audio - speech_end,
        "ttfa_from_vad": first_audio - vad_end,
    }
    for stage in SERVER_STAGES:
        seconds = timing.get(f"{stage}_seconds")
        if seconds is not None:
            stages[stage] = seconds
    return stages


def summarise(results: Sequence[TurnResult]) -> dict[str, tuple[int, float, float]]:
    measured = [result for result in results if not result.split]
    summary = {}
    for stage in STAGES:
        values = [result.stages[stage] for result in measured if stage in result.stages]
        if values:
            summary[stage] = (len(values), *latency_percentiles(values))
    return summary


async def send_paced(
    websocket: ClientConnection, frames: Iterable[bytes], clock: Iterator[float]
) -> None:
    for frame, due in zip(frames, clock, strict=False):
        await asyncio.sleep(max(0.0, due - time.perf_counter()))
        await websocket.send(frame)


def frame_clock(start: float) -> Iterator[float]:
    index = 0
    while True:
        yield start + index * FRAME_SAMPLES / SAMPLE_RATE_HZ
        index += 1


async def measure_turn(url: str, question: Question, clip: np.ndarray) -> TurnResult:
    result = TurnResult(question.id, question.lang)
    async with connect(url, max_size=None) as websocket:
        await websocket.send(json.dumps({"type": "start", "sample_rate": 16000, "consent": True}))
        async for greeting in websocket:
            if isinstance(greeting, str) and json.loads(greeting)["type"] == "reply_audio_end":
                break
        speech = speech_frames(clip)
        silence = float_to_pcm16(np.zeros(FRAME_SAMPLES, dtype=np.float32))
        clock = frame_clock(time.perf_counter())
        await send_paced(websocket, speech, clock)
        speech_end = time.perf_counter()
        silence_task = asyncio.create_task(send_paced(websocket, repeat(silence), clock))
        vad_end = first_audio = None
        timing: dict[str, float | None] = {}
        try:
            async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
                while not timing:
                    message = await websocket.recv()
                    now = time.perf_counter()
                    if isinstance(message, bytes):
                        first_audio = first_audio or now
                        continue
                    event = json.loads(message)
                    if event["type"] == "vad" and event["event"] == "speech_end":
                        vad_end = vad_end or now
                    elif event["type"] == "bot_text":
                        result.kind = event["kind"]
                    elif event["type"] == "timing":
                        timing = event
        finally:
            silence_task.cancel()
        await websocket.send(json.dumps({"type": "stop"}))
    result.split = vad_end is None or vad_end < speech_end
    if vad_end is not None and first_audio is not None:
        result.stages = turn_stages(speech_end, vad_end, first_audio, timing)
    return result


def wait_until_ready(base_url: str, deadline_seconds: float) -> Readiness:
    give_up_at = time.perf_counter() + deadline_seconds
    while time.perf_counter() < give_up_at:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/ready", timeout=2) as response:
                return Readiness.model_validate_json(response.read())
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError(f"{base_url} did not become ready within {deadline_seconds} s")


def launch_server(port: int, log_path: Path) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-m", "uvicorn", "kotha.api.app:app", "--host", "127.0.0.1"]
    command += ["--port", str(port), "--ws-max-size", "65536"]
    with log_path.open("wb") as log:
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def synthesise_clips(questions: Sequence[Question]) -> list[np.ndarray]:
    from kotha.tts.service import load_tts_service

    tts = load_tts_service()
    return [
        np.concatenate(tts.synthesize(question.text, caller_language(question.text)))
        for question in questions
    ]


def write_results(path: Path, results: Sequence[TurnResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(RESULT_COLUMNS)
        for result in results:
            stages = [
                f"{result.stages[stage]:.4f}" if stage in result.stages else "" for stage in STAGES
            ]
            writer.writerow([result.question_id, result.lang, result.kind, result.split, *stages])


def print_summary(summary: Mapping[str, tuple[int, float, float]]) -> None:
    print(f"{'stage':<16}{'n':>4}{'p50 s':>9}{'p95 s':>9}")
    for stage, (count, p50, p95) in summary.items():
        print(f"{stage:<16}{count:>4}{p50:>9.3f}{p95:>9.3f}")


async def measure_all(
    url: str, questions: Sequence[Question], clips: Sequence[np.ndarray]
) -> list[TurnResult]:
    results = []
    for question, clip in zip(questions, clips, strict=True):
        result = await measure_turn(url, question, clip)
        print(f"{question.id}: {result.kind or 'no reply'}{' (split)' if result.split else ''}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure per-stage latency of warm calls.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--launch", action="store_true", help="start the app and time cold start")
    parser.add_argument("--limit", type=int, help="use only the first N questions")
    parser.add_argument("--mlflow-dir", type=Path)
    args = parser.parse_args()
    output: Path = args.output.resolve()
    if output.is_relative_to(REPO_ROOT):
        parser.error(f"--output {output} must be outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    tags = git_tags(REPO_ROOT)
    known_doc_ids = {document.doc_id for document in read_documents(KB_FOLDER)}
    questions = read_testset(TESTSET_PATH, known_doc_ids)[: args.limit]
    clips = synthesise_clips(questions)

    base_url = f"http://127.0.0.1:{args.port}"
    server = None
    cold_start_seconds = None
    launched = time.perf_counter()
    if args.launch:
        server = launch_server(args.port, output.with_suffix(".server.log"))
    try:
        readiness = wait_until_ready(base_url, READY_TIMEOUT_SECONDS)
        if server is not None:
            cold_start_seconds = time.perf_counter() - launched
        url = f"ws://127.0.0.1:{args.port}/v1/calls/stream"
        results = asyncio.run(measure_all(url, questions, clips))
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=60)

    summary = summarise(results)
    split = sum(result.split for result in results)
    kinds = Counter(result.kind or "no_reply" for result in results)
    print(f"turns: {len(results)}, split by VAD (excluded): {split}, replies: {dict(kinds)}")
    if cold_start_seconds is not None:
        print(f"cold start (process start to ready): {cold_start_seconds:.1f} s")
    print_summary(summary)
    write_results(output, results)
    if args.mlflow_dir:
        client, experiment = open_experiment(args.mlflow_dir, EXPERIMENT_NAME)
        metrics = {
            f"{stage}.{name}": value
            for stage, (_, p50, p95) in summary.items()
            for name, value in (("p50", p50), ("p95", p95))
        }
        metrics["turns"] = len(results)
        metrics["split_turns"] = split
        metrics.update({f"replies.{kind}": count for kind, count in kinds.items()})
        metrics["startup_seconds"] = readiness.startup_seconds
        if cold_start_seconds is not None:
            metrics["cold_start_seconds"] = cold_start_seconds
        params = {
            "testset_sha256": hashlib.sha256(TESTSET_PATH.read_bytes()).hexdigest(),
            "questions": str(len(questions)),
            "asr_model": os.environ.get("KOTHA_ASR_MODEL", ""),
            "asr_device": os.environ.get("KOTHA_ASR_DEVICE", "cpu"),
            "input": "synthetic: rag_test questions spoken by the MMS voices",
            "frame_ms": str(1000 * FRAME_SAMPLES // SAMPLE_RATE_HZ),
        }
        run_id = log_run(client, experiment, params, metrics, tags, [output])
        print(f"mlflow run: {run_id}")


if __name__ == "__main__":
    main()
