import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from playwright.sync_api import WebSocket, sync_playwright

from kotha.audio.convert import SAMPLE_RATE_HZ, resample
from kotha.dialogue.routing import caller_language
from kotha.tts.mms import MmsVoice
from kotha.tts.speakable import speakable_sentences

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTION = "সিম হারিয়ে গেলে কী করব?"
# the question starts after the greeting has played, as a caller would wait for it
LEAD_SILENCE_S = 12.0
TAIL_SILENCE_S = 3.0
CALL_TIMEOUT_S = 120.0


@dataclass
class CallLog:
    events: list[tuple[float, str, dict]] = field(default_factory=list)
    closed_at: float | None = None

    def watch(self, websocket: WebSocket) -> None:
        websocket.on("framereceived", self.received)
        websocket.on("close", lambda _: setattr(self, "closed_at", time.monotonic()))

    def received(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            self.events.append((time.monotonic(), "audio", {}))
        else:
            event = json.loads(payload)
            self.events.append((time.monotonic(), event["type"], event))

    def kinds(self) -> list[str]:
        return [kind for _, kind, _ in self.events]

    def bot_text(self, kind: str) -> dict | None:
        texts = [event for _, name, event in self.events if name == "bot_text"]
        return next((event for event in texts if event["kind"] == kind), None)

    def audio_after(self, kind: str) -> bool:
        names = self.kinds()
        index = next(
            (i for i, (_, n, e) in enumerate(self.events) if n == "bot_text" and e["kind"] == kind),
            None,
        )
        return index is not None and "audio" in names[index:]


def question_wav(question: str, path: Path) -> None:
    lang = caller_language(question)
    voice = MmsVoice(lang)
    speech = np.concatenate(
        [voice.synthesize(sentence) for sentence in speakable_sentences(question, lang)]
    )
    speech = resample(speech, voice.sampling_rate, SAMPLE_RATE_HZ)
    lead = np.zeros(int(LEAD_SILENCE_S * SAMPLE_RATE_HZ), dtype=np.float32)
    tail = np.zeros(int(TAIL_SILENCE_S * SAMPLE_RATE_HZ), dtype=np.float32)
    sf.write(path, np.concatenate([lead, speech, tail]), SAMPLE_RATE_HZ, subtype="PCM_16")


def run_call(url: str, wav: Path, headless: bool) -> dict[str, object]:
    log = CallLog()
    flags = [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={wav}%noloop",
        "--autoplay-policy=no-user-gesture-required",
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, args=flags)
        page = browser.new_page()
        page.on("websocket", log.watch)
        page.goto(url)
        page.check("#consent")
        page.click("#start")
        deadline = time.monotonic() + CALL_TIMEOUT_S
        while time.monotonic() < deadline and "timing" not in log.kinds():
            page.wait_for_timeout(250)
        page.wait_for_timeout(2000)
        sources = page.locator("#transcript li.bot .sources").count()
        page.click("#stop")
        page.wait_for_timeout(3000)
        status = page.inner_text("#status")
        start_enabled = page.is_enabled("#start")
        browser.close()
    answer = log.bot_text("answer")
    handoffs = [event["reason"] for _, name, event in log.events if name == "handoff"]
    return {
        "greeting_audio": log.audio_after("greeting"),
        "question_transcribed": "transcript" in log.kinds(),
        "cited_answer": answer is not None and len(answer["citations"]) > 0 and sources > 0,
        "answer_audio": log.audio_after("answer"),
        "end_released": log.closed_at is not None and status == "Call ended." and start_enabled,
        "reply_kinds": [event["kind"] for _, name, event in log.events if name == "bot_text"],
        "handoff_reasons": handoffs,
        "cited_documents": [c["doc_id"] for c in answer["citations"]] if answer else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One browser call through the real page.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    work_dir: Path = args.work_dir.resolve()
    if work_dir.is_relative_to(REPO_ROOT):
        parser.error(f"--work-dir {work_dir} must be outside the repository")
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "question.wav"
    question_wav(args.question, wav)
    result = run_call(args.url, wav, headless=not args.headed)
    for name, value in result.items():
        print(f"{name}: {value}")
    checks = (
        "greeting_audio",
        "question_transcribed",
        "cited_answer",
        "answer_audio",
        "end_released",
    )
    sys.exit(0 if all(result[check] for check in checks) else 1)


if __name__ == "__main__":
    main()
