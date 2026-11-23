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
# the fake microphone starts with the call, so the question waits out the greeting (about 9 s)
LEAD_SILENCE_S = 15.0
TAIL_SILENCE_S = 3.0
CALL_TIMEOUT_S = 150.0
CHECKS = ("greeting_audio", "question_transcribed", "cited_answer", "answer_audio", "end_released")


@dataclass
class CallLog:
    events: list[tuple[str, dict]] = field(default_factory=list)
    closed: bool = False

    def watch(self, websocket: WebSocket) -> None:
        websocket.on("framereceived", self.received)
        websocket.on("close", self.socket_closed)

    def received(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            self.events.append(("audio", {}))
        else:
            event = json.loads(payload)
            self.events.append((event["type"], event))

    def socket_closed(self, websocket: WebSocket) -> None:
        self.closed = True

    def kinds(self) -> list[str]:
        return [name for name, _ in self.events]

    def reply(self, kind: str) -> dict | None:
        return next(
            (event for name, event in self.events if name == "bot_text" and event["kind"] == kind),
            None,
        )

    def reply_had_audio(self, kind: str) -> bool:
        speaking = False
        for name, event in self.events:
            if name == "bot_text":
                speaking = event["kind"] == kind
            elif name == "reply_audio_end":
                speaking = False
            elif name == "audio" and speaking:
                return True
        return False


def question_wav(question: str, path: Path) -> None:
    lang = caller_language(question)
    voice = MmsVoice(lang)
    sentences = speakable_sentences(question, lang)
    speech = np.concatenate([voice.synthesize(sentence) for sentence in sentences])
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
        while time.monotonic() < deadline and "timing" not in log.kinds() and not log.closed:
            page.wait_for_timeout(250)
        timed_out = "timing" not in log.kinds() and not log.closed
        page.wait_for_timeout(2000)
        sources = page.locator("#transcript li.bot .sources").count()
        ended_by_caller = page.is_enabled("#stop")
        if ended_by_caller:
            page.click("#stop")
            page.wait_for_timeout(3000)
        page_reset = page.inner_text("#status") == "Call ended." and page.is_enabled("#start")
        browser.close()
    answer = log.reply("answer")
    return {
        "greeting_audio": log.reply_had_audio("greeting"),
        "question_transcribed": "transcript" in log.kinds(),
        "cited_answer": answer is not None and len(answer["citations"]) > 0 and sources > 0,
        "answer_audio": log.reply_had_audio("answer"),
        "end_released": ended_by_caller and log.closed and page_reset,
        "timed_out": timed_out,
        "reply_kinds": [event["kind"] for name, event in log.events if name == "bot_text"],
        "handoff_reasons": [event["reason"] for name, event in log.events if name == "handoff"],
        "cited_documents": [citation["doc_id"] for citation in answer["citations"]]
        if answer
        else [],
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
    sys.exit(0 if all(result[check] for check in CHECKS) else 1)


if __name__ == "__main__":
    main()
