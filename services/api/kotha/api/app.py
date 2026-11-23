import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kotha.api.protocol import Readiness
from kotha.api.session import CallServices, CallSession, Transcriber
from kotha.dialogue.routing import Language
from kotha.logs import configure_logging, log_event
from kotha.tts.service import SentenceAudio
from kotha.vad import WINDOW_SAMPLES

ServiceLoader = Callable[[ExitStack], CallServices]
STATIC_DIR = Path(__file__).resolve().parent / "static"
WARM_UP_SENTENCE = "প্যাকের মেয়াদ ত্রিশ দিন।"


class WarmablePipeline(Protocol):
    def warm_up(self) -> None: ...


class SentenceSynthesizer(Protocol):
    def synthesize(self, text: str, lang: Language) -> SentenceAudio: ...


def create_app(load_services: ServiceLoader) -> FastAPI:
    # uvicorn accepts no connection until the lifespan yields, so a call never meets a cold model
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with ExitStack() as resources:
            started = time.perf_counter()
            app.state.services = await asyncio.to_thread(load_services, resources)
            app.state.startup_seconds = round(time.perf_counter() - started, 3)
            log_event("ready", seconds=app.state.startup_seconds)
            yield

    app = FastAPI(title="Kotha", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/v1/ready")
    async def ready(request: Request) -> Readiness:
        return Readiness(startup_seconds=request.app.state.startup_seconds)

    @app.websocket("/v1/calls/stream")
    async def call(websocket: WebSocket) -> None:
        await CallSession(websocket, websocket.app.state.services).run()

    return app


def warm_up(
    new_vad: Callable[[], Callable[[np.ndarray], float]],
    asr: Transcriber,
    pipeline: WarmablePipeline,
    tts: SentenceSynthesizer,
) -> None:
    new_vad()(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    asr.transcribe(np.concatenate(tts.synthesize(WARM_UP_SENTENCE, "bn")))
    pipeline.warm_up()


def new_silero_probability() -> Callable[[np.ndarray], float]:
    from kotha.vad.silero import SileroVad

    return SileroVad().probability


def load_live_services(resources: ExitStack) -> CallServices:
    from kotha.asr.whisper import WhisperAsr
    from kotha.dialogue.chat import live_pipeline
    from kotha.tts.service import load_tts_service

    configure_logging()
    database_url = os.environ.get("DATABASE_URL")
    asr_model = os.environ.get("KOTHA_ASR_MODEL")
    if not database_url or not asr_model:
        raise RuntimeError("DATABASE_URL and KOTHA_ASR_MODEL must be set (see docs/runbook.md)")
    asr = WhisperAsr(
        asr_model,
        device=os.environ.get("KOTHA_ASR_DEVICE", "cpu"),
        compute_type=os.environ.get("KOTHA_ASR_COMPUTE_TYPE", "int8"),
    )
    pipeline = resources.enter_context(live_pipeline(database_url))
    tts = load_tts_service()
    warm_up(new_silero_probability, asr, pipeline, tts)
    return CallServices(new_vad=new_silero_probability, asr=asr, pipeline=pipeline, tts=tts)


app = create_app(load_live_services)
