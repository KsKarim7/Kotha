import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket

from kotha.api.session import CallServices, CallSession
from kotha.logs import configure_logging

ServiceLoader = Callable[[ExitStack], CallServices]


def create_app(load_services: ServiceLoader) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with ExitStack() as resources:
            app.state.services = await asyncio.to_thread(load_services, resources)
            yield

    app = FastAPI(title="Kotha", lifespan=lifespan)

    @app.websocket("/v1/calls/stream")
    async def call(websocket: WebSocket) -> None:
        await CallSession(websocket, websocket.app.state.services).run()

    return app


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
    asr = WhisperAsr(asr_model, device=os.environ.get("KOTHA_ASR_DEVICE", "cpu"))
    pipeline = resources.enter_context(live_pipeline(database_url))
    return CallServices(
        new_vad=new_silero_probability, asr=asr, pipeline=pipeline, tts=load_tts_service()
    )


app = create_app(load_live_services)
