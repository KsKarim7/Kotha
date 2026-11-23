import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from kotha.dialogue import chat


class ClosingLlm:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_the_live_pipeline_times_the_llm_client_embedder_and_index_and_closes_the_llm(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    llm = ClosingLlm()

    @contextmanager
    def fake_connect(database_url: str) -> Iterator[str]:
        yield f"connection to {database_url}"

    monkeypatch.setattr(chat, "GeminiClient", lambda: llm)
    monkeypatch.setattr(chat, "connect", fake_connect)
    monkeypatch.setattr(chat, "Embedder", lambda: "embedder")
    monkeypatch.setattr(chat, "HybridRetriever", lambda connection, embedder, lexicon: embedder)
    caplog.set_level(logging.INFO, logger="kotha")
    with chat.live_pipeline("postgresql://test") as pipeline:
        assert isinstance(pipeline, chat.AnswerPipeline)
    steps = [message.rsplit("stage=", 1)[1] for message in caplog.messages]
    assert steps == ["llm_client", "embedder_load", "retriever_index"]
    assert llm.closed
