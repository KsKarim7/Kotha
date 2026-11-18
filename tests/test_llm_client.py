import json
import socket
import threading
import time
from collections.abc import Callable

import httpx2
import pytest

from kotha.llm import gemini
from kotha.llm.client import (
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from kotha.llm.fake import FakeLLMCall, FakeLLMClient
from kotha.llm.gemini import GeminiClient

SECRET_KEY = "test-key-value"
PROVIDER_SECRET = "caller said a secret"
Handler = Callable[[httpx2.Request], httpx2.Response]


def answer(text: str, finish_reason: str = "STOP") -> httpx2.Response:
    body = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish_reason}
        ]
    }
    return httpx2.Response(200, json=body)


def status(code: int) -> httpx2.Response:
    error = {"error": {"code": code, "message": PROVIDER_SECRET, "status": "X"}}
    return httpx2.Response(code, json=error)


class Server:
    def __init__(self, outcomes: list[httpx2.Response | Exception | Handler]) -> None:
        self.outcomes = outcomes
        self.requests: list[httpx2.Request] = []
        self.lock = threading.Lock()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        with self.lock:
            self.requests.append(request)
            outcome = self.outcomes.pop(0) if self.outcomes else answer("Fine.")
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome) and not isinstance(outcome, httpx2.Response):
            return outcome(request)
        return outcome

    def models(self) -> list[str]:
        return [request.url.path.rsplit("/", 1)[-1].split(":")[0] for request in self.requests]

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_KEY)


def client_for(server: Server) -> GeminiClient:
    return GeminiClient(transport=httpx2.MockTransport(server))


def test_the_settings_are_the_ones_the_adr_records() -> None:
    assert (gemini.PRIMARY_MODEL, gemini.RETRY_MODEL) == (
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
    )
    assert gemini.TOTAL_DEADLINE_S == 12.0
    assert gemini.PRIMARY_ATTEMPT_S == 4.0
    assert gemini.MAX_OUTPUT_TOKENS == 512


def test_a_request_sends_the_key_prompts_and_settings_and_returns_the_answer() -> None:
    server = Server([answer("  An answer. \n")])
    assert client_for(server).generate("system", "question") == "An answer."
    [request] = server.requests
    body = server.body()
    assert request.headers["x-goog-api-key"] == SECRET_KEY
    assert request.headers["x-server-timeout"] == "12"
    assert set(request.extensions["timeout"].values()) == {12.0}
    assert server.models() == ["gemini-3.1-flash-lite"]
    assert body["contents"][0]["parts"][0]["text"] == "question"
    assert body["systemInstruction"]["parts"][0]["text"] == "system"
    assert body["generationConfig"]["temperature"] == 0.2
    assert body["generationConfig"]["maxOutputTokens"] == 512
    assert body["generationConfig"]["thinkingConfig"]["thinking_level"] == "MINIMAL"


@pytest.mark.parametrize("value", [None, ""])
def test_a_missing_key_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    if value is not None:
        monkeypatch.setenv("GEMINI_API_KEY", value)
    with pytest.raises(LLMConfigurationError, match="is not set"):
        GeminiClient()


@pytest.mark.parametrize("value", ["kéy", "key\nvalue", "key value"])
def test_a_malformed_key_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", value)
    with pytest.raises(LLMConfigurationError, match="invalid characters") as caught:
        GeminiClient()
    assert value not in str(caught.value)


@pytest.mark.parametrize(
    "failure",
    [
        status(429),
        status(408),
        status(500),
        status(503),
        status(504),
        httpx2.ReadTimeout("slow"),
        httpx2.ConnectError("refused"),
        httpx2.ReadError("reset"),
        httpx2.RemoteProtocolError("broken"),
        httpx2.Response(200, text="<html>proxy login</html>"),
        httpx2.Response(503, content=b"\xff\xfe not utf-8"),
    ],
)
def test_a_transient_failure_is_retried_once_on_the_second_model(failure: object) -> None:
    server = Server([failure, answer("Recovered.")])
    assert client_for(server).generate("system", "question") == "Recovered."
    assert server.models() == ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (status(429), LLMRateLimitError),
        (status(408), LLMTimeoutError),
        (status(503), LLMUnavailableError),
        (httpx2.ReadTimeout("slow"), LLMTimeoutError),
        (httpx2.ConnectError("refused"), LLMUnavailableError),
        (httpx2.Response(200, text="not json"), LLMUnavailableError),
        (httpx2.Response(200, json={"candidates": "wrong"}), LLMEmptyResponseError),
    ],
)
def test_a_second_failure_raises_a_typed_error(failure: object, expected: type[LLMError]) -> None:
    server = Server([status(503), failure])
    with pytest.raises(expected):
        client_for(server).generate("system", "question")
    assert len(server.requests) == 2


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_a_rejected_request_is_not_retried(code: int) -> None:
    server = Server([status(code)])
    with pytest.raises(LLMRequestError, match=str(code)):
        client_for(server).generate("system", "question")
    assert len(server.requests) == 1


def test_a_redirect_is_not_followed() -> None:
    redirect = httpx2.Response(307, headers={"location": "https://elsewhere.example/x"})
    server = Server([redirect, redirect])
    with pytest.raises(LLMError):
        client_for(server).generate("system", "question")
    assert all(request.url.host != "elsewhere.example" for request in server.requests)


@pytest.mark.parametrize(
    "response",
    [
        answer("   "),
        answer("A cut-off ans", "MAX_TOKENS"),
        answer("Blocked text", "SAFETY"),
        httpx2.Response(200, json={"candidates": []}),
    ],
)
def test_an_empty_or_incomplete_answer_is_an_error_and_is_not_retried(
    response: httpx2.Response,
) -> None:
    server = Server([response])
    with pytest.raises(LLMEmptyResponseError):
        client_for(server).generate("system", "question")
    assert len(server.requests) == 1


def test_errors_carry_no_provider_text_key_or_prompt() -> None:
    server = Server([status(400)])
    with pytest.raises(LLMRequestError) as caught:
        client_for(server).generate("system", "the private question")
    error = caught.value
    assert error.__cause__ is None and error.__context__ is None
    for secret in (PROVIDER_SECRET, SECRET_KEY, "the private question"):
        assert secret not in str(error)


def test_a_slow_attempt_is_abandoned_at_its_budget_and_the_retry_gets_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gemini, "PRIMARY_ATTEMPT_S", 0.2)
    monkeypatch.setattr(gemini, "TOTAL_DEADLINE_S", 3.0)
    monkeypatch.setattr(gemini, "MIN_RETRY_S", 0.5)

    def stall(request: httpx2.Request) -> httpx2.Response:
        time.sleep(1.0)
        return answer("Too late.")

    server = Server([stall, answer("In time.")])
    started = time.monotonic()
    assert client_for(server).generate("system", "question") == "In time."
    assert time.monotonic() - started < 1.0


def test_no_retry_starts_when_too_little_time_is_left(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini, "PRIMARY_ATTEMPT_S", 0.3)
    monkeypatch.setattr(gemini, "TOTAL_DEADLINE_S", 0.6)
    monkeypatch.setattr(gemini, "MIN_RETRY_S", 0.5)

    def stall(request: httpx2.Request) -> httpx2.Response:
        time.sleep(1.0)
        return answer("Too late.")

    server = Server([stall])
    started = time.monotonic()
    with pytest.raises(LLMTimeoutError):
        client_for(server).generate("system", "question")
    assert time.monotonic() - started < 0.9
    assert len(server.requests) == 1


def test_concurrent_calls_each_get_their_own_answer() -> None:
    def echo(request: httpx2.Request) -> httpx2.Response:
        return answer(json.loads(request.content)["contents"][0]["parts"][0]["text"])

    server = Server([echo] * 40)
    client = client_for(server)
    results: dict[int, str] = {}

    def ask(number: int) -> None:
        results[number] = client.generate("system", f"q{number}")

    threads = [threading.Thread(target=ask, args=(number,)) for number in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {number: f"q{number}" for number in range(40)}


def test_a_closed_client_raises_a_typed_error() -> None:
    client = client_for(Server([]))
    client.close()
    with pytest.raises(LLMUnavailableError, match="closed"):
        client.generate("system", "question")


def test_a_call_still_queued_when_the_client_closes_raises_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gemini, "WORKER_THREADS", 1)
    release = threading.Event()

    def hold(request: httpx2.Request) -> httpx2.Response:
        release.wait(2.0)
        return answer("First.")

    client = client_for(Server([hold]))
    outcomes: list[object] = []

    def ask() -> None:
        try:
            outcomes.append(client.generate("system", "question"))
        except LLMError as error:
            outcomes.append(type(error))

    first = threading.Thread(target=ask)
    first.start()
    time.sleep(0.1)
    queued = threading.Thread(target=ask)
    queued.start()
    time.sleep(0.1)
    client.close()
    release.set()
    first.join()
    queued.join()
    assert sorted(map(str, outcomes)) == sorted(map(str, ["First.", LLMUnavailableError]))


def test_a_real_stalled_socket_frees_its_worker_at_the_backstop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gemini, "BACKSTOP_TIMEOUT_MS", 300)
    monkeypatch.setattr(gemini, "WORKER_THREADS", 1)
    monkeypatch.setattr(gemini, "PRIMARY_ATTEMPT_S", 0.1)
    monkeypatch.setattr(gemini, "TOTAL_DEADLINE_S", 0.15)
    with socket.create_server(("127.0.0.1", 0)) as silent_server:
        host, port = silent_server.getsockname()
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://{host}:{port}")
        client = GeminiClient()
        with pytest.raises(LLMTimeoutError):
            client.generate("system", "question")
        next_job = client._workers.submit(time.monotonic)
        assert next_job.result(timeout=2.0)
        client.close()


def test_the_fake_client_replays_replies_and_records_calls() -> None:
    client = FakeLLMClient(["first", LLMTimeoutError("slow")])
    assert client.generate("system", "one") == "first"
    with pytest.raises(LLMTimeoutError):
        client.generate("system", "two")
    assert client.calls == [FakeLLMCall("system", "one"), FakeLLMCall("system", "two")]
