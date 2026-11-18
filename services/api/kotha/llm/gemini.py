import json
import os
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import httpx2
from google import genai
from google.genai import errors, types

from kotha.llm.client import (
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMTransientError,
    LLMUnavailableError,
)

API_KEY_VARIABLE = "GEMINI_API_KEY"
PRIMARY_MODEL = "gemini-3.1-flash-lite"
RETRY_MODEL = "gemini-3.5-flash-lite"
TOTAL_DEADLINE_S = 12.0
PRIMARY_ATTEMPT_S = 4.0
MIN_RETRY_S = 2.0
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 512
THINKING_LEVEL = types.ThinkingLevel.MINIMAL
# frees an abandoned worker; Gemini refuses a server deadline under 10 s
BACKSTOP_TIMEOUT_MS = 12_000
WORKER_THREADS = 16
RATE_LIMITED = 429
REQUEST_TIMEOUT = 408


def error_for_status(status_code: int | None) -> LLMError:
    message = f"Gemini returned status {status_code}"
    if status_code == RATE_LIMITED:
        return LLMRateLimitError(message)
    if status_code == REQUEST_TIMEOUT:
        return LLMTimeoutError(message)
    if status_code is not None and 400 <= status_code < 500:
        return LLMRequestError(message)
    return LLMUnavailableError(message)


def response_text(response: types.GenerateContentResponse) -> str:
    candidates = response.candidates or []
    finished = bool(candidates) and candidates[0].finish_reason == types.FinishReason.STOP
    text = (response.text or "").strip() if finished else ""
    if not text:
        raise LLMEmptyResponseError("Gemini returned no complete answer")
    return text


def usable_key(api_key: str) -> bool:
    return api_key.isascii() and api_key.isprintable() and not any(c.isspace() for c in api_key)


class GeminiClient:
    def __init__(self, transport: httpx2.BaseTransport | None = None) -> None:
        api_key = os.environ.get(API_KEY_VARIABLE, "")
        if not api_key:
            raise LLMConfigurationError(f"{API_KEY_VARIABLE} is not set")
        if not usable_key(api_key):
            raise LLMConfigurationError(f"{API_KEY_VARIABLE} contains invalid characters")
        # a redirect would send the key header and the prompt to another host
        self._http = httpx2.Client(transport=transport, follow_redirects=False)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(httpx_client=self._http, timeout=BACKSTOP_TIMEOUT_MS),
        )
        self._workers = ThreadPoolExecutor(WORKER_THREADS, thread_name_prefix="gemini")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        deadline = time.monotonic() + TOTAL_DEADLINE_S
        try:
            return self._attempt(PRIMARY_MODEL, system_prompt, user_prompt, PRIMARY_ATTEMPT_S)
        except LLMTransientError:
            remaining = deadline - time.monotonic()
            if remaining < MIN_RETRY_S:
                raise
            return self._attempt(RETRY_MODEL, system_prompt, user_prompt, remaining)

    def close(self) -> None:
        self._workers.shutdown(wait=False, cancel_futures=True)
        self._http.close()

    def _attempt(self, model: str, system_prompt: str, user_prompt: str, budget_s: float) -> str:
        try:
            future = self._workers.submit(self._request, model, system_prompt, user_prompt)
        except RuntimeError:
            raise LLMUnavailableError("the Gemini client is closed") from None
        try:
            return future.result(timeout=budget_s)
        except FutureTimeoutError:
            future.cancel()
        except CancelledError:
            raise LLMUnavailableError("the Gemini client is closed") from None
        raise LLMTimeoutError(f"Gemini did not answer within {budget_s:.1f} s")

    def _request(self, model: str, system_prompt: str, user_prompt: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        )
        # the mapped error is raised after the handler so it carries no SDK error, whose
        # request holds the key header and the prompt
        try:
            response = self._client.models.generate_content(
                model=model, contents=user_prompt, config=config
            )
            return response_text(response)
        except errors.APIError as error:
            failure = error_for_status(error.code)
        except httpx2.TimeoutException:
            failure = LLMTimeoutError("Gemini did not answer in time")
        except httpx2.RequestError:
            failure = LLMUnavailableError("Gemini could not be reached")
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            errors.UnknownApiResponseError,
            AttributeError,
        ):
            failure = LLMUnavailableError("Gemini sent a response that could not be read")
        raise failure
