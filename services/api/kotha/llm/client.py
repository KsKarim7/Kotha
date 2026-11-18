from typing import Protocol


class LLMError(Exception):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMEmptyResponseError(LLMError):
    pass


class LLMTransientError(LLMError):
    pass


class LLMTimeoutError(LLMTransientError):
    pass


class LLMRateLimitError(LLMTransientError):
    pass


class LLMUnavailableError(LLMTransientError):
    pass


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
