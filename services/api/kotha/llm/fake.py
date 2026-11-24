from collections.abc import Iterable
from dataclasses import dataclass

from kotha.llm.client import Generation

FAKE_MODEL = "fake-model"


@dataclass(frozen=True)
class FakeLLMCall:
    system_prompt: str
    user_prompt: str


class FakeLLMClient:
    def __init__(self, replies: Iterable[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[FakeLLMCall] = []

    def generate(self, system_prompt: str, user_prompt: str) -> Generation:
        self.calls.append(FakeLLMCall(system_prompt, user_prompt))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Generation(reply, FAKE_MODEL)
