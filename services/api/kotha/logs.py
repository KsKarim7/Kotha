import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager

LOGGER = logging.getLogger("kotha")
ALLOWED_FIELDS = frozenset(
    {"call_id", "error", "frames", "kind", "lang", "reason", "seconds", "sentences", "stage"}
)
# a token starts with a letter, so a phone number, PIN or NID number cannot pass as one
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")

LogValue = str | int | float | bool


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def log_event(event: str, **fields: LogValue) -> None:
    if not TOKEN.fullmatch(event):
        raise ValueError(f"log event {event!r} is not a short token")
    for name, value in fields.items():
        if name not in ALLOWED_FIELDS:
            raise ValueError(f"log field {name!r} is not on the allow-list")
        if isinstance(value, str) and not TOKEN.fullmatch(value):
            raise ValueError(f"log field {name!r} must be a short token")
    parts = [f"event={event}", *(f"{name}={value}" for name, value in sorted(fields.items()))]
    LOGGER.info(" ".join(parts))


@contextmanager
def logged_duration(step: str) -> Iterator[None]:
    started = time.perf_counter()
    yield
    log_event("startup_step", stage=step, seconds=round(time.perf_counter() - started, 3))
