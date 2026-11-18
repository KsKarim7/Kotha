import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from kotha.dialogue.answer import AnswerPipeline, BotReply, ReplyKind
from kotha.dialogue.chat import database_url_or_exit, live_pipeline, shown
from kotha.dialogue.routing import Language

SCENARIOS_PATH = Path(__file__).resolve().parents[4] / "data" / "scenarios" / "text_scenarios.json"


@dataclass(frozen=True)
class Turn:
    caller: str
    kind: ReplyKind
    lang: Language
    doc_id: str | None


@dataclass(frozen=True)
class Scenario:
    id: str
    turns: tuple[Turn, ...]


def read_turn(raw: dict[str, str], location: str) -> Turn:
    missing = {"caller", "kind", "lang"} - raw.keys()
    if missing:
        raise ValueError(f"{location}: missing {', '.join(sorted(missing))}")
    if raw["kind"] not in set(ReplyKind):
        raise ValueError(f"{location}: unknown kind {raw['kind']!r}")
    kind = ReplyKind(raw["kind"])
    lang = raw["lang"]
    if lang not in get_args(Language):
        raise ValueError(f"{location}: unknown lang {lang!r}")
    doc_id = raw.get("doc_id")
    if (kind == ReplyKind.ANSWER) != (doc_id is not None):
        raise ValueError(f"{location}: doc_id is required for an answer and only for an answer")
    return Turn(raw["caller"], kind, lang, doc_id)


def read_scenarios(path: Path) -> list[Scenario]:
    raw_scenarios = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
    scenarios = [
        Scenario(
            raw["id"],
            tuple(
                read_turn(turn, f"{raw['id']} turn {number}")
                for number, turn in enumerate(raw["turns"], 1)
            ),
        )
        for raw in raw_scenarios
    ]
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError(f"{path}: scenario ids must be unique")
    return scenarios


def turn_failures(turn: Turn, reply: BotReply) -> list[str]:
    failures = []
    if reply.kind != turn.kind:
        failures.append(f"expected {turn.kind}, got {reply.kind}")
    if reply.lang != turn.lang:
        failures.append(f"expected lang {turn.lang}, got {reply.lang}")
    if turn.doc_id and turn.doc_id not in {source.doc_id for source in reply.sources}:
        failures.append(f"expected a citation of {turn.doc_id}")
    return failures


def run_scenarios(
    pipeline: AnswerPipeline, scenarios: Sequence[Scenario], report: Callable[[str], None]
) -> int:
    failed_turns = 0
    for scenario in scenarios:
        for number, turn in enumerate(scenario.turns, 1):
            reply = pipeline.reply(turn.caller)
            failures = turn_failures(turn, reply)
            failed_turns += bool(failures)
            report(f"{'FAIL' if failures else 'pass'} {scenario.id} turn {number}")
            report(f"you> {turn.caller}\n{shown(reply)}")
            for failure in failures:
                report(f"  {failure}")
    turn_count = sum(len(scenario.turns) for scenario in scenarios)
    report(f"{turn_count - failed_turns} of {turn_count} turns passed")
    return failed_turns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay scripted text conversations.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    args = parser.parse_args(argv)
    scenarios = read_scenarios(args.scenarios)
    with live_pipeline(database_url_or_exit(parser)) as pipeline:
        failed_turns = run_scenarios(pipeline, scenarios, print)
    sys.exit(1 if failed_turns else 0)


if __name__ == "__main__":
    main()
