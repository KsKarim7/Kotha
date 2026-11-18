import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager

from kotha.db.schema import connect
from kotha.dialogue.answer import AnswerPipeline, BotReply
from kotha.llm.gemini import GeminiClient
from kotha.rag.embedder import Embedder
from kotha.rag.retriever import HybridRetriever
from kotha.text.lexicon import LEXICON_PATH, read_lexicon


@contextmanager
def live_pipeline(database_url: str) -> Iterator[AnswerPipeline]:
    llm = GeminiClient()
    try:
        with connect(database_url) as connection:
            retriever = HybridRetriever(connection, Embedder(), read_lexicon(LEXICON_PATH))
            yield AnswerPipeline(retriever, llm)
    finally:
        llm.close()


def database_url_or_exit(parser: argparse.ArgumentParser) -> str:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is not set")
    return database_url


def shown(reply: BotReply) -> str:
    label = f"{reply.kind}: {reply.handoff_reason}" if reply.handoff_reason else reply.kind
    lines = [f"kotha [{label}]> {reply.text}"]
    lines += [f"  source: {source.title} ({source.chunk_id})" for source in reply.sources]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask Kotha questions as text.")
    parser.parse_args(argv)
    with live_pipeline(database_url_or_exit(parser)) as pipeline:
        while True:
            try:
                question = input("you> ")
            except EOFError:
                break
            print(shown(pipeline.reply(question)))


if __name__ == "__main__":
    main()
