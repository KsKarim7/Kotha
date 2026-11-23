# 0006: Knowledge base and hybrid retrieval with bge-m3, pgvector and in-process BM25
Status: proposed · Date: 2026-11-17

## Context
M4 adds a knowledge base (KB) for Nodi Telecom and hybrid retrieval over it: dense search with
the bge-m3 embedding model in PostgreSQL + pgvector, sparse search with BM25, and a fusion of
the two. These are the project's largest new pieces, so they were checked on this machine before
any code was written. Numbered sources are listed under References.

- **Model.** `BAAI/bge-m3` at revision `5617a9f61b028005a4858fdac845db406aefb181` (the `main`
  revision on 2026-11-17) is MIT-licensed, gives 1024-dimensional dense vectors and handles more
  than 100 languages and inputs up to 8192 tokens [1]. That revision has no safetensors file: the
  weights are `pytorch_model.bin` (a pickle, 2166 MiB) plus an ONNX copy (2162 MiB) [2]. Safetensors
  versions exist only in open, unmerged pull requests opened by the `SFconvertbot` bot, not by
  BAAI [3]. The pinned snapshot that sentence-transformers downloads is 2187 MiB.
- **Hidden download.** When transformers 5.23.0 loads a `.bin`, it starts a background thread
  that looks for a bot safetensors pull request, asks a conversion Space to open one if none
  exists, and downloads that pull request's `model.safetensors` into the cache "for next time"
  [4]. On this machine it fetched 2.2 GB from `refs/pr/130`. The model actually used was the
  pinned `.bin`, loaded with `torch.load(weights_only=True)` [4]. The thread is skipped when
  `DISABLE_SAFETENSORS_CONVERSION` is true or the Hub is offline [4]. The bot snapshot was
  deleted from the cache afterwards.
- **Loader.** sentence-transformers 6.3.0 and FlagEmbedding 1.4.2 [5][6] produced identical
  dense vectors for three test sentences (Bangla, English and code-mixed), already L2-normalised;
  the model's `modules.json` is Transformer, CLS pooling, Normalize [2]. Added to a copy of this
  project with `uv add`, both kept torch `2.15.0+cpu` from the pytorch-cpu index (ADR 0003) and
  changed no locked version. sentence-transformers adds 13 packages (about 92 MiB installed);
  FlagEmbedding adds 27, including sentence-transformers itself, datasets, accelerate, peft,
  pandas and pyarrow (about 333 MiB). FlagEmbedding's loader calls `snapshot_download` without a
  revision, so it cannot be pinned by model name, and it would also download the ONNX copy [6].
- **`uv run --with`.** Running the probe with `uv run --with sentence-transformers` installed a
  CUDA build of torch (`2.15.0+cu132`) into the temporary environment, because `--with` does not
  apply the project's `tool.uv.sources` index for torch.
- **pgvector.** The current release is 0.8.7 (2026-10-01) under the PostgreSQL licence, and the
  official image has the tag `pgvector/pgvector:0.8.7-pg16` [7][8]. By default pgvector does exact
  nearest-neighbour search with perfect recall; `<=>` is cosine distance; an HNSW index indexes
  `vector` columns up to 2,000 dimensions and trades some recall for speed [7]. The Python package
  `pgvector` 0.5.1 registers the `vector` type on a psycopg 3 connection with `register_vector`
  after `CREATE EXTENSION IF NOT EXISTS vector`, and on a pool through `configure=` [9]. psycopg
  3.3.9's pure-Python package needs a system `libpq`, which this machine does not have;
  `psycopg[binary]` ships its own [10].
- **CI.** GitHub-hosted jobs can run a PostgreSQL service container on Ubuntu runners, with a
  `pg_isready` health check and the port mapped to `localhost` [11]. CI uses `ubuntu-latest`.
  Not yet tried in this repository's CI.
- **Docker.** The Docker CLI (29.9.2) and Compose (v5.6.1) on this machine come from Docker
  Desktop on Windows, and its daemon is not reachable from WSL. No PostgreSQL runs locally.
- **BM25.** `rank_bm25` 0.2.2 was released on 2022-02-16 and depends only on numpy; its last
  commit is from 2024-10-08 [12]. Its `BM25Okapi` is a few lines: idf
  `log(N − n + 0.5) − log(n + 0.5)`, negative idf replaced by `epsilon · mean idf`, term score
  `idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · len / avgdl))`, with `k1 = 1.5`, `b = 0.75`,
  `epsilon = 0.25` [12].

## Decision
**KB format.** One Markdown file per document per language in `data/kb/`, for example
`packages-internet.bn.md` and `packages-internet.en.md`. Each file starts with a header giving
`doc_id`, `lang`, `title` and a line saying the content is fictional. Every document exists in
both Bangla and English.

**KB versus the mock BSS (M5).** The KB holds descriptions, rules and policies, and stable
product facts such as package code, data volume and validity. It never holds prices or balances,
which come only from the BSS. The M5 seed data must match the KB's product facts, and M5 adds a
test that checks this.

**Chunking.** Split by heading, about 150–300 words per chunk. A package table is never split.
Each chunk stores `doc_id`, `lang`, `title` and its heading.

**Storage.** PostgreSQL + pgvector from `pgvector/pgvector:0.8.7-pg16`, in Docker Compose for
local development and as a service container in CI. Search is exact cosine search (`<=>`) with no
index while the KB has only hundreds of chunks. An HNSW index is added when a measurement shows
exact search is too slow; as a plan, that is checked once the KB passes about 10,000 chunks. The
client is `psycopg[binary]` 3 with `pgvector`'s `register_vector`.

**Embedding.** sentence-transformers in the runtime dependencies, with `BAAI/bge-m3` pinned at
`5617a9f61b028005a4858fdac845db406aefb181`:
- The model is downloaded once at the pinned revision. The service then loads it with
  `local_files_only=True` (or `HF_HUB_OFFLINE=1`), so it never contacts the Hub at runtime.
- The pinned `pytorch_model.bin` is loaded as transformers does, with
  `torch.load(weights_only=True)`, and `DISABLE_SAFETENSORS_CONVERSION=true` is set before
  loading.
- Weights from bot pull requests are never used. This is a supply-chain choice: only files
  published by the model's owner at a pinned commit are loaded.
- Only the dense vectors are used (1024 dimensions, already normalised). bge-m3's sparse and
  multi-vector (ColBERT) outputs are not used.
- Inference runs off the event loop, on the CPU by default so the GPU stays with ASR.

**Sparse retrieval.** BM25 in process, built from the stored chunks when the service starts.
Tokens are `kotha.text.normalize.normalize(text).split()`, after the Markdown characters `` ` ``,
`|`, `<` and `>` are replaced by spaces, because the KB writes every code in backticks and
`normalize` keeps these characters; `normalize` itself stays unchanged because WER uses it. We
write our own `BM25Okapi` in the code task, matching `rank_bm25`'s formula and defaults, with a
hand-computed test; `rank_bm25` is not a dependency.

**Query expansion.** The loanword lexicon moves from `data/testsets/loanwords.csv` to
`data/lexicon/loanwords.csv` in the code task, because the running service will read it and test
data is the wrong home for that. The move updates `LEXICON_PATH` in `eval/wer.py`,
`eval/lint_references.py`, `data/testsets/GUIDELINE.md` and the runbook. It stays a spelling
lexicon owned by the transcription guideline and is never tuned on test results.

**Fusion.** Reciprocal rank fusion with `k = 60` over the top 20 of each retriever, returning
the best 4 chunks. The paper's authors fixed `k = 60` in a pilot study and found its exact value
not critical [13].

**Testing.** Database tests read `DATABASE_URL` and are skipped when it is not set, so the suite
still runs locally without Docker. CI sets it, using the service container.

**Evaluation plan.** `data/testsets/rag_test.csv` holds typed questions in bn, en and mix, each
with its gold `doc_id`. The evaluation reports hit@3 and MRR for dense, BM25, hybrid, and hybrid
with lexicon expansion. A column for ASR-transcribed versions of the questions is filled after
Khaled's recordings. Questions written by the KB's author inflate the scores, because they reuse
the KB's wording; Khaled reviews them and adds his own.

## Update 2026-11-23: exact package codes
- **Defect.** A bare `NT-30GB` did not bring the internet packages document into the hybrid top 4,
  although exact codes are BM25's job. `normalize` splits a code at its hyphen, so `NT-30GB`
  becomes `nt` (in 20 of 126 chunks) and `30gb` (in 14). A whole-code token would not help:
  `30gb` occurs only inside `NT-30GB`, so it already has the code's IDF. The cause is the ranking.
  `NT-30GB` is the KB's example code, and the rollover and activation chunks repeat it two or
  three times, while the packages table names it once in a long chunk, which BM25's length
  normalisation penalises. Dense search ranks the same chunks first. Markup stripping and the
  lexicon play no part; the lexicon only maps `জিবি` to `gb`.
- **Decision.** BM25 also scores every table row on its own, with its chunk's title and heading,
  and a chunk counts with its best unit, the whole chunk or one of its rows. A row that lists a code
  is short, so it outranks long chunks that only mention the code. Every package code in the
  catalogue is a regression test: the document that lists it must be in BM25's top 4 and in
  hybrid search's top 4. The test does not use `rag_test` questions. The rows join the index
  statistics, so every BM25 ranking changes a little; `EVALUATION.md` shows the old and new rows.
- **Not done: Bangla stemming.** Exact tokens miss inflected forms (`প্যাকেজের` against
  `প্যাকেজ`). A light suffix stripper is an experiment for later, and it needs a dev/test split of
  `rag_test` first, so that the suffix list is chosen on the dev half and scored once on the
  test half, not tuned on the questions that report it.
- **The live retriever stays hybrid.** Dense alone scores higher on the typed questions, but they
  were written by the KB's author in the KB's wording, which favours dense search. Callers' words
  reach retrieval through ASR, where spelling errors, loanwords and codes are where exact tokens
  are meant to help. The choice is made when the ASR-transcribed column exists (it needs the
  recorded `asr_test` set); until then, hybrid keeps both paths. This record stays proposed.

## Alternatives considered
- **Qdrant or Chroma:** another service or store to run, while PostgreSQL is already in the stack
  and pgvector covers exact and approximate search.
- **FAISS in memory:** no persistence or metadata filtering without extra code, and the KB is
  small enough for exact search in the database.
- **LlamaIndex or LangChain:** large abstractions over a few dozen lines of retrieval code; not in
  the runtime. At most an optional notebook comparison, as decided earlier.
- **FlagEmbedding:** the same dense vectors, but 27 extra packages instead of 13 and no revision
  pinning by model name.
- **bge-m3's own sparse output instead of BM25:** not used; BM25 on normalised tokens is easy to
  inspect and test, and works on exact loanword spellings.
- **A cross-encoder reranker:** later, if a measurement shows the top 4 often miss the right
  chunk.
- **`rank_bm25`:** no release since 2022, and the algorithm is short enough to own and test.
- **Safetensors from a bot pull request, or a local conversion:** rejected or not needed; the
  first loads weights BAAI did not publish, the second adds a manual step and a second copy.

## Consequences
- **New runtime dependencies:** sentence-transformers (13 new packages, about 92 MiB installed),
  `psycopg[binary]` and `pgvector`, plus a 2187 MiB model download into the Hugging Face cache.
- **transformers becomes a runtime dependency** through sentence-transformers. That reverses the
  earlier rule that transformers is never a project dependency (it was conversion-only, through
  `uv run --with transformers==5.9.0` in the runbook). The CTranslate2 conversion command must be
  re-checked against the project's own transformers when the code lands (M4-6).
- **Never use `uv run --with` for packages that bring torch:** it bypasses the pytorch-cpu source
  and installs a CUDA build.
- **Docker:** local development needs Docker Desktop running with WSL integration turned on.
  Khaled does that later; it is not the agent's task.
- **Memory budget:** WSL has about 10 GB of RAM, shared by the ASR model, bge-m3 in fp32 on the
  CPU (about 2.2 GB of weights) and Docker Desktop's PostgreSQL, which runs in the same WSL VM.
  On the GPU, bge-m3 in fp16 would take about 1.1 GiB of the 4 GiB next to ASR. This is a budget
  to measure in M8, not a measurement.
- **Model updates** are deliberate: a new bge-m3 revision means a new pin, a re-embedded KB and a
  new evaluation run.
- **Approximate search** (HNSW) would change results compared with exact search, so adding it
  needs its own evaluation run.

## References
1. BAAI/bge-m3 model card at `5617a9f6`:
   https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/README.md
2. BAAI/bge-m3 files at `5617a9f6` (sizes from the Hub API, `modules.json`,
   `1_Pooling/config.json`):
   https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181
3. BAAI/bge-m3 pull requests by `SFconvertbot`, for example
   https://huggingface.co/BAAI/bge-m3/discussions/130
4. transformers 5.23.0, `src/transformers/modeling_utils.py` (`can_auto_convert`, the
   `Thread-auto_conversion` thread, `load_state_dict` with `weights_only=True`) and
   `src/transformers/safetensors_conversion.py`:
   https://github.com/huggingface/transformers/tree/v5.23.0/src/transformers
5. sentence-transformers 6.3.0: https://pypi.org/project/sentence-transformers/6.3.0/
6. FlagEmbedding 1.4.2, `FlagEmbedding/finetune/embedder/encoder_only/m3/runner.py`
   (`snapshot_download` without a revision): https://github.com/FlagOpen/FlagEmbedding/tree/v1.4.2
7. pgvector README, CHANGELOG (0.8.7, 2026-10-01) and LICENSE:
   https://github.com/pgvector/pgvector/tree/v0.8.7
8. pgvector Docker image tags: https://hub.docker.com/r/pgvector/pgvector/tags
9. pgvector-python README, section "Psycopg 3": https://github.com/pgvector/pgvector-python
10. psycopg 3 installation: https://www.psycopg.org/psycopg3/docs/basic/install.html
11. GitHub Docs, "Creating PostgreSQL service containers":
    https://docs.github.com/en/actions/use-cases-and-examples/using-containerized-services/creating-postgresql-service-containers
12. rank_bm25 0.2.2 and `rank_bm25.py`: https://github.com/dorianbrown/rank_bm25 and
    https://pypi.org/project/rank-bm25/0.2.2/
13. Cormack, Clarke and Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual
    Rank Learning Methods", SIGIR 2009, which defines `RRFscore(d) = Σ 1/(k + r(d))` with `k = 60`
    fixed in a pilot investigation: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
