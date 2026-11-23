# 0007: Gemini through the google-genai SDK behind an LLMClient interface
Status: proposed · Date: 2026-11-19

## Context
M6 needs an LLM that writes short, cited answers in Bangla, English and code-mixed text. The
stack names Gemini for the cloud path and Ollama for on-prem, so the rest of the system must not
depend on one provider. The project has a Gemini API key on the free tier. Facts below were
checked on the date above; numbered sources are under References.

- **SDK.** Google's documentation names `google-genai` as the official Python SDK, generally
  available; the older `google-generativeai` is deprecated [1]. Version 3.5.0 needs Python 3.11 or
  later and is licensed Apache-2.0 [2].
- **SDK behaviour, read from the installed 3.5.0.** `HttpOptions.timeout` is in milliseconds. With
  no `retry_options` a request is tried once. API failures raise `errors.ClientError` or
  `errors.ServerError` with a `code`. Transport failures surface as `httpx2` exceptions.
  `response.text` can be empty, for example when the output limit is reached.
- **Models.** `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` and
  `gemini-3.1-flash-lite` are listed as stable. `gemini-2.5-flash` and `gemini-2.5-pro` are
  deprecated and will be routed to other models from 2026-12-01 [3].
- **Free tier.** The pricing page lists a free tier for the stable models above and says free-tier
  content is used to improve Google's products [4]. Rate limits apply per project, and daily quotas
  reset at midnight Pacific time. The documentation does not print free-tier numbers for these text
  models; it says limits can be viewed in Google AI Studio [5].
- **Thinking.** Output-token limits include thinking tokens, and hitting the limit while thinking
  returns cut-off or empty output; both chosen models default to `minimal` [6]. Both accepted an
  explicit `MINIMAL` (observed on this key).
- **Server deadline.** `HttpOptions.timeout` also sends `X-Server-Timeout`, and the API refused 4 s
  with "Minimum allowed deadline is 10s" (observed on this key).
- **Anecdotal probe, not a measurement and not committed.** A handful of fictional prompts on this
  key: `gemini-3.1-flash-lite` answered quickly but sometimes returned 503, `gemini-3.5-flash-lite`
  answered every time but more slowly, and the two larger flash models returned 503. This only
  suggested trying a second model on retry; M8 measures latency properly.

## Decision
- `kotha.llm.client` defines the `LLMClient` protocol, `generate(system_prompt, user_prompt)`, and
  typed errors. Callers depend only on these. `kotha.llm.fake.FakeLLMClient` serves tests.
- `kotha.llm.gemini.GeminiClient` uses `google-genai`. The key comes only from `GEMINI_API_KEY`.
- Every call's first attempt uses `gemini-3.1-flash-lite`, even right after it failed (see the
  second update). Up to two retries, on a timeout, rate limit, 5xx or connection failure, use
  `gemini-3.5-flash-lite`, so a retry does not depend on the same overloaded model. Rejected
  requests and empty or incomplete answers are not retried.
- `generate` has a hard deadline of 12 s in total. Every attempt but the last gets at most 4 s, a
  retry runs only if at least 2 s are left, and the last attempt gets whatever remains. Each attempt
  runs on a pool of 16 workers and is abandoned at its budget. The 2 s rule is a guard for other
  settings: with these, two capped attempts always leave at least 4 s. A 12 s client timeout is
  also set; it applies per httpx phase and sends `X-Server-Timeout: 12`, so it is a backstop that
  frees an abandoned worker and lets the server drop the work.
- Thinking is set to `MINIMAL` and the output limit is 512 tokens. An answer counts only if the model
  finished normally.
- Redirects are not followed, so the key header and the prompt cannot be sent to another host.
- Every SDK, transport and parsing failure becomes a typed error, raised outside the handler so it
  carries neither the provider's message nor the SDK error, whose request holds the key and prompt.
- The key must be printable ASCII without spaces, checked at start-up.
- `generate` is blocking. Callers on the event loop must use `asyncio.to_thread`.

## Update 2026-11-23: one retry was not enough
A real browser call handed off with `llm_seconds` 12.84, the client's deadline. Diagnostic probes
on this key that day, recording only status codes and timings (not measurements, and not from
`eval/`), showed:
- `gemini-3.1-flash-lite` returned `503 UNAVAILABLE` ("This model is currently experiencing high
  demand") to nearly every request, each within a second or two.
- `gemini-3.5-flash-lite` mostly answered, but sometimes returned 503 too or did not answer for
  several seconds.
- With one retry, the retry inherited about 11 s, and a stalled request used all of it.

Hence the extra retry and the 4 s cap on all but the last attempt. The free tier's capacity stays
outside this code: when both models are overloaded, the call still hands off.

## Update 2026-11-24: no rest for the primary
The first update also let `gemini-3.1-flash-lite` rest for 60 s after a transient failure, to save
the second or so a 503 costs. The next committed latency run (MLflow `d4a53096`, CPU) handed off
28 of 72 turns, 24 of them `no_answer`, where the MS1 run handed off 14. The model that answered
is not logged per turn, but given the rest and the primary's 503 rate that day, most answers
most likely came from `gemini-3.5-flash-lite`. A diagnostic probe sent the app's own prompts for
some of those turns to both models (not a measurement): the primary's answers passed the verifier
every time, while `gemini-3.5-flash-lite` often replied `NO_ANSWER` although the gold passage was
among the four; retrieval put the gold passage among the four both before and after #68. So the
rest is removed: every call starts on the primary again. The cost while the primary is
overloaded: a fast 503 on each turn, or, when it stalls, up to 4 s, after which two stalled
retries leave the last attempt only about 4 s.

## Alternatives considered
- **`google-generativeai`:** deprecated [1].
- **Plain HTTP calls:** more code to keep correct, and no benefit over the official SDK.
- **The SDK's own retry option:** it retries the same model, which does not help when that model is
  overloaded.
- **`gemini-3.5-flash-lite` alone:** dependable in the probe but slower, which matters on a call.
- **Ollama now:** planned for on-prem; it can implement the same protocol later.

## Consequences
- **Data use.** On the free tier, what is sent may be used to improve Google's products. Prompts
  therefore hold only the fictional knowledge base and the caller's question, never account data.
  Before real people's questions are sent, this needs their consent or a paid tier.
- **Limits are not known in advance.** Quota errors surface as `LLMRateLimitError`, and the caller
  falls back to the handoff phrase.
- **The deadline is 12 s of silence in the worst case** before the handoff phrase. The numbers are
  settings, not measurements; M8 measures latency and may change them.
- **An abandoned attempt keeps its worker** until the backstop ends it, which can take 12 s per
  stalled phase, and longer if a server trickles bytes, because the read timeout restarts on each
  one; the caller's deadline still holds. A stalled call can abandon three workers, so 6 concurrent
  stalled calls fill the pool of 16 for tens of seconds, and later calls time out meanwhile. Size
  the pool to at least three times the expected concurrent calls (M8).
- **Slow answers are cut off.** An attempt that is not the last gets 4 s. When the first two
  attempts time out, the last gets 4 s too, where the old single retry got about 8 s, so an answer
  that would have taken 5-8 s now ends in a handoff. Fast failures (a quick 503) leave the last
  attempt most of the deadline.
- **The two models are not interchangeable.** `gemini-3.5-flash-lite` declines more answerable
  questions with `NO_ANSWER`, so when the primary is overloaded, more turns hand off. Choosing the
  retry model by answer rate needs its own evaluation; until then the primary goes first.
- **More load during an overload.** Each turn can send three requests, and abandoned ones keep
  running on the server until `X-Server-Timeout`, so the client can add to its own 503s and 429s.
- **Unexpected exceptions are not hidden.** Only known transport and parse failures become typed
  errors; anything else is a bug and is raised as it is.
- **One client per process.** `GeminiClient` owns a connection pool and workers; `close()` frees them.
- **Model ids change often.** They are constants in one file, and this record should be revisited
  when Google deprecates either model.

## References
1. Gemini API libraries: https://aistudio.google.com/docs/libraries
2. google-genai on PyPI: https://pypi.org/project/google-genai/3.5.0/
3. Gemini models: https://aistudio.google.com/docs/models
4. Gemini API pricing: https://aistudio.google.com/docs/pricing
5. Gemini API rate limits: https://aistudio.google.com/docs/rate-limits
6. Gemini thinking: https://aistudio.google.com/docs/thinking
