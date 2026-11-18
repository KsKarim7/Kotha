# 0008: Answer-only pipeline with guardrails in code
Status: accepted · Date: 2026-11-19

## Context
MS1 needs Kotha to answer policy questions (UC4) from the knowledge base, by text first and by voice
later, before tools, caller verification and the BSS exist (MS2). The LLM (ADR 0007) can invent a
price, claim it activated a pack, answer in the wrong language or follow an instruction hidden in
the caller's words. A wrong number or a false "done" on a phone call is worse than no answer. The
blueprint asks for citations on every sentence, numbers checked against the cited chunks, one
regeneration and then the handoff phrase, and permission checks in code rather than in the prompt.

## Decision
`kotha.dialogue.answer.AnswerPipeline` answers one question at a time:

1. `kotha.dialogue.routing.is_account_request` runs first. A question about the caller's own
   balance or credit, or a request to buy, activate, cancel, recharge or block something, gets a
   fixed reply that this is not available yet. Neither retrieval nor the LLM runs.
2. Hybrid retrieval (ADR 0006) returns 4 chunks. Nothing retrieved gives the fixed "please repeat"
   reply; a database or embedding error (`psycopg.Error`, `RuntimeError`) gives the handoff phrase.
3. The LLM gets a constant system prompt and a user prompt with the reply language, the numbered
   passages and the question last. The reply language is Bangla when the question has any Bengali
   letter, so code-mixed questions get Bangla replies; otherwise English.
4. Code checks the answer: `verify_answer` (ADR 0006 / M4-9) with no money amounts, so any amount
   of money fails; the reply language; at most two sentences; and no claim about the caller's own
   account (`ACCOUNT_CLAIMS`: first-person actions such as "I have activated NT-30GB", passive
   actions such as "your pack has been activated", states such as "NT-30GB is active now" or
   "your balance is low", holdings such as "you have 30 GB left", and the Bangla forms), checked
   clause by clause on normalised text and skipped after a conditional lead-in ("if", "once",
   "whether", ...).
5. A rejected answer is regenerated once, with a note before the question naming what was wrong
   but not the rejected text, and only if the first answer came within 4 s
   (`REGENERATE_WITHIN_S`), so a turn waits at most about 16 s for the LLM, not 24 s. A second
   rejection, a reply containing `NO_ANSWER`, or any `LLMError` gives the fixed handoff phrase in
   the caller's language ("let me connect you to an agent").

## Alternatives considered
- **Let the LLM decide what is an account question.** Rejected: a prompt injection could talk it
  out of the rule, and MS1 has no tools to make the decision safe. The intent classifier comparison
  in M6 full (MS2) can replace these rules.
- **Regenerate after an LLM error too.** Rejected: `GeminiClient` already retried on a second model
  within 12 s; a second round could leave the caller in up to 24 s of silence.
- **Truncate long answers to two sentences.** Rejected: the cut can drop a condition ("only in
  person") and change the meaning. A too-long answer is regenerated instead.
- **Ask the LLM to say "I don't know" in its own words.** Rejected: that sentence has no citation,
  so it would fail the check and cost a regeneration. A fixed `NO_ANSWER` token goes straight to
  the handoff.

## Consequences
- Prices, balances and account actions are kept from the LLM in two layers: account questions
  never reach it, and its answers are rejected for any money amount or any recognised claim about
  the caller's account.
- The routing rules are keyword patterns. They catch the phrasings in `tests/test_dialogue_routing.py`
  and route none of the 72 retrieval test questions, but they will miss other phrasings. A direct
  request ("please activate", "for me", a leading imperative, a Bangla imperative such as "চালু
  করে দিন") always counts; a balance question or a chained verb ("... and activate") does not
  count when its clause has a how-to word ("how", "where", "কীভাবে", "code", ...). ASR text has no
  punctuation, so a spoken request can hide behind a how-to word; it then goes to the LLM, where
  the answer checks still apply.
- The claim check is a list of patterns, not an understanding of the sentence. Three review rounds
  each found new wordings that got through and new policy sentences that were wrongly rejected;
  the patterns now cover every wording found and no knowledge-base sentence matches them, but
  that is a test of known cases, not a proof. A claim about "the pack" without "your" or a code, or
  a new phrasing, can still get through, and a policy answer phrased like a claim can be handed
  off. Known examples from the last review: bare product requests ("NT-30GB please",
  "NT-30GB চাই") are not routed, and casual state words ("your NT-30GB pack is all set", "you're
  now on NT-30GB", "ready to use on your line") are not flagged. In MS1 no claim can be true, because nothing can change an account; MS2 should replace
  the patterns with a structural rule (answers built from tool results) and keep these as tests.
- An LLM error or a slow first answer hands off rather than trying again, so a flaky provider means
  more handoffs; M8's latency harness should show how often that happens.
- Handoff in MS1 is only the phrase and `BotReply.handoff_reason`; routing to a person is MS2.
- `AnswerPipeline.reply` blocks (retrieval and the LLM), so the WebSocket loop (M8) must call it
  through `asyncio.to_thread`.
