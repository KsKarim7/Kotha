Task: M?-?
Summary:

**Correctness**

- [ ] Does exactly what the brief asked, nothing extra
- [ ] I ran it for real (not only the tests) and saw it work
- [ ] Edge cases tried: silence, noise, a very long utterance, malformed input, the network down

**Tests**

- [ ] New behaviour has tests; all tests and CI pass
- [ ] No test was weakened or deleted to make it pass

**Simplicity**

- [ ] No new dependency without a written reason
- [ ] Comment policy followed: no explanatory comments, docstrings, banners or emojis; at most a short one-line comment above a major block
- [ ] No dead code, commented-out code or speculative abstraction
- [ ] Could it be shorter without losing clarity?

**Performance and reliability**

- [ ] Model inference stays off the event loop
- [ ] Every network call has a timeout, and failures fall back gracefully (re-prompt or handoff)
- [ ] Models load once, not per request

**Security and privacy**

- [ ] No secrets, `.env`, model weights or personal audio in the diff
- [ ] No transcripts, numbers, PINs or NIDs in logs
- [ ] Account actions are checked in code, not by the prompt

**Documentation**

- [ ] README, ARCHITECTURE, CHANGELOG and decision records updated where relevant
- [ ] Any new number in docs comes from a committed evaluation run

**Study**

- [ ] Everything I don't yet understand is logged in `docs/STUDY.md`
