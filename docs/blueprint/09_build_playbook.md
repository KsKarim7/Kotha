# Kotha Build Playbook: Claude Project + Claude Code

You plan and learn in a Claude Project, Claude Code writes and runs code in your repository, and you review and approve every change. Follow the steps in order; each one ends with a check you can verify.

## How the Setup Works

Four parts, each with one job. The Claude Project thinks, Claude Code types, GitHub records, and you decide.

| Part | Its job | It never |
| --- | --- | --- |
| **You** | Product owner and reviewer: approve plans, run everything, read every changed line, write evaluation interpretations and the README story, merge | Merge code you can't explain |
| **Claude Project** "Kotha — Architect & Mentor" (claude.ai) | Plans each module, teaches the concepts, writes task briefs for Claude Code, reviews your diffs and logs, quizzes you | Puts code straight into the repository |
| **Claude Code** (VS Code panel, running in WSL: Ubuntu) | Implements one task brief at a time on a branch, runs tests, proposes a commit | Works outside the brief, commits to `main`, pushes without your OK, invents results |
| **GitHub** (KsKarim7/Kotha) | Source of truth: issues, branches, pull requests, CI, milestones | — |

**One task, start to finish:**

1. The Project plans the task.
2. The Project writes a brief.
3. Claude Code plans, then implements on a branch.
4. You run and test it.
5. The Project reviews the diff.
6. You fix, log what to study, and merge.
7. The docs are updated.

Step 5 below spells this out with copy-ready prompts.

## Code Style and the Study Phase

### Comment and style rules for the agent

The code should read like a careful engineer wrote it. These rules are built into the Project instructions (Step 3) and `CLAUDE.md` (Step 4):

- **No explanatory comments by default.** Names, types and small functions carry the meaning.
- **At most a short, plain comment above a major block**, and only when the *why* isn't obvious. For example: `# keep ~300 ms before speech so the first syllable isn't clipped`. One line, no step numbers.
- **None of these:** docstrings, section banners (`# ====`), "Step 1:" comments, emojis, or comments that restate the code.
- **No commented-out code**, and no TODO without an issue number.
- **Specific, idiomatic names:** `turn_audio`, not `data` or `result`.
- **No speculative abstractions:** no base classes, wrappers or config options the current task doesn't need.
- **Error handling only where something can actually fail**, not blanket `try/except` around everything.
- **Small commits** with Conventional Commit messages.

### The study phase

Since the agent writes the code, your understanding comes entirely from studying it, so make the study as rigorous as the build.

- **During the build,** log anything unclear in `docs/STUDY.md` (file, lines, your question) instead of stopping.
- **Study each milestone right after it merges,** not all seven weeks at the end. One week of code is learnable in a few days; seven weeks at once is not.
- **For each module:**
  1. Follow one request through the code from entry point to response.
  2. Run the tests, then break something on purpose and predict the failure before running again.
  3. Rebuild the core function from a blank file, then compare it with the original.
  4. Take the module quiz (template G).
- **A module is done** when every entry in `STUDY.md` is answered and you pass the quiz.

**If asked how you built it,** say it plainly: "I used Claude Code under a written spec, reviewed every change, and studied every module afterwards. I can walk you through any line." Leave Claude Code's commit attribution as it is; that answer only works if the history is honest.

## Step 1: Your Windows 11 Setup (done and verified)

Completed and verified on 5 November 2026; every check in `~/check_setup.sh` passed. This section records what is installed and where, so you can troubleshoot or rebuild it.

### Your setup at a glance

| Item | What you have |
| --- | --- |
| PC | Windows 11 Home (build 26200), Ryzen 5 5600H, 16 GB RAM, NVIDIA GTX 1650 (4 GB) |
| WSL | Version 3.0.3, kernel 6.18; distro **Ubuntu 26.04 LTS**, user `khaled_karim` |
| Ubuntu's disk | `C:\WSL\Ubuntu\ext4.vhdx`, on the NVMe SSD |
| WSL limits | 10 GB RAM and 8 GB swap, set in `%USERPROFILE%\.wslconfig`; the swap file is on G: |
| Docker | Docker Desktop, WSL 2 engine, Ubuntu integration on; **disk at `G:\Kotha\docker`**; Docker 29.9.2, Compose v5.6.1 |
| Editor | VS Code with the WSL extension; the project always opens as **WSL: Ubuntu** |
| Claude Code | VS Code extension, installed **in WSL: Ubuntu**, signed in, Manual mode |
| Linux tools | git 2.53, gh 2.46, ffmpeg 8.0.1, gcc 15.2, curl, unzip |
| Python | uv 0.13.7 managing **Python 3.11.18** for Kotha; Ubuntu's own Python 3.14 is left alone |
| GitHub | `gh` logged in as **KsKarim7** over HTTPS; repo **KsKarim7/Kotha** (public) cloned to `~/kotha` |
| GPU | Visible in Ubuntu through the Windows driver 592.82 (CUDA 13.1); roughly 2.5–2.7 GB of the 4 GB is free for models, because the GPU also drives the screen |

### Where things live

- **Code:** `~/kotha` inside Ubuntu, on the fast SSD. From Windows File Explorer it's at `\\wsl.localhost\Ubuntu\home\khaled_karim\kotha`.
- **G:\Kotha** holds machine files only: `docker` (images, databases, downloaded models) and `wsl` (the swap file). G: is a hard disk, so it gets the large, rarely-touched data. **Never put code there.**
- Large datasets, recordings and backups can go in a new folder such as `G:\Kotha\data`.

### Starting work each day

1. Start **Docker Desktop**.
2. In an Ubuntu terminal: `cd ~/kotha && code .`
3. Check that the bottom-left of VS Code shows **WSL: Ubuntu**.
4. Open the Claude Code panel (orange ✳ icon) in that window.

### Re-check any time

Run `bash ~/check_setup.sh` in the Ubuntu terminal. Every line should start with **OK**. The script, in case you ever need to recreate it:

```bash
cat > ~/check_setup.sh <<'EOF'
#!/usr/bin/env bash
check() {
  out=$("$@" 2>&1); rc=$?
  first=$(printf '%s\n' "$out" | grep -m1 .)
  if [ $rc -eq 0 ]; then echo "OK    $* -> $first"; else echo "FAIL  $* -> $first"; fi
}
echo "== Kotha setup check =="
check grep -m1 PRETTY_NAME /etc/os-release
check uname -r
check git --version
check gh auth status
check uv --version
check uv run --no-project --python 3.11 python --version
check ffmpeg -version
check gcc --version
check docker --version
check docker compose version
check docker run --rm hello-world
check code --version
check git -C "$HOME/kotha" remote get-url origin
check git -C "$HOME/kotha" status -sb
check nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "WSL memory: $(free -h | awk '/Mem:/ {print $2}')   swap: $(free -h | awk '/Swap:/ {print $2}')"
echo "free on C: $(df -h /mnt/c | awk 'NR==2 {print $4}')   free on G: $(df -h /mnt/g | awk 'NR==2 {print $4}')"
EOF
bash ~/check_setup.sh
```

### If you ever rebuild from scratch

**PowerShell (Admin):**

```powershell
wsl --install -d Ubuntu
wsl --update
@'
[wsl2]
memory=10GB
swap=8GB
swapFile=G:\\Kotha\\wsl\\swap.vhdx
'@ | Set-Content -Path "$env:USERPROFILE\.wslconfig" -Encoding ascii
wsl --shutdown
```

Then install Docker Desktop (WSL 2 engine, Ubuntu integration on, disk image location `G:\Kotha\docker`) and VS Code with the **WSL** extension.

**Ubuntu terminal:**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git gh curl ffmpeg build-essential unzip
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.11
git config --global user.name "KsKarim7"
git config --global user.email "kskarim177@gmail.com"
git config --global init.defaultBranch main
git config --global core.autocrlf input
git config --global pull.rebase false
gh auth login
gh auth setup-git
gh repo clone KsKarim7/Kotha ~/kotha
cd ~/kotha && code .
```

In that VS Code window, install **Claude Code** with **Install in WSL: Ubuntu**, then sign in.

### Lessons from the setup (troubleshooting)

- **Never press Ctrl+C during `wsl --manage --move`.** If a move is interrupted, check the registered location and make a backup copy before doing anything else.
- **Run Linux commands inside the Ubuntu terminal,** not through `wsl -- bash -c '…'` from PowerShell, which mangles quotes.
- **VS Code server install fails:** delete `~/.vscode-server/bin/<id>*`, then rerun `code .`.
- **`docker` "not recognized" in PowerShell:** that window was opened before Docker was installed. Open a new one.
- **Forgot the Ubuntu password:** run `wsl -d Ubuntu -u root passwd khaled_karim` in PowerShell.
- **`gh auth login` can't open the browser:** go to github.com/login/device and enter the code printed in the terminal.
- **Never install an NVIDIA driver inside Ubuntu.** Update the Windows driver instead.
- **If the Claude Code panel shows Windows paths (`C:\`),** it's running on the Windows side. Open the WSL window and install the extension **in WSL: Ubuntu**.

**Last housekeeping:** delete `G:\Kotha\wsl-backup`, the temporary safety copy from the move.

## Step 2: Create the Claude Project

1. In claude.ai, open **Projects** → **New project**.
   - Name: `Kotha — Architect & Mentor`
   - Description: "Plan, teach and review the Kotha voice AI build for the AI Engineer (Voice) role."
2. **Add project knowledge** (the files Claude reads in every chat of this project): the 9 files in the `kotha-project-knowledge` zip (job description, original and rewritten CV, the five blueprint tabs, and this playbook).
3. **Later, your GitHub repo:** once real code is merged, in project knowledge click **+** → **GitHub**, pick **KsKarim7/Kotha** and the main folders. Select key folders rather than everything, and click **Sync** after each merge so reviews see current code.
4. **Set the project instructions:** paste the text from Step 3.
5. **One chat per module** ("M0 — Repo setup", "M1 — Audio foundation"), never one endless chat. Start each new chat with a 3-line status: what's done, what's next, any open problem.

**Check:** in a new chat, ask "What is Milestone 1's definition of done?" The answer should come from the blueprint, not generic advice.

## Step 3: Project Instructions (paste as-is)

Paste this into the project's instructions field. Edit it as you learn which corrections you keep making.

```markdown
# Role
You are my senior AI engineer, architect and mentor for "Kotha": a Bangla-English voice AI agent
and call-analytics platform for a fictional telecom operator, Nodi Telecom. I am Khaled, a junior
AI engineer building this to learn voice AI for an AI Engineer (Voice) role. The Kotha blueprint
in project knowledge is the source of truth for scope, architecture, stack and milestones.
Claude Code writes the code; I review it, run it and study it after each milestone.

# What you do here
1. Plan: split the current module into tasks of at most ~150 changed lines each, in blueprint order.
2. Brief: write task briefs for Claude Code in the format below. Don't write the implementation
   here unless I explicitly ask.
3. Review: when I paste a diff, test output or logs, review against the brief, the blueprint and
   the checklist below. Report findings as blocker / should-fix / nit, citing the exact line.
4. Teach (study phase): when I ask, explain concepts and code in plain words with one small
   example, then ask me 2 check questions and correct my answers.
5. Push back: if I ask for something that contradicts the blueprint, adds unjustified technology
   or skips evaluation, say so and propose the smallest alternative.

# Non-negotiable rules
- Use the blueprint's stack. Any new dependency needs a one-line reason tied to the JD or to a
  measured problem. Prefer the standard library and boring, well-documented tools.
- Simplest design that meets the requirement. No speculative abstraction. Measure before optimising.
- Code style: idiomatic, specific names, small functions. No explanatory comments by default; at
  most a short one-line comment above a major block when the reason isn't obvious. No docstrings,
  section banners, step-numbered comments, emojis or comments that restate the code.
- Every number in README, EVALUATION.md or my CV must come from a committed evaluation run.
  Never invent, estimate or quote "typical" results. Missing numbers stay as [measure].
- All telecom data is fictional. Never use a real operator's name, tariffs or logo. Test audio
  only from people who consented.
- No secrets in code. Permissions are enforced in code, not prompts. PII is never logged.
- If unsure about a library's API or version, say so and tell me how to check (official docs,
  installed version, a 5-line test). Never guess an API.
- Lead with the answer, then the reasoning. Keep it focused.

# Task brief format
## Task <ID>: <title>
Context: <module, why, blueprint section>
Goal: <one sentence>
In scope (files): <list> | Out of scope: <list>
Steps: <numbered, small>
Acceptance criteria: <testable bullets, including the tests to add>
Verify with: <exact commands>
Commit message: <type(scope): summary>
Style reminder: follow the comment policy in CLAUDE.md.

# Review checklist
Matches the brief | tests added and passing | comment policy followed | no dead code or speculative
abstraction | names and structure match the repo | errors, timeouts and fallbacks handled |
no secrets or PII | blocking inference kept off the event loop | docs updated |
unclear parts listed for my docs/STUDY.md.

# Study mode
After each milestone, walk me through its modules on request, then ask me 5 interview-style
questions per module and grade my answers honestly.
```

## Step 4: Set Up the Repository and Claude Code

1. **Repo basics.** `KsKarim7/Kotha` already exists and is cloned to `~/kotha` with a first commit. Add a Python `.gitignore` (it must cover `.env`, `.venv/`, model weights and audio files) and a licence for your code (MIT is common). Note in the README that MMS-TTS weights are CC-BY-NC, which is separate from your licence.
2. **Tooling.** Create the folder layout from the Build Guide, then run `uv init`, `uv python pin 3.11`, and add ruff, pytest and pre-commit.
3. **Protect `main`** in GitHub repository settings: require a pull request and passing CI before merging, even though you work alone. This is your safety net against the agent and against yourself.
4. **Add the spec to the repo:** put the blueprint files in `docs/blueprint/`.
5. **Open the project:** `cd ~/kotha && code .`, check the bottom-left says **WSL: Ubuntu**, and open the Claude Code panel.
6. **Create `CLAUDE.md`** at the repo root with the content below, either by pasting it yourself or by asking Claude Code to create it. Commit it. Claude Code reads it at the start of every session.

```markdown
# Kotha: guide for Claude Code

Bangla-English voice AI agent and call analytics for a fictional operator (Nodi Telecom).
Spec: docs/blueprint/. The current task brief defines the scope of any work.

## Stack (ask before adding anything)
Python 3.11, FastAPI + WebSocket, Silero VAD, faster-whisper, bge-m3 + pgvector + rank_bm25,
Gemini / Ollama, MMS-TTS, PostgreSQL 16, Asterisk AudioSocket, Prometheus, MLflow,
Docker Compose, pytest.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest -q`
- Lint and format check: `uv run ruff check . && uv run ruff format --check .`
- Stack: `docker compose up -d`

## How to work
- Do only what the current task brief asks. If something outside it looks wrong, list it at the
  end of your reply; don't fix it.
- Read the files you will touch and state a short plan first. Wait for my OK when the plan
  touches more than 3 files.
- Work in small steps and run the tests after each one. Never leave tests failing.
- Every behaviour change gets a test. Prefer small real fixtures (short WAVs in tests/fixtures)
  over heavy mocks.
- Work on the feature branch. Never commit to main, never force-push, never push without my OK.
- Small commits with Conventional Commit messages, e.g. `feat(vad): add hangover-based turn detector`.
- At the end of each task, list the parts most worth studying so I can add them to docs/STUDY.md.

## Code style
- Write code that reads like a careful engineer wrote it: idiomatic Python, specific names
  (turn_audio, not data or result), small functions.
- Comments: none by default. At most a short one-line comment above a major block, only when
  the reason isn't obvious from the code. No docstrings, section banners, step-numbered comments,
  emojis, or comments that restate what the code does.
- No commented-out code; no TODO without an issue number.
- No abstractions, base classes, wrappers or config options the current task doesn't need.
- Handle errors only where something can actually fail; no blanket try/except.
- Type hints everywhere; Pydantic models at service boundaries.
- Blocking model inference never runs on the event loop: use asyncio.to_thread or a worker.
- Every network call has a timeout. Retry only idempotent calls.
- Internal audio is float32 mono 16 kHz. Convert only at the edges (WebSocket: 16 kHz PCM16 LE;
  AudioSocket: 8 kHz PCM16 LE).

## Hard rules
- Never invent metrics. Results come only from scripts in eval/; I write EVALUATION.md.
- Never commit secrets, .env files, model weights, or audio of people without consent.
- Never log transcripts, phone numbers, PINs or NID numbers. Use the allow-list logger.
- Prices, balances and account actions come only from the BSS service, never from the LLM.
- If unsure how a library behaves, check the installed version and its docs, or write a tiny
  test. Don't guess.

## Docs to keep current
README.md, docs/ARCHITECTURE.md, EVALUATION.md (results only; I write the interpretation),
docs/adr/, CHANGELOG.md.
```

**Check:** ask Claude Code "What are the rules for working in this repo?" It should restate the key rules from the file. If it doesn't, the file isn't in the repo root or `.claude/` folder.

## Step 5: The Working Loop (repeat for every task)

1. **Plan** (Project, module chat): ask for the module's task list.
2. **Brief** (Project): ask for the Claude Code brief, then create a GitHub issue from it and a branch named after it (e.g. `m2-1-turn-detector`).
3. **Plan first** (Claude Code): paste the brief, ask for a plan only, read it, then approve.
4. **Implement** (Claude Code): it codes in small steps, following the comment policy, and runs the tests.
5. **Run it yourself:** tests, then the real thing (speak into the mic, call the extension, run the eval).
6. **Review** (Project): paste `git diff main` and the test output; fix blockers through Claude Code.
7. **Log for study:** add the parts you don't yet understand to `docs/STUDY.md` (file, lines, question).
8. **Merge:** open a pull request, let CI pass, merge, update `CHANGELOG.md`, close the issue, and **Sync** GitHub in the Project knowledge.
9. **End of module:** run the evaluation, write the interpretation in `EVALUATION.md` yourself, and add a decision record if you made a design choice.
10. **End of milestone:** run the study phase for its modules (above), using templates B and G.

### Prompt templates

**A. Module kickoff (Project)**

```text
Start module M<n>: <name>. Status: <what's merged so far>. Split it into tasks of at most
~150 changed lines each, in order, with acceptance criteria per task. Flag anything in the
blueprint you think is wrong or risky before we start.
```

**B. Concept lesson (Project)**

```text
Teach me the concepts behind <task ID or module>, in plain words with one small example each.
Then ask me 2 check questions and wait for my answers.
```

**C. Brief request (Project)**

```text
Write the Claude Code brief for task <ID> in our brief format. Keep scope to the listed files.
Include the tests to add and the exact commands to verify.
```

**D. Task prompt (Claude Code)**

```text
Here is the task brief: <paste>. First read the relevant files and give me a short plan.
Don't edit anything until I approve. Then implement step by step, running the tests after
each step, and finish with a summary of what changed and anything out of scope you noticed.
```

**E. Review request (Project)**

```text
Review this change for task <ID> against the brief and our checklist.
Diff: <paste git diff main>
Tests: <paste pytest output>
List blockers, should-fix and nits with exact lines. Then tell me the 2 lines most worth
understanding deeply, and why.
```

**F. Debug request (Project or Claude Code)**

```text
Problem: <what happens> | Expected: <what should happen> | Error/log: <paste> |
Already tried: <list>. Give me the most likely causes in order and a way to confirm each
before changing code.
```

**G. End-of-module quiz (Project)**

```text
Module M<n> is merged. Ask me 5 interview-style questions on it, one at a time, and grade
each answer honestly with what a strong answer would add.
```

## Step 6: Review Checklist Before Every Merge

Don't merge until every box applies. Copy the list into `.github/pull_request_template.md` so it appears on every pull request.

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
- [ ] The study phase for the previous milestone is finished before this milestone's last merge

## Step 7: Run It Like a Professional Project

**Project management on GitHub:**

- **Milestones** MS1–MS6 from the roadmap.
- **One issue per task brief**, labelled by module (`m2-vad`, `m3-asr`…).
- **A project board** with Backlog → In progress → Review → Done.
- **One-week sprints** with a written goal at the start and a short review at the end.
- **A release tag per milestone** (`v0.1.0` = MS1, `v0.2.0` = MS2…) with release notes and the demo video link.

**Documentation set:**

| File | What it holds | Updated | Written by |
| --- | --- | --- | --- |
| `README.md` | What Kotha is, demo video, quickstart, architecture picture, results summary, limitations, licence notes | Each milestone | You (agent may draft structure) |
| `docs/ARCHITECTURE.md` | Components, data flow, WebSocket and AudioSocket protocols | When design changes | Agent drafts, you correct |
| `EVALUATION.md` | Test sets, method, results tables, error analysis, limitations | After each eval run | Scripts write tables; you write interpretation |
| `docs/adr/NNNN-title.md` | One decision each: context, decision, alternatives, consequences | When you choose between options | You |
| `CHANGELOG.md` | User-visible changes per version (Keep a Changelog format) | Every merge | Either; you check |
| `docs/sprints/YYYY-WW.md` | Sprint goal, done, slipped, learned | Weekly | You |
| `docs/STUDY.md` | Questions about code you haven't understood yet, and their answers | During the build and study phase | You |
| `docs/runbook.md` | Start and stop, common failures and fixes, retention job, rotating keys | When ops change | Agent drafts, you test every command |
| `data/testsets/DATASHEET.md` | Who recorded, consent, transcription guideline, sizes, splits | When data changes | You |
| `MODEL_CARD.md` | Your fine-tuned ASR: data, metrics, limits, licence | MS6 | You |
| `.env.example`, `SECURITY.md`, `LICENSE` | Config template, privacy and security notes, licence | Once, then as needed | Either |

**Decision record template** (`docs/adr/0001-hybrid-retrieval.md`):

```markdown
# 0001: Use hybrid retrieval (dense + BM25) for the knowledge base
Status: accepted · Date: YYYY-MM-DD

## Context
What problem forced a choice, with any numbers you had.

## Decision
What you chose, in one or two sentences.

## Alternatives considered
Each option and why it lost.

## Consequences
What gets easier, what gets harder, what you'll watch.
```

## Keep It Simple Without Slowing It Down

The easy way and the fast, reliable way usually coincide if you follow these rules:

1. **Boring, documented tools.** One database, one framework, one service split (BSS). Less to break, less to explain.
2. **Make it work, then right, then fast.** Optimise only what a metric shows is slow.
3. **Load models once and warm them up.** The biggest latency win in the project, and trivial to do.
4. **Keep inference off the event loop.** One line (`asyncio.to_thread`) separates "works for one caller" from "works for many".
5. **Timeouts everywhere, one fallback path.** When anything fails, re-prompt or hand off. Never go silent.
6. **Configuration in one place:** environment variables read by one Pydantic settings class.
7. **Small fixtures, fast CI.** Keep the CI smoke set under about two minutes so you never skip it.
8. **Delete before you add.** Removing a feature flag beats adding another.

## Common Mistakes to Avoid

- **Letting the agent build a whole module at once.** The diff becomes too big to review properly, and bugs slip through.
- **One endless chat.** Context drifts; start a new chat per module with a short status.
- **Postponing the study phase to the very end.** Seven weeks of unfamiliar code at once won't stick; study each milestone as it lands.
- **"Features first, evaluation later."** Later never comes, and the numbers are the point.
- **Committing `.env` files, model weights or personal audio.** Set up `.gitignore` on day one.
- **Debugging dialogue through the microphone.** Use the text-mode harness.
- **Trusting library APIs from memory** (yours or the agent's). Pin versions and check the installed docs.
- **Forgetting to Sync GitHub** in the Project after merging, so reviews see stale code.
- **Fighting SIP over the internet.** Keep telephony on your own machine or LAN.

## Sources

- Claude Code install and setup: https://code.claude.com/docs/en/setup
- Claude Code installation notes (Windows and WSL): https://code.claude.com/docs/en/install.md
- How Claude remembers your project (CLAUDE.md): https://code.claude.com/docs/en/memory
- Using the GitHub integration in Claude projects: https://support.claude.com/en/articles/10167454-using-the-github-integration
- WSL file system guidance: https://learn.microsoft.com/en-us/windows/wsl/filesystems
