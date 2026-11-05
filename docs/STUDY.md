# Study log

Questions about code I haven't understood yet. Answer each one in my own words once I have studied it.

| File | Lines | Question | Answer |
| --- | --- | --- | --- |
| `pyproject.toml` | 7-12 | What is the difference between `[dependency-groups]` dev and `[project].dependencies`, and why do the dev tools belong in the group? | |
| `pyproject.toml` | 14-16 | With no build backend, `kotha` is never installed; how does pytest still import it through `pythonpath = ["services/api"]`? | |
| `.gitignore` | 29-35 | Why does `!tests/fixtures/**/*.wav` make fixture audio trackable again, and why would it stop working if `tests/fixtures/` itself were ignored? | |
| `.gitignore` | all | How does `git check-ignore -v <path>` show which rule matched a path, and how do I use it to test a new pattern? | |
| `.python-version`, `pyproject.toml`, `uv.lock` | 1; 4; all | How do `.python-version`, `requires-python` and `uv.lock` together decide which interpreter and package versions uv installs? | |
| `pyproject.toml` | 18-20 | How does `ruff check --show-settings` show which files ruff selects, and why does its default include list contain `*.md`? | |
| `.pre-commit-config.yaml` | 16-27 | Why does a `repo: local` hook with `language: system` and `uv run` use exactly the ruff version in `uv.lock`, and what does that cost? | |
| `.pre-commit-config.yaml` | 20, 25 | Why does ruff check a blueprint file passed by name unless `--force-exclude` is set? | |
| `.pre-commit-config.yaml` | 22, 27 | How does pre-commit pick files with `types_or` and identify tags, and why does the `markdown` tag make ruff-format run on `.md` files? | |
| `.pre-commit-config.yaml` | all | Why does `pre-commit run --all-files` skip a new file until it is staged? | |
| `.pre-commit-config.yaml` | 7-14 | What is the difference between hooks that fix files and hooks that only check, and what should I do after a fixing hook fails? | |
| `.github/workflows/ci.yml` | 8-9 | What does `permissions: contents: read` restrict, and what could the workflow do without it? | |
| `.github/workflows/ci.yml` | 3-6 | Why do `pull_request` plus `push` to `main` cover every change without running CI twice for one branch push? | |
| GitHub branch protection for `main` | n/a | What do `strict: true` and the `app_id` on the required `checks` status check each do? | |
| `.pre-commit-config.yaml` | 14 | Why does `--enforce-all` matter only for files that are already committed? | |
