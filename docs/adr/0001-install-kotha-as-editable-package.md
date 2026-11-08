# 0001: Install kotha as an editable package
Status: accepted · Date: 2026-11-08

## Context
`kotha` lives at `services/api/kotha`, and the project had no build backend, so uv never
installed it. Only pytest could import it, through `pythonpath = ["services/api"]`. Any script
run with `uv run python` needed `PYTHONPATH=services/api`, and the next tasks (the spectrogram
script, the manifest validator, the M3 eval CLI) are all scripts.

## Decision
Use uv's own build backend, `uv_build`, with `module-name = "kotha"` and
`module-root = "services/api"`, so `uv sync` installs `kotha` in editable mode.

## Alternatives considered
- `PYTHONPATH` in every command and CI step: easy to forget, and repeated everywhere.
- hatchling or setuptools: another build dependency and its configuration, with no gain over
  the backend that ships with uv.
- A uv workspace with per-service projects: deferred to M5, when there is more than one service.

## Consequences
- `uv run python` and pytest both import the installed package; code edits apply without
  reinstalling, and pytest no longer needs `pythonpath`.
- `uv sync` and `uv build` always use the `uv_build` bundled with the running uv. When uv is
  upgraded past the `<0.14.0` bound they keep working and print a warning; we bump the bound
  deliberately when we upgrade uv. The bound only selects the backend version for isolated
  builds, such as `uv build --force-pep517` or pip.
- Ruff's `src = ["services/api"]` is still needed: ruff decides which imports are first-party
  from source paths, not from installed packages.
