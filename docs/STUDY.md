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
| `services/api/kotha/audio/convert.py` | 11-18 | Why does scaling by 32768 in both directions, with rounding and clipping, make int16 -> float32 -> int16 exact, when the blueprint's 32767 and truncation do not? | |
| `services/api/kotha/audio/convert.py` | 12, 18 | What does the `<` in `dtype="<i2"` mean, and what would go wrong on the wire with a native-order `int16` on a big-endian machine? | |
| `services/api/kotha/audio/convert.py` | 12 | Why does `np.frombuffer` raise `ValueError` on an odd number of bytes, and why is that enough validation here? | |
| `services/api/kotha/audio/convert.py` | 21-26 | How does `resample_poly` turn 44.1 kHz into 16 kHz with up=160, down=441, and what does its anti-aliasing filter do at the edges of the signal? | |
| `services/api/kotha/audio/convert.py` | 29-32 | What do `always_2d=True` and `dtype="float32"` change in what `soundfile.read` returns, and how does soundfile scale PCM_16 to float? | |
| `tests/test_audio_convert.py` | 70-79 | Why does a 1 s signal at 16 kHz give FFT bins exactly 1 Hz apart, and why are 100 samples at each edge ignored when measuring RMS? | |
| `pyproject.toml`, `tests/test_audio_convert.py` | 20, 22-24; 1-12 | Why does ruff sort `kotha` with the third-party imports, and how would the ruff `src` setting change that? | |
| `services/api/kotha/audio/telephony.py` | 12 | Why does a sign plus a 7-bit magnitude give an exact zero level and odd symmetry, when the blueprint's `round((y + 1) / 2 * 255)` gives neither? | |
| `services/api/kotha/audio/telephony.py` | 11-13 | Why does mu-law keep the SNR of a sine roughly constant from 0.01 to 0.9 amplitude, when uniform 8-bit quantisation loses about 6 dB per halving? | |
| `services/api/kotha/audio/telephony.py` | 12 | Why is the round trip idempotent, and why does the test still need a tolerance instead of exact equality? | |
| `services/api/kotha/audio/telephony.py` | 19, 21 | Why does resampling to 8 kHz remove a 6 kHz tone, and what does the `resample_poly` low-pass filter do near 4 kHz? | |
| `services/api/kotha/audio/telephony.py` | 19-22 | Why does a 16,001-sample input come back as 16,002 samples before the trim, and why trim the end rather than the start? | |
| `tests/test_audio_telephony.py` | 51-60 | Why are 200 samples ignored at each edge when measuring RMS after two resampling passes? | |
| `pyproject.toml` | 24 | What does ruff's `src` setting change, and why does setting it replace the default `[".", "src"]` rather than add to it? | |
