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
| `pyproject.toml` | 11-13 | What does `[build-system]` tell a frontend such as uv or pip, and what does the `uv_build` backend produce from this repo? | |
| `pyproject.toml` | 22-24 | How do `module-name` and `module-root` map `services/api/kotha` to the import name `kotha`, and why must `services/api/kotha/__init__.py` exist for `uv_build`? | |
| `.venv/lib/python3.11/site-packages/kotha.pth` | 1 | How does a one-line `.pth` file make `import kotha` resolve to the repo, and why do code edits apply without reinstalling? | |
| `uv.lock` | 84-86 | What is the difference between `source = { virtual = "." }` and `source = { editable = "." }`, and what decides which one uv writes? | |
| `pyproject.toml` | 12 | Why do `uv sync` and `uv build` use the bundled `uv_build` even when the `<0.14.0` bound excludes the running uv, and when is the bound honoured (`--force-pep517`, pip)? | |
| `pyproject.toml` | 31 | Why does ruff still need `src = ["services/api"]` to treat `kotha` as first-party after it is installed? | |
| `eval/plot_spectrograms.py` | 17-22 | How do `NFFT=512` and `noverlap=384` trade time resolution against frequency resolution at 16 kHz, and what would 256 or 1024 change in the figure? | |
| `eval/plot_spectrograms.py` | 24-30 | Why must both panels share `vmin`/`vmax` taken from the original, and why is `spectrum` linear power while the image is `10 * log10` of it? | |
| `eval/plot_spectrograms.py` | 16-30 | Why do exact-zero frames from the mu-law zero level become -inf dB, why did matplotlib draw them white, and how does `with_extremes(bad=...)` fix that? | |
| `eval/plot_spectrograms.py` | 14, 42 | How does `Figure(...).savefig` render a PNG without pyplot, and which canvas and backend does it switch to? | |
| `docs/img/spectrogram_telephony.png` | n/a | Why does faint energy remain between 4 and about 4.5 kHz after `simulate_telephony` (the `resample_poly` filter's transition band), and why is it dark above that? | |
| `pyproject.toml`, `eval/` | 29, 33; all | Why does `eval/` have no `__init__.py`, how does `pythonpath = ["eval"]` let tests import `plot_spectrograms`, and why would a package named `eval` be a problem? | |
| `eval/plot_spectrograms.py` | 45-50 | Why does `main(argv: list[str] \| None = None)` pass `argv` to `parse_args`, and how does that make the CLI testable without a subprocess? | |
| `pyproject.toml` | 5-11, 30-31 | Why does `[tool.uv.sources]` pin torch to the CPU index only when torch is a direct dependency, and what happened when it was only pulled in by silero-vad? | |
| `pyproject.toml` | 25-28 | What does `explicit = true` on the `pytorch-cpu` index prevent, and why would leaving it out risk other packages resolving from the PyTorch index? | |
| `uv.lock` | 585-624 | Why are there two torch entries, `2.15.0+cpu` for non-macOS and `2.15.0` for macOS, and what is a `+cpu` local version? | |
| `services/api/kotha/vad/silero.py` | 12 | Where do Silero's weights come from (`silero_vad/data/silero_vad.jit` through `importlib.resources`), and why does shipping them inside the package matter for on-prem mode? | |
| `services/api/kotha/vad/silero.py` | 21, 24 | What state does the Silero model keep between windows, why is one instance per call needed, and what does `reset_states()` clear? | |
| `services/api/kotha/vad/silero.py` | 20-21 | What does `torch.no_grad()` turn off, and why do silero-vad's own `get_speech_timestamps` and `VADIterator` use it? | |
| `services/api/kotha/vad/silero.py` | 15-19 | Why must the wrapper check dtype and shape itself, when the model rejects 511 samples and float64 with a generic error but accepts a `(1, 512)` array? | |
| `tests/test_vad_silero.py` | 55-66 | How does feeding two instances interleaved windows show that they don't share state, and why compare against a fresh, reset instance? | |
| `services/api/kotha/vad/turns.py` | 53-54 | Why does `-(-ms * SAMPLE_RATE_HZ // (1000 * WINDOW_SAMPLES))` round up, and why is ceil ("at least" the configured time) the right choice over floor or round for 250, 700 and 300 ms? | |
| `services/api/kotha/vad/turns.py` | 80-86 | Why is `SpeechStarted` emitted only once minimum speech is confirmed, and what does that guarantee M8 about pairing every start with exactly one end? | |
| `services/api/kotha/vad/turns.py` | 70-81 | How do the two thresholds form a hysteresis band, so 0.4 can continue a turn but never start one? | |
| `services/api/kotha/vad/turns.py` | 80-91 | Why do speech windows count towards minimum speech even when they are not consecutive, while the silence run resets on any speech window? | |
| `services/api/kotha/vad/turns.py` | 83, 95 | Why does `end_sample` mark the last speech window while the audio runs on through the hangover tail, and what does each give ASR and latency measurement? | |
| `services/api/kotha/vad/turns.py` | 43, 64, 73 | How does the rolling pre-roll buffer work, and why can the next turn's pre-roll overlap the previous turn's hangover tail (windows 45-46 in `test_two_utterances_have_separate_audio`)? | |
| `services/api/kotha/vad/turns.py` | 60 | Why does `push` store a float32 copy of each window instead of the caller's array, and what would go wrong if M8 reused one buffer? | |
| `services/api/kotha/vad/turns.py`, `tests/test_vad_turns.py` | 24-31; 8-15 | Why does the detector take a probability callable instead of a `SileroVad`, and how does that let the tests script probabilities without torch? | |
| `services/api/kotha/vad/__init__.py` | 1 | Why did `WINDOW_SAMPLES` move out of `silero.py`, and how does importing a constant from a module that imports torch make every importer pay for torch? | |
| `services/api/kotha/vad/turns.py` | 92-101 | Why close an open turn with `flush()` instead of padding the file with zeros, and how would padding tie the runner to the hangover setting? | |
| `services/api/kotha/vad/turns.py` | 94-101 | Why does `flush()` leave the sample counter and pre-roll alone, and what does `test_pushing_after_a_flush_continues_the_stream` show about the next turn's pre-roll? | |
| `services/api/kotha/vad/turns.py`, `tests/test_vad_turns.py` | 94-101; 206-213 | Why does a flushed turn's `end_sample` stay at the last speech window while its audio runs through the last pushed window? | |
| `services/api/kotha/vad/turns.py` | 89-92 | How does `_advance` reuse `flush()` when the hangover completes, and why do the 23 earlier tests prove the behaviour did not change? | |
| `eval/vad_turns.py` | 12-23 | Why does `find_turns` drop a trailing partial window instead of padding it, and how does `reshape` give row views without copying? | |
| `eval/vad_turns.py`, `tests/test_vad_turns_runner.py` | 31; 45-56 | Why does each file (and each test run) need a fresh `SileroVad` and `TurnDetector`, and what would leak between files if one were reused? | |
| `tests/test_vad_turns_runner.py` | 59-75 | How does pytest's `capsys` fixture capture what `main()` prints, and why test `main(argv)` in-process rather than through a subprocess? | |
| `pyproject.toml`, `uv.lock` | 6; all | Why does faster-whisper run without torch (CTranslate2), and why does it still bring in onnxruntime and av? | |
| `services/api/kotha/asr/whisper.py` | 27 | What do `device="cpu"` and `compute_type="int8"` do in CTranslate2, and what does int8 quantisation trade for speed and memory? | |
| `services/api/kotha/asr/whisper.py` | 41-50 | Why does `transcribe` return a lazy generator, why must `list()` consume it before the call is finished, and why does language detection still happen eagerly? | |
| `services/api/kotha/asr/whisper.py` | 46 | Why turn off `condition_on_previous_text` for short turns, and how can conditioning carry an error from one segment into the next? | |
| `services/api/kotha/asr/whisper.py` | 48 | Why disable faster-whisper's own Silero VAD (`vad_filter=False`) when the turn detector has already cut the audio? | |
| `services/api/kotha/asr/whisper.py` | 51-58 | What do `avg_logprob` and `no_speech_prob` measure, why take the worst segment for each, and why does an empty result report -inf and 1.0? | |
| `services/api/kotha/asr/whisper.py` | 41-48 | How does faster-whisper's default temperature fallback (retrying with sampling when `avg_logprob` < -1.0 or the compression ratio is high) make two runs on the same audio differ? | |
| `tests/test_asr_whisper.py` | 49-52 | How does `monkeypatch.setattr("kotha.asr.whisper.WhisperModel", ...)` replace the name the wrapper looks up, and why would patching `faster_whisper.WhisperModel` not work? | |
| `docs/adr/0004-local-gpu-for-asr.md` | 6-11 | Why does CTranslate2 load `libcublas.so.12` by name at runtime (no `NEEDED` entry in `readelf -d`), and why must `LD_LIBRARY_PATH` be set before Python starts rather than from inside it? | |
| `pyproject.toml` | 25-30 | How do `[dependency-groups]` and `[tool.uv] default-groups` interact, and why would a plain `uv sync` remove a group that is not a default? | |
| `.github/workflows/ci.yml` | 16-17 | Why is `UV_NO_GROUP: gpu` set for the whole job and not only on `uv sync`, given that `uv run` also syncs default groups (inexactly)? | |
| `pyproject.toml`, `uv.lock` | 26; all | What does the `sys_platform == 'linux'` marker do in a universal lock, and why does `nvidia-cuda-nvrtc-cu12` still come in with cuBLAS? | |
| `docs/runbook.md` | 13-14 | Which startup files do a login shell, the VS Code terminal (`--init-file` shell integration) and VS Code's extension host read, and why does the export go after `~/.bashrc`'s interactive check? | |
| `eval/asr_device_check.py` | 43-78 | What do the compute types float16, int8_float16 and int8 store and compute in, and why can all three run on compute capability 7.5? | |
| `eval/asr_device_check.py` | 52-64, 98 | Why read GPU memory right after loading, and how do `del`, returning from the function and `gc.collect()` make sure one model is freed before the next loads? | |
| `eval/asr_device_check.py` | 55-62 | Why do the transcribe times here mostly measure the temperature fallback and repetition loops (compression ratio about 20 at temperature 0), not the compute type? | |
| `eval/asr_decoding_probe.py` | 20-21 | How does faster-whisper's temperature fallback work (retry at 0.2, 0.4, ... 1.0 when the compression ratio is above 2.4 or `avg_logprob` is below -1.0), and why does it make a looping clip slow and unrepeatable? | |
| `eval/asr_decoding_probe.py` | 91, 110 | Why is the compression ratio (gzip ratio of the text) a cheap loop detector, and what ratio does normal speech have compared with a repetition loop? | |
| `eval/asr_decoding_probe.py` | 21-22 | What is the difference between greedy decoding (`beam_size=1`) and beam search (`beam_size=5`), and why can beam search avoid a loop that greedy decoding falls into? | |
| `eval/asr_decoding_probe.py` | 23-24 | How do `repetition_penalty` and `no_repeat_ngram_size` differ in what they forbid or discourage, and what can each break in normal text? | |
| `eval/asr_decoding_probe.py` | 25-26 | How does `max_new_tokens` bound decoding time, why must it stay at or below 448 minus the prompt, and why does hitting it cut the output rather than fix the loop? | |
| `eval/asr_decoding_probe.py` | 77-80 | Why can GPU float16 (and sampling at temperature > 0 without a seed) give different results on the same audio, and what does that mean for comparing runs? | |
