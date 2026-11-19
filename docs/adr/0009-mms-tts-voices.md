# 0009: MMS-TTS voices for Bangla and English, fed only text inside each voice's vocabulary
Status: proposed · Date: 2026-11-19

## Context
M7 needs a Bangla voice and an English voice that run locally on the CPU (ADR 0003). The blueprint
names `facebook/mms-tts-ben` and "an English voice". Facts below were checked on the date above;
numbered sources are under References.

- **Models.** `facebook/mms-tts-ben` and `facebook/mms-tts-eng` are VITS models
  (`model_type: vits`) for the `text-to-speech` pipeline [1][2]. Both are licensed
  **CC-BY-NC 4.0**: non-commercial use only. That fits a portfolio project and rules out
  commercial use without other voices [1][2][3].
- **Revisions and sizes.** Pinned commits: `mms-tts-ben` at
  `0da99de6074c8829121cdabfbdba423af18e8e56`, `mms-tts-eng` at
  `c71de0fe7204c83f1c10820a7d696d0b450048ba`. `model.safetensors` is 145,255,160 bytes
  (138.5 MiB) and 145,227,512 bytes (138.5 MiB). Each repo also has a `pytorch_model.bin` of the
  same weights, which we do not download [1][2].
- **Sampling rate.** Both `config.json` files say `sampling_rate: 16000` [1][2], which is the
  internal audio format, so no resampling is needed inside Kotha. Phone audio needs 8 kHz.
- **Tokenizer.** Both use `VitsTokenizer` with `normalize: true`, `is_uroman: false` and
  `phonemize: false` [1][2]. In the installed transformers 5.23.0, `normalize` lowercases the text
  and then **silently drops every character that is not in `vocab.json`** [4]. Nothing is logged
  and no unknown token is used.
- **Vocabularies.** `mms-tts-ben` has the Bengali letters and signs, the ASCII digits 0-9, space,
  `'`, `-`, `_` and `—`. It has no Latin letters, no Bengali digits, no `।` and no precomposed
  য়, ড় or ঢ়, only the nukta ় [1]. `mms-tts-eng` has a-z, the ASCII digits 0-6 (not 7, 8 or 9),
  space, `'`, `-`, `_` and `–` [2]. NFC writes য়, ড় and ঢ় as base letter plus nukta (checked
  with `unicodedata` in Python 3.11), which is inside the Bangla vocabulary.
- **Blank token.** With `add_blank: true` the tokenizer puts id 0 between every pair of characters
  [4], and id 0 is also a vocabulary character: `6` in Bangla and `k` in English (the
  `pad_token`). Checked with the installed tokenizers: `the pack` encodes its `k` as 0, and
  `দিন 6 দিন` encodes `6` as 0. So `6` and `k` reach the model only as blanks. Whisper still heard
  "pack" and "Ask" in the English clips, so the English model may cope with this; it is not
  confirmed by listening.

### What the probe showed
`eval/tts_probe.py` synthesised 15 fixed sentences (9 bn, 6 en) at seed 0 on the CPU and printed
what each tokenizer keeps. As a rough hint at what the audio sounds like, it transcribed each clip
back with faster-whisper. The probe takes one `--asr-model` for all clips, so it ran twice; the
table takes the Bangla "heard" column from the run with `models/bn-tugstugi-medium-ct2` and the
English one from the run with `medium` (the runbook has both commands). Nobody has listened to the
clips yet, and an ASR transcript is not a pronunciation score; the Bangla ASR also "repairs" words
from its language model.

| Input | What the tokenizer keeps | What ASR heard |
| --- | --- | --- |
| `প্যাক NT-30GB সাত দিন চলবে।` | `প্যাক -30 সাত দিন চলবে` | `সাত দিন চলবে` |
| `SIM বদলাতে NID কার্ড লাগবে।` | `বদলাতে  কার্ড লাগবে` | `ভদ্ররাতে কার্ড ও লাগবে` |
| `প্যাকের মেয়াদ 30 দিন।` | unchanged except `।` | `জাকের রিয়াল দিন` |
| `প্যাকের মেয়াদ ৩০ দিন।` | `প্যাকের মেয়াদ  দিন` | `প্রাকের মেয়াদ দিন` |
| `প্যাকের মেয়াদ ত্রিশ দিন।` | unchanged except `।` | `তাকের মেয়াদ ত্রিশ দিন` |
| `মেয়াদ` with precomposed য় (U+09DF) | `মোদ` | (ASR wrote `মেয়াদ`) |
| `প্যাক দেখতে *747# ডায়াল করুন।` | `প্যাক দেখতে 747 ডায়াল করুন` | `ব্রাক দেখতে দেয়াল করুন` |
| `The pack lasts 30 days.` | `the pack lasts 30 days` | `The pact lasts. He days.` |
| `The pack gives 8 GB for 7 days.` | `the pack gives  gb for  days` | `The pack gives bid four days.` |
| `The pack gives eight gigabytes for seven days.` | unchanged except `.` | `The pack gives eight database for seven days.` |
| `Dial *747# to see the packs.` | `dial 4 to see the packs` | `DOA to see the packs.` |
| `Ask at any নদী service centre.` | `ask at any  service centre` | `Ask at any service center.` |

So, on these sentences: Latin words in Bangla text and Bengali digits are not spoken at all;
ASCII digits are kept but did not come back as numbers in either language; numbers written as
words came back as words; the tokenizer removes a precomposed য় (`মেয়াদ` becomes `মোদ`), which
listening has not yet confirmed in the audio; codes like `*747#` lose their symbols and, in English,
their 7s. Loanwords written in Bangla script (`এনআইডি`, `অ্যাপ`) came back
garbled, so they need the listening test before we rely on them.

## Decision
Use `facebook/mms-tts-ben` for Bangla replies and `facebook/mms-tts-eng` for English replies,
through transformers' `VitsModel`, at the pinned revisions above, from `model.safetensors` only,
loaded offline (`local_files_only=True`, `DISABLE_SAFETENSORS_CONVERSION=true`), on the CPU, with a
fixed seed. Each voice gets only text that is entirely inside its own vocabulary: the speakable-text
step (M7-2) writes numbers as words, codes symbol by symbol in words, Latin terms in Bangla script
for the Bangla voice and the brand in Latin for the English voice, and applies NFC. No digit reaches
either voice, which also keeps the Bangla `6` blank out of the input. The TTS service (M7-3) treats
any character its tokenizer would drop as a bug that tests catch, not something to pass to the
model, with one known exception: the English `k`, which cannot be avoided in English text. It is
allowed and goes on the listening sheet.

`kotha.tts.speakable` (M7-2) applies these rules. A number of one or two digits, a number with
grouping commas or a decimal point, and a number followed by a unit (GB, MB, days, দিন, ঘণ্টা, a
count suffix such as টা) is read as a number. Any other run of three or more digits, and any run
with a leading zero, is read digit by digit, because in this domain it is almost always a short
code, an SMS number or a phone number (`747`, `7470`, `01700000009`). The cost: a year (`2025`),
a pack name such as `NV-100` and a bare table value such as `300` are also read digit by digit.
Clock times (`10:00`, `৯:৩০`) are read as times. An unknown Latin word in a Bangla reply is spelled
with letter names; a known one comes from a small loanword map. Accents are folded (`café` to
`cafe`). The last step keeps only characters in the voice's vocabulary, so nothing reaches a voice
that its tokenizer would drop. In an English reply this removes any Bangla word except the brand
names, which is safe because the answer pipeline rejects an English reply that contains Bengali
letters (`in_language`). Ranges (`5-10`), signs (`-5`) and fractions (`1/2`) lose their symbol;
none occurs in the knowledge base.

The status stays "proposed" until the listening sheet (M7-3) is rated and RTF comes from a
committed run.

## Alternatives considered
- **Cloud bn-BD voices** (named in the blueprint): likely better quality, but the caller's reply
  text and audio leave the machine, which breaks on-prem mode. Not tried.
- **Passing raw text and relying on the model.** The tokenizer drops what it does not know, so
  codes, data sizes and English terms would vanish without an error.
- **uroman romanisation** (supported by the tokenizer for `is_uroman` models): the Bangla model
  was trained on Bangla script (`is_uroman: false`), so it does not help here.
- **Other open voices** were not evaluated in this task; MMS was the blueprint's choice and keeps
  one model class and one code path for both languages.

## Consequences
- The voices are non-commercial. The README says so; commercial use needs other voices.
- The quality of numbers and English terms rests on the speakable-text step, so that step needs
  exhaustive unit tests, and the listening sheet must include data sizes, codes and English terms.
- The dialogue keeps digits (the verifier in `kotha.rag.citations` checks them) and only the TTS
  input turns them into words.
- About 277 MiB of weights for both voices; they load once per process.
- To watch: Bangla loanword pronunciation, English words with `k` (sent as blanks), and how often a
  reply needs the other language's voice for a single word.

## References
1. Hugging Face model repo `facebook/mms-tts-ben` at `0da99de6`, read through `huggingface_hub`
   `HfApi.model_info` and the downloaded `config.json`, `tokenizer_config.json` and `vocab.json`,
   2026-11-19: https://huggingface.co/facebook/mms-tts-ben
2. Hugging Face model repo `facebook/mms-tts-eng` at `c71de0fe`, read the same way:
   https://huggingface.co/facebook/mms-tts-eng
3. Creative Commons, Attribution-NonCommercial 4.0 International:
   https://creativecommons.org/licenses/by-nc/4.0/
4. transformers 5.23.0 as installed, `transformers/models/vits/tokenization_vits.py`:
   `VitsTokenizer.normalize_text` and `prepare_for_tokenization`.
