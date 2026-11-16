# 0002: Consent, storage, retention and publication of test-set recordings
Status: accepted · Date: 2026-11-16

## Context
M1 builds an ASR test set from people who agree to be recorded. Voice recordings are personal
data: a voice can identify a person. The repository is public, and the manifest that M1-6
commits holds the reference text of every clip. The project needs fixed rules for consent,
storage, access, retention, withdrawal and what may be published, before anyone other than
Khaled records.

## Decision
- **Consent:** every participant is 18 or older and signs the bilingual (Bangla and English)
  consent form before recording. It has two choices: (1) taking part (required); (2) publishing
  the text of what they said under their speaker code (optional).
- **Recordings are never published** and never committed. There is no consent choice for
  publishing audio. Exception: fixtures of Khaled's own voice in `tests/fixtures/`, each with his
  explicit OK.
- **Storage:** audio lives at `G:\Kotha\data\testsets\asr` (WSL `/mnt/g/Kotha/data/testsets/asr`),
  with one backup copy on a drive kept offline. No cloud service. Only Khaled has access.
- **Identity:** speakers are `spk01`, `spk02`, ... in every file. The name-to-code list and the
  signed forms are kept offline together and never enter the repository.
- **Use:** only for evaluating speech recognition in Kotha. Not for training models (blueprint
  M13: never train on the test set), not for voice biometrics (M14), not shared. Any other use
  needs new consent.
- **Publication of text:** the public manifest `data/testsets/asr_test.csv` holds only the rows of
  speakers who said Yes to choice 2. Rows of speakers who said No go in a private manifest next to
  the audio on G:, used for evaluation runs but never committed. EVALUATION.md states how many
  rows are private, so a reader knows the public manifest alone can't reproduce the numbers.
- **Retention:** audio, the private manifest, the forms and the name list are deleted on
  31 December 2027 unless the participant agrees to a longer period in writing.
- **Withdrawal:** on request, within 7 days, the participant's audio is deleted and their rows are
  removed from both manifests in a new commit. Text already published stays in git history; the
  consent form says so. Results already reported are recomputed at the next evaluation run.
- **Outputs:** evaluation outputs that contain transcripts (per-utterance CSVs) are written
  outside the repository (enforced by `eval/wer.py`) and follow the same retention.

## Alternatives considered
- **A consent choice for publishing audio:** dropped, because the project never publishes test-set
  audio; offering the choice would promise something the project won't do.
- **Excluding speakers who say No to publishing text:** simpler, but loses data from a small
  speaker pool (at least 5 speakers); a private manifest keeps their clips.
- **Cloud storage:** easier backup, but puts voice data with a third party; not needed for about
  150 short clips.
- **Rewriting git history on withdrawal:** possible, but it breaks every clone and existing links;
  the consent form states the limit instead.

## Consequences
- The manifest validator (M1-4) and `eval/wer.py` must accept the private manifest as well;
  `eval/wer.py --manifest` already takes any path.
- Reported numbers may depend on private rows; EVALUATION.md must say how many.
- Khaled keeps a short deletion checklist for 31 December 2027 and for withdrawals.
- This is a project rule, not legal advice; if the project is ever used beyond learning, the
  rules must be checked against Bangladesh's data-protection law at that time.
