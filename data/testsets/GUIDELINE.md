# Transcription guideline for the Kotha ASR test set

This guideline says how to write the reference text of every clip in `asr_test.csv`. The
reference lint checks rules 1-5.

**The rule behind all the others: write what the speaker actually said, in the form that
`kotha.text.normalize.normalize` already produces.** Then WER measures recognition errors, not
spelling or formatting differences.

## 1. Form of every reference

- No punctuation at all: no danda (।), comma, question mark, hyphen or quotes.
- English words in lowercase Latin letters.
- Digits are ASCII (0-9), never Bangla digits (০-৯).
- Words separated by single spaces; no leading or trailing space.
- No invisible characters (ZWJ, ZWNJ). Type normally with a Unicode Bangla keyboard (Avro or
  similar); the lint flags anything else.
- Standard Bangla spelling (Bangla Academy). For a dialect word with no standard spelling, write
  it as it sounds.

## 2. Bangla and English words

- A Bangla word is written in Bangla script.
- An English word is written in Latin script exactly as spoken, even when Bangla speakers use it
  every day: `আমার mobile balance কত`, not `আমার মোবাইল ব্যালেন্স কত`.
- Every English word in `data/lexicon/loanwords.csv` (the `latin` column) is always written in
  Latin. If you meet a new English word, write it in Latin and add a row to that file with its
  usual Bangla spelling(s), so the loanword-agnostic score (`lexicon_wer`) can map it.
- Do not add ordinary Bangla words that also stand for an English one (for example নম্বর for
  "number"): they stay Bangla.
- "সিমকার্ড" is two English words: `sim card`.

## 3. Numbers, units and codes

- Any spoken number is written in digits, whatever language it was said in: "তিরিশ" or "thirty"
  -> `30`; "দুইশো নিরানব্বই টাকা" -> `299 টাকা`.
- A unit is its own word: "thirty GB" -> `30 gb`; "500 MB" -> `500 mb`.
- A package code is written letter group and number apart, as spoken: "NT thirty GB" -> `nt 30 gb`.
- A phone number is one run of digits with no spaces: `01700000000`. Use only the fictional
  numbers from the prompt sheet.
- Never write the taka sign (৳); write the word `টাকা` if it was said.

## 4. Classifiers and suffixes (টা, টি, খানা, গুলো, এর ...)

- After a Bangla word, attach them as in normal spelling: `নম্বরটা`, `টাকার`.
- After an English word written in Latin, write them as a separate word: `package টা`,
  `sim এর`. A Latin word can't carry a Bangla suffix, so this keeps both parts readable.

## 5. What to leave out

- Fillers and hesitation sounds: আ, উম, এ, uh, um, hmm.
- Cut-off fragments of a word the speaker abandoned ("রি-... রিচার্জ" -> `recharge`).
- Keep complete words even when repeated ("আমার আমার ব্যালেন্স" -> `আমার আমার balance`).
- Never write background speech, noise labels or comments.

## 6. When a clip can't be transcribed

- Listen up to three times. If you still can't tell what was said, mark the clip for
  re-recording or drop it. Never guess.
- If a spontaneous clip contains a real name, phone number, address or any other personal
  detail, drop the whole clip. Don't redact it.

## 7. The `lang` column

- `bn`: every word is Bangla.
- `en`: every word is English.
- `mix`: both, in any proportion.
- Decide from the reference text, after applying rules 2-4 (digits don't count either way).

## 8. Checking the references

- Every reference passes the lint before it is used.
- A second transcriber (or Khaled again, at least a week later and without looking) transcribes
  10% of the clips. The agreement between the two passes goes into the datasheet, and every
  disagreement is settled in this guideline, so the rule is written down once.

## Examples

| What was said | Reference | lang |
| --- | --- | --- |
| আমার মোবাইল ব্যালেন্স কত, আর থার্টি জিবি ইন্টারনেট প্যাকেজটা অ্যাক্টিভেট করতে চাই। | আমার mobile balance কত আর 30 gb internet package টা activate করতে চাই | mix |
| আমার সিমকার্ডটা হারিয়ে গেছে | আমার sim card টা হারিয়ে গেছে | mix |
| আমার রিচার্জ হয়নি | আমার recharge হয়নি | mix |
| Check my data balance, please. | check my data balance please | en |
| দুইশো নিরানব্বই টাকার প্যাকেজটা কত দিনের? | 299 টাকার package টা কত দিনের | mix |
| আমি কথা বলতে চাই, উম, একজন মানুষের সাথে | আমি কথা বলতে চাই একজন মানুষের সাথে | bn |
