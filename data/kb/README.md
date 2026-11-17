# Knowledge base

The Nodi Telecom knowledge base that Kotha retrieves from (ADR 0006). Everything here is fictional:
Nodi Telecom is not a real operator, and its packs, codes and numbers are invented for the demo.

## Format

- One file per document per language: `<doc_id>.<lang>.md`, where `doc_id` is lowercase
  kebab-case and `lang` is `bn` or `en`. Every document exists in both languages.
- The file starts with front matter between two `---` lines, one `key: value` per line, with
  exactly the keys `doc_id`, `lang`, `title` and `notice`. The notice is always
  `Fictional content for the Kotha demo. Nodi Telecom is not a real operator.`
- Then a Markdown body: one `# ` title and `## ` sections of about 150–300 words, because the
  chunker splits by heading. The bn and en files of a document have the same sections in the same
  order and say the same facts.

`kotha.kb.documents.read_document` reads a file and refuses one that breaks these rules, and
`tests/test_kb_content.py` checks the whole folder.

## Content rules

- No prices, fees, balances or amounts of money: no `৳`, and no digits next to টাকা, taka, tk or
  bdt. Prices come only from the BSS; the text sends customers to the app or customer care.
- Product facts (package codes, data, minutes, validity) come only from the package catalogue in
  `tests/test_kb_content.py`.
- No real operator, brand, wallet or app names, no real logos or URLs.
- USSD codes, the SMS short code and the helpline are invented. They are listed under "How to reach
  us" in `balance-check`, and every document uses the same ones. Codes and short numbers are always
  written in backticks.
- Phone numbers: example customer numbers are `01700000000` to `01700000008`. `01700000009` is
  reserved for the helpline from other phones.
- Ticket numbers have the form `TKT-` plus 6 digits; examples use only `TKT-000001` to
  `TKT-000009`.
- No ID-card numbers, PINs or one-time codes as examples, not even invented ones: write "your PIN",
  never a sample value.
- No personal data.
- The bn files are plain customer-care Bangla. Loanwords may be written in Bengali script as people
  normally write them; package codes, USSD codes and SMS keywords stay in Latin letters.
