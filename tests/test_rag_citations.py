import re
from pathlib import Path

import pytest

from kotha.kb.chunks import chunk_documents
from kotha.kb.documents import read_documents
from kotha.rag.citations import (
    CitationProblem,
    ProblemKind,
    cited_chunk_ids,
    numbers_in,
    split_sentences,
    strip_citations,
    verify_answer,
)

KB_FOLDER = Path(__file__).resolve().parent.parent / "data" / "kb"
KB_CHUNKS = chunk_documents(read_documents(KB_FOLDER))
LIST_MARKER = re.compile(r"^\s*[\d০-৯]+\.\s+")
PACK_ID = "packages-internet.en.0"
SIM_ID = "sim-replacement.bn.1"
RETRIEVED = {
    PACK_ID: "NT-30GB gives 30 GB for 30 days. We have 1,000 agents.",
    SIM_ID: "নতুন সিম নিতে জাতীয় পরিচয়পত্র আনুন। সেবা ১০টা থেকে।",
}
UNSUPPORTED_30 = [CitationProblem(ProblemKind.UNSUPPORTED_NUMBER, "30")]


def kinds(answer: str, retrieved: dict[str, str] = RETRIEVED) -> list[ProblemKind]:
    return [problem.kind for problem in verify_answer(answer, retrieved)]


def unsupported(answer: str, source: str) -> list[str]:
    return [
        problem.detail
        for problem in verify_answer(answer, {"c": source})
        if problem.kind == ProblemKind.UNSUPPORTED_NUMBER
    ]


@pytest.mark.parametrize(
    "answer",
    [
        f"NT-30GB gives 30 GB. [{PACK_ID}]",
        f"NT-30GB gives 30 GB [{PACK_ID}].",
        f"NT-30GB gives 30 GB [{PACK_ID}]",
        f"জাতীয় পরিচয়পত্র আনুন। [{SIM_ID}]",
        f"জাতীয় পরিচয়পত্র আনুন [{SIM_ID}]।",
        f"It lasts 30 days. [{PACK_ID}] Bring your ID. [{SIM_ID}]",
        f"It lasts 30 days. [{PACK_ID}] [{SIM_ID}]",
        f"It lasts 30 days. [{PACK_ID}].",
        f"মেয়াদ ৩০ দিন। [{PACK_ID}]।",
        f"It lasts 30 days!! [{PACK_ID}]",
    ],
)
def test_a_cited_answer_passes(answer: str) -> None:
    assert verify_answer(answer, RETRIEVED) == []


def test_every_factual_sentence_needs_its_own_citation() -> None:
    assert verify_answer(f"It lasts 30 days. Bring your ID. [{SIM_ID}]", RETRIEVED) == [
        CitationProblem(ProblemKind.MISSING_CITATION, "It lasts 30 days."),
        CitationProblem(ProblemKind.UNSUPPORTED_NUMBER, "30"),
    ]


def test_a_bangla_sentence_needs_its_own_citation_too() -> None:
    assert verify_answer(f"আইডি আনুন। সেন্টারে যান। [{SIM_ID}]", RETRIEVED) == [
        CitationProblem(ProblemKind.MISSING_CITATION, "আইডি আনুন।")
    ]


@pytest.mark.parametrize(
    "answer", [f"[{PACK_ID}] It lasts 30 days.", f"It lasts [{PACK_ID}] 30 days."]
)
def test_a_citation_before_the_end_of_a_sentence_does_not_count(answer: str) -> None:
    assert kinds(answer) == [ProblemKind.MISSING_CITATION]


def test_an_answer_with_no_citation_fails() -> None:
    assert kinds("Bring your national ID.") == [ProblemKind.MISSING_CITATION]


@pytest.mark.parametrize(
    "answer",
    [
        f"Bring your ID\nGo to the centre [{SIM_ID}]",
        f"Bring your ID\rGo to the centre [{SIM_ID}]",
        f"- Bring your ID\n- Pay at the desk\n- Collect the SIM [{SIM_ID}]",
        f"Bring your ID.Go to the centre. [{SIM_ID}]",
        f'"Bring your ID." Go to the centre. [{SIM_ID}]',
        f"**Bring your ID.** Go to the centre. [{SIM_ID}]",
        f"(Bring your ID.) Go to the centre. [{SIM_ID}]",
        f"আইডি আনুন॥ সেন্টারে যান। [{SIM_ID}]",
        f"আইডি আনুন | সেন্টারে যান। [{SIM_ID}]",
        f"Bring your ID؟ Go to the centre! Refunds are instant. [{SIM_ID}]",
        f"Bring your ID... Refunds are instant. [{SIM_ID}]",
        f"Bring your ID… Refunds are instant. [{SIM_ID}]",
        f"আইডি আনুন... রিফান্ড সাথে সাথে হয়। [{SIM_ID}]",
        f"Centres close at 6 p.m. Refunds are instant. [{SIM_ID}]",
        f"You pay in tk. Refunds are instant. [{SIM_ID}]",
        f"Bring your ID.​ Refunds are instant. [{SIM_ID}]",
    ],
)
def test_an_uncited_sentence_cannot_hide_behind_any_boundary(answer: str) -> None:
    assert ProblemKind.MISSING_CITATION in kinds(answer)


@pytest.mark.parametrize(
    "answer", [f"It lasts 30 days. [{PACK_ID}] Anything else?", "আর কিছু জানতে চান?", "هل؟"]
)
def test_a_question_back_to_the_caller_needs_no_citation(answer: str) -> None:
    assert verify_answer(answer, RETRIEVED) == []


@pytest.mark.parametrize(
    "answer",
    [
        f"Sure! It lasts 30 days. [{PACK_ID}]",
        f"জি। মেয়াদ ৩০ দিন। [{PACK_ID}]",
        f"It lasts 30 days. [{PACK_ID}] Thank you for calling.",
        f"You're welcome. It lasts 30 days. [{PACK_ID}]",
        f"মেয়াদ ৩০ দিন। [{PACK_ID}] আপনাকে ধন্যবাদ।",
    ],
)
def test_courtesy_phrases_need_no_citation(answer: str) -> None:
    assert verify_answer(answer, RETRIEVED) == []


@pytest.mark.parametrize("answer", ["Yes.", "No.", "হ্যাঁ।"])
def test_yes_and_no_can_state_a_fact_so_they_need_a_citation(answer: str) -> None:
    assert kinds(answer) == [ProblemKind.MISSING_CITATION]


def test_a_cited_id_must_be_among_the_retrieved_chunks() -> None:
    assert verify_answer("Bring your ID. [refunds.en.9]", RETRIEVED) == [
        CitationProblem(ProblemKind.UNKNOWN_CITATION, "refunds.en.9")
    ]


def test_an_invented_number_is_rejected_in_any_digits() -> None:
    expected = [CitationProblem(ProblemKind.UNSUPPORTED_NUMBER, "499")]
    assert verify_answer(f"You get 499 GB. [{PACK_ID}]", RETRIEVED) == expected
    assert verify_answer(f"আপনি ৪৯৯ জিবি পাবেন। [{PACK_ID}]", RETRIEVED) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "NT-30GB costs 30 taka. [c]",
        "NT-30GB-এর দাম ৩০ টাকা। [c]",
        "NT-30GB costs Tk 30. [c]",
        "You pay Tk: 30. [c]",
        "You pay Tk-30. [c]",
        "You pay tk30. [c]",
        "You pay 30tk. [c]",
        "You pay 30taka. [c]",
        "You pay 30BDT. [c]",
        "You pay 30/-. [c]",
        "You pay Rs 30. [c]",
        "You pay 30 Bangladeshi taka. [c]",
        "আপনাকে ৩০ বাংলাদেশি টাকা দিতে হবে। [c]",
        "Calls are 30 paisa per minute. [c]",
        "প্রতি মিনিটে ৩০ পয়সা। [c]",
        "NT-30GB is ৳30. [c]",
        "The price is 30. [c]",
        "The charge is 30. [c]",
        "You will be charged 30 for this pack. [c]",
        "You pay 30 for NT-30GB. [c]",
        "The rate is 30. [c]",
        "The fees are 30. [c]",
        "30 is the price of this pack. [c]",
        "৩০ হলো এই প্যাকের দাম। [c]",
        "এই প্যাকের মূল্য ৩০। [c]",
        "NT-30GB-এর দাম ত্রিশ টাকা। [c]",
        "তিরিশ টাকা দিন। [c]",
        "Your balance is 30. [c]",
        "আপনার ব্যালেন্স ৩০। [c]",
        "You can get NT-30GB for 30. [c]",
        "NT-30GB is only 30. [c]",
        "We charge 30 for NT-30GB. [c]",
        "30 will be deducted from your account. [c]",
        "You will get 30 back. [c]",
        "আপনাকে ৩০ দিতে হবে। [c]",
        "NT-30GB নিতে ৩০ লাগবে। [c]",
        "৩০ কেটে নেওয়া হবে। [c]",
        "NT-30GB কিনতে খরচ ৩০। [c]",
        "NT-30GB is Taka 30. [c]",
        "NT-30GB টাকা ৩০। [c]",
        "What does NT-30GB cost? It is 30. [c]",
        "NT-30GB-এর দাম কত? ৩০। [c]",
        "NT-30GB is 30 rupees. [c]",
        "NT-30GB price-30. [c]",
        "Buy the Tk 30 pack. [c]",
        "The ৳30 pack is good. [c]",
        "NT-30GB 30/=. [c]",
        "NT-30GB ৩০ টাঃ। [c]",
        "NT-30GB is $30. [c]",
        "NT-30GB is 30$. [c]",
    ],
)
def test_a_money_amount_is_not_supported_by_digits_in_a_chunk(answer: str) -> None:
    assert "30" in unsupported(answer, "NT-30GB gives 30 GB for 30 days")


def test_a_money_amount_is_accepted_only_from_money_values() -> None:
    answer = f"NT-30GB costs ৪৯৯ টাকা। [{PACK_ID}]"
    assert kinds(answer) == [ProblemKind.UNSUPPORTED_NUMBER]
    assert verify_answer(answer, RETRIEVED, ["price: 499"]) == [
        CitationProblem(ProblemKind.UNSUPPORTED_NUMBER, "499")
    ]
    assert verify_answer(answer, RETRIEVED, money_amounts=["499.00"]) == []


def test_a_pack_size_in_a_tool_result_is_not_a_price() -> None:
    tool_result = ['{"code": "NT-30GB", "data_gb": 30, "validity_days": 30}']
    assert verify_answer("NT-30GB costs 30 taka.", {}, tool_result, ["499"]) == [
        CitationProblem(ProblemKind.MISSING_CITATION, "NT-30GB costs 30 taka."),
        CitationProblem(ProblemKind.UNSUPPORTED_NUMBER, "30"),
    ]


@pytest.mark.parametrize(
    ("answer", "amount"),
    [("Your balance is 250 taka.", "250.00"), ("আপনার ব্যালেন্স ২৫০.৫০ টাকা।", "250.50")],
)
def test_a_billing_amount_still_needs_a_citable_source(answer: str, amount: str) -> None:
    problems = verify_answer(answer, {}, money_amounts=[amount])
    assert [problem.kind for problem in problems] == [ProblemKind.MISSING_CITATION]
    source = {"bss.balance": "account balance"}
    assert verify_answer(f"{answer} [bss.balance]", source, money_amounts=[amount]) == []


@pytest.mark.parametrize(
    ("answer", "money"),
    [
        ("NT-30GB costs 499 taka and lasts 30 days. [c]", ["499"]),
        ("NT-30GB-এর দাম ৪৯৯ টাকা, মেয়াদ ৩০ দিন। [c]", ["499"]),
        ("The 30 GB pack costs 499 taka. [c]", ["499"]),
        ("The 30-day pack costs 499 taka. [c]", ["499"]),
        ("There is no fee for the first 30 days. [c]", []),
        ("প্যাকের দাম অ্যাপে দেখুন, মেয়াদ ৩০ দিন। [c]", []),
        ("For the price, call 01700000009. [c]", []),
    ],
)
def test_a_number_with_a_non_money_unit_in_a_price_sentence_is_not_money(
    answer: str, money: list[str]
) -> None:
    source = "NT-30GB gives 30 GB for 30 days. Call 01700000009."
    assert verify_answer(answer, {"c": source}, money_amounts=money) == []


@pytest.mark.parametrize(
    "answer", ["Refunds take 30 days.", "NT-5GB is free forever.", "Your balance is low."]
)
def test_a_tool_result_does_not_waive_the_citation_of_other_facts(answer: str) -> None:
    tool_results = ["pack: NT-5GB, validity_days: 30"]
    assert ProblemKind.MISSING_CITATION in [
        problem.kind for problem in verify_answer(answer, {}, tool_results, ["250"])
    ]


def test_a_tool_result_can_support_a_number_that_is_not_money() -> None:
    answer = f"You have 3 GB left. [{PACK_ID}]"
    assert kinds(answer) == [ProblemKind.UNSUPPORTED_NUMBER]
    assert verify_answer(answer, RETRIEVED, ["data_left_gb: 3"]) == []


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("The price of NT-30GB is shown in the app. [c]", "NT-30GB. See the app."),
        ("For the current price, call 747. [c]", "Call 747."),
        ("For the current price, dial *747*4#. [c]", "Dial *747*4#."),
        ("দাম জানতে ৭৪৭ নম্বরে কল করুন। [c]", "কল করুন ৭৪৭"),
        ("বিনামূল্যে ৫ জিবি পাবেন। [c]", "বিনামূল্যে ৫ জিবি"),
        ("টাকা ৭ দিনের মধ্যে ফেরত পাবেন। [c]", "৭ দিনের মধ্যে"),
        ("We check the charge and reply in 30 days. [c]", "reply in 30 days"),
        ("Recharge within 15 minutes. [c]", "15 minutes"),
        ("The helpline number is 747. [c]", "Call 747"),
        ("The closing time is 18:00. [c]", "open until 18:00"),
        ("৩ কর্মদিবসের মধ্যে টাকা ফেরত পাবেন। [c]", "৩ কর্মদিবস"),
        ("টাকা ফেরত পেতে ৩ কর্মদিবস লাগবে। [c]", "৩ কর্মদিবস"),
        ("আপনাকে ৪ অঙ্কের পিন দিতে হবে। [c]", "৪ অঙ্কের পিন"),
        ("The priority one is open. [c]", "Text."),
        ("সকাল ১০টা: সেন্টার খোলে। [c]", "১০টা"),
        ("You get 2.3 lakh minutes. [c]", "230000 minutes"),
    ],
)
def test_ordinary_sentences_near_money_words_are_accepted(answer: str, source: str) -> None:
    assert verify_answer(answer, {"c": source}) == []


def test_a_word_that_only_contains_the_bangla_word_for_free_is_not_a_price() -> None:
    assert verify_answer("আফ্রিকায় রোমিং চালু আছে। [c]", {"c": "রোমিং"}) == []


@pytest.mark.parametrize(
    "answer",
    [
        "NT-60GB is free. [c]",
        "NT-60GB বিনামূল্যে পাবেন। [c]",
        "NT-60GB costs nothing. [c]",
        "There is no charge for NT-60GB. [c]",
        "NT-60GB নিতে কোনো টাকা লাগবে না। [c]",
    ],
)
def test_calling_something_free_needs_support(answer: str) -> None:
    assert unsupported(answer, "NT-60GB gives 60 GB") == ["0"]
    assert unsupported(answer, "NT-60GB is free for staff, বিনামূল্যে") == []


@pytest.mark.parametrize(
    ("answer", "number"),
    [
        ("It lasts thirty-one days. [c]", "31"),
        ("It lasts thirty one days. [c]", "31"),
        ("সাড়ে ৫ জিবি পাবেন। [c]", "5.5"),
        ("সোয়া ৫ জিবি পাবেন। [c]", "5.25"),
        ("পৌনে ৫ জিবি পাবেন। [c]", "4.75"),
    ],
)
def test_combined_number_forms_are_not_read_as_their_parts(answer: str, number: str) -> None:
    assert unsupported(answer, "5 GB for 30 days") == [number]


def test_the_bangla_word_for_notice_is_not_a_lakh() -> None:
    assert verify_answer("লক্ষ করুন, মেয়াদ ৩০ দিন। [c]", {"c": "৩০ দিন"}) == []


@pytest.mark.parametrize(
    ("answer", "number"),
    [
        ("NT-30GB-এর দাম পাঁচশো টাকা। [c]", "500"),
        ("It costs five hundred taka. [c]", "500"),
        ("দাম ৩০ হাজার টাকা। [c]", "30000"),
        ("৫ লক্ষ টাকা জমা হবে। [c]", "500000"),
        ("৫হাজার টাকা দিন। [c]", "5000"),
        ("5lakh taka please. [c]", "500000"),
        ("NT-77 is 30k taka. [c]", "30000"),
        ("নয় টাকা জমা হবে। [c]", "9"),
        ("শূন্য টাকা জমা হবে। [c]", "0"),
        ("zero taka please. [c]", "0"),
        ("প্যাকটি নয় দিন চলে। [c]", "9"),
        ("এক মাস চলে। [c]", "1"),
        ("You get one GB. [c]", "1"),
        ("তিনবার চেষ্টা করুন। [c]", "3"),
        ("তিনমাস চলে। [c]", "3"),
        ("ছয়ঘণ্টা চলে। [c]", "6"),
        ("ত্রিশের বেশি আছে। [c]", "30"),
        ("তিনটা প্যাক আছে। [c]", "3"),
        ("পঁচিশ টাকা দিন। [c]", "25"),
        ("নিরানব্বই টাকা দিন। [c]", "99"),
        ("তেরো টাকা দিন। [c]", "13"),
        ("কুড়ি টাকা দিন। [c]", "20"),
        ("দেড়শ টাকা দিন। [c]", "150"),
        ("আড়াইশো টাকা দিন। [c]", "250"),
        ("আপনাকে পাঁচ শত টাকা দিতে হবে। [c]", "500"),
        ("আপনাকে এক টাকা দিতে হবে। [c]", "1"),
        ("আপনাকে এক লক্ষ টাকা দিতে হবে। [c]", "100000"),
        ("You pay one taka. [c]", "1"),
        ("It costs thirteen taka. [c]", "13"),
        ("It lasts seventeen days. [c]", "17"),
        ("দুটি প্যাক আছে। [c]", "2"),
        ("সেন্টার পাঁচটায় বন্ধ হয়। [c]", "5"),
        ("প্যাকটি নব্বইদিন চলবে। [c]", "90"),
        ("You get twenty-five GB. [c]", "25"),
        ("You get two hundred and fifty GB. [c]", "250"),
    ],
)
def test_numbers_written_as_words_are_checked(answer: str, number: str) -> None:
    assert number in unsupported(answer, "NT-77 gives 77 GB")


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("You get five hundred GB. [c]", "5 GB or 100 minutes"),
        ("দুই হাজার জিবি পাবেন। [c]", "২ দিন, ১০০০ মিনিট"),
    ],
)
def test_compound_number_words_are_one_number(answer: str, source: str) -> None:
    assert len(unsupported(answer, source)) == 1


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("There are 6 packs. [c]", "We sell six packs."),
        ("There are six packs. [c]", "We sell 6 packs."),
        ("ছয়টি প্যাক আছে। [c]", "৬টি প্যাক"),
        ("It has 200 minutes. [c]", "It has two hundred minutes."),
        ("৫০০ মিনিট পাবেন। [c]", "পাঁচ শত মিনিট পাবেন"),
    ],
)
def test_a_number_word_and_its_digits_support_each_other(answer: str, source: str) -> None:
    assert verify_answer(answer, {"c": source}) == []


def test_ordinary_words_are_not_read_as_numbers() -> None:
    assert verify_answer("This one is not a pack. [c]", {"c": "Text."}) == []
    assert verify_answer("এটি একটি প্যাক নয়। [c]", {"c": "লেখা"}) == []


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("You get 4,99 points. [c]", "499 points"),
        ("You get 1,5 GB. [c]", "15 GB pack"),
        ("Open until 08:30. [c]", "from 00:00 to 08:00, valid 30 days"),
        ("It gives 8 GB. [c]", "from 00:00 to 08:00"),
        ("It lasts 12 days. [c]", "from 00:00 to 08:00"),
        ("Centres open at 6 am. [c]", "open from 10:00 to 18:00"),
        ("You get 3 GB. [c]", "NT-30GB gives 30 GB"),
    ],
)
def test_a_different_number_is_not_supported_by_similar_digits(answer: str, source: str) -> None:
    assert unsupported(answer, source)


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("Choose 10 GB. [c]", "packs of 10,20,30 GB"),
        ("The night pack ends at 8:00. [c]", "from 00:00 to 08:00"),
        ("You get 1.50 GB. [c]", "1.5 GB"),
        ("আপনি ১.৫ জিবি পাবেন। [c]", "1.5 GB"),
        ("You have 1,00,000 points. [c]", "100000 points"),
        ("You have 1000 points. [c]", "1,000 points"),
        ("Use code ০৭ today. [c]", "code 7"),
        ("Open from 10. [c]", "সেবা ১০টা থেকে"),
    ],
)
def test_the_same_number_in_another_form_is_supported(answer: str, source: str) -> None:
    assert verify_answer(answer, {"c": source}) == []


def test_a_citation_cannot_join_digits_into_a_supported_number() -> None:
    assert unsupported("You get 4[c]99 points. [c]", "499 points") == ["4", "99"]


def test_format_characters_cannot_split_or_hide_a_number() -> None:
    assert unsupported("You get 4‌99 points. [c]", "4 and 99") == ["499"]


def test_compatibility_digits_are_read_as_digits() -> None:
    assert unsupported("You get ⑤ GB. [c]", "1 GB") == ["5"]
    assert unsupported("You get ５ GB. [c]", "1 GB") == ["5"]


def test_an_unsupported_number_is_reported_once() -> None:
    assert kinds(f"It is 77 and 77. [{PACK_ID}]") == [ProblemKind.UNSUPPORTED_NUMBER]


def test_a_number_from_an_uncited_chunk_is_not_supported() -> None:
    assert verify_answer(f"It lasts 30 days. [{SIM_ID}]", RETRIEVED) == UNSUPPORTED_30


def test_a_number_cited_only_through_an_unknown_id_is_not_supported() -> None:
    assert kinds(f"It lasts 30 days. [{SIM_ID}]", {PACK_ID: "No numbers here."}) == [
        ProblemKind.UNKNOWN_CITATION,
        ProblemKind.UNSUPPORTED_NUMBER,
    ]


def test_digits_inside_a_chunk_id_are_not_numbers_in_the_answer() -> None:
    assert verify_answer("Bring your ID. [plans.en.77]", {"plans.en.77": "Bring ID."}) == []


def test_digits_in_a_malformed_citation_are_still_checked() -> None:
    assert ProblemKind.UNSUPPORTED_NUMBER in kinds(f"Bring ID [a 77]. [{PACK_ID}]")


@pytest.mark.parametrize("answer", ["", "   ", f"[{PACK_ID}]", f"। [{PACK_ID}]", "...", "​"])
def test_an_answer_without_words_is_empty(answer: str) -> None:
    assert verify_answer(answer, RETRIEVED) == [CitationProblem(ProblemKind.EMPTY_ANSWER, "")]


@pytest.mark.parametrize("argument", ["tool_results", "money_amounts"])
def test_a_single_string_is_refused_as_a_list_of_values(argument: str) -> None:
    with pytest.raises(TypeError, match="sequences of strings"):
        verify_answer("Balance is 5. [c]", {"c": "x"}, **{argument: "balance: 250"})


def test_sentences_split_on_every_ending_but_not_inside_numbers_or_ids() -> None:
    answer = f"Use 1.5 GB. [{PACK_ID}] কত বাকি? জানাবো! Done [{SIM_ID}]।"
    assert split_sentences(answer) == [
        f"Use 1.5 GB. [{PACK_ID}]",
        "কত বাকি?",
        "জানাবো!",
        f"Done [{SIM_ID}]।",
    ]


def test_cited_ids_keep_their_order_without_repeats() -> None:
    assert cited_chunk_ids(f"A [{SIM_ID}]. B [{PACK_ID}]. C [{SIM_ID}].") == [SIM_ID, PACK_ID]


def test_strip_citations_leaves_the_speakable_text() -> None:
    assert strip_citations(f"It lasts 30 days [{PACK_ID}] here. [{SIM_ID}]") == (
        "It lasts 30 days here."
    )


def test_numbers_are_read_from_codes_decimals_times_and_bangla_digits() -> None:
    assert numbers_in("*747*4# NT-30GB 1.5 GB ১২ 08:00 007") == [
        "747",
        "4",
        "30",
        "1.5",
        "12",
        "8:00",
        "7",
    ]


def test_real_chunks_support_their_own_numbers_but_not_a_price() -> None:
    chunk = next(c for c in KB_CHUNKS if c.lang == "en" and "| NT-30GB |" in c.text)
    retrieved = {chunk.id: chunk.text}
    assert verify_answer(f"NT-30GB gives 30 GB for 30 days. [{chunk.id}]", retrieved) == []
    assert verify_answer(f"NT-30GB costs 60 taka. [{chunk.id}]", retrieved) == [
        CitationProblem(ProblemKind.UNSUPPORTED_NUMBER, "60")
    ]


def test_every_prose_sentence_of_the_knowledge_base_passes_when_cited_to_its_chunk() -> None:
    rejected = []
    for chunk in KB_CHUNKS:
        prose_lines = [
            LIST_MARKER.sub("", line)
            for line in chunk.text.splitlines()
            if not line.startswith("|")
        ]
        for sentence in split_sentences(" ".join(" ".join(prose_lines).split())):
            if verify_answer(f"{sentence} [{chunk.id}]", {chunk.id: chunk.text}):
                rejected.append((chunk.id, sentence))
    assert rejected == []
