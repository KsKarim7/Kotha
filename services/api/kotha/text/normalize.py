import unicodedata

BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def normalize(text: str) -> str:
    visible = "".join(char for char in text if unicodedata.category(char) != "Cf")
    folded = unicodedata.normalize("NFC", visible).translate(BANGLA_DIGITS).lower()
    spaced = "".join(" " if unicodedata.category(char).startswith("P") else char for char in folded)
    return " ".join(spaced.split())
