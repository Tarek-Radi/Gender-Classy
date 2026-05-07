import math
import re
import unicodedata
from typing import Optional

import pandas as pd

from config import USE_FIRST_TOKEN_ONLY


_SPACE_RE = re.compile(r"\s+")
_DASH_LIKE_RE = re.compile(r"[‐‑‒–—―]+")
_APOSTROPHE_LIKE_RE = re.compile(r"[`´‘’ʼʹʾ]")
_QUOTE_RE = re.compile(r'["“”]')


LABEL_MAPPING = {
    "m": "male",
    "male": "male",
    "man": "male",
    "boy": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
    "girl": "female",
    "?": "both",
    "u": "both",
    "b": "both",
    "both": "both",
    "unknown": "both",
    "unisex": "both",
    "ambiguous": "both",
    "mixed": "both",
    "other": "both",
}


def _is_valid_char(char: str) -> bool:
    if char in {" ", "-", "'"}:
        return True
    return unicodedata.category(char).startswith("L")


def normalize_name(name: object, use_first_token_only: bool = USE_FIRST_TOKEN_ONLY) -> str:
    if name is None:
        return ""
    if isinstance(name, float) and math.isnan(name):
        return ""

    text = str(name).strip()
    if not text:
        return ""

    text = text.casefold()
    text = _DASH_LIKE_RE.sub("-", text)
    text = _APOSTROPHE_LIKE_RE.sub("'", text)
    text = _QUOTE_RE.sub("", text)
    text = "".join(ch for ch in text if not ch.isdigit())

    cleaned = []
    for ch in text:
        if _is_valid_char(ch):
            cleaned.append(ch)
    text = "".join(cleaned)

    text = text.strip(" -'")
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    if use_first_token_only:
        for token in text.split(" "):
            token = token.strip(" -'")
            if any(unicodedata.category(ch).startswith("L") for ch in token):
                return token
        return ""

    return text


def standardize_gender_label(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    key = str(value).strip().casefold()
    if not key:
        return None

    label = LABEL_MAPPING.get(key)
    if label:
        return label

    # Handle values like "male " / "female."
    key = re.sub(r"[^a-z?]", "", key)
    return LABEL_MAPPING.get(key)


def normalize_name_series(series: pd.Series, use_first_token_only: bool = USE_FIRST_TOKEN_ONLY) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.casefold().str.strip()
    cleaned = cleaned.str.replace(_DASH_LIKE_RE.pattern, "-", regex=True)
    cleaned = cleaned.str.replace(_APOSTROPHE_LIKE_RE.pattern, "'", regex=True)
    cleaned = cleaned.str.replace(_QUOTE_RE.pattern, "", regex=True)
    cleaned = cleaned.str.replace(r"\d+", "", regex=True)
    cleaned = cleaned.str.replace("_", " ", regex=False)
    cleaned = cleaned.str.replace(r"[^\w\s\-']", "", regex=True)
    cleaned = cleaned.str.replace(_SPACE_RE.pattern, " ", regex=True).str.strip(" -'")

    if use_first_token_only:
        cleaned = cleaned.str.split(" ").str[0].fillna("")

    cleaned = cleaned.str.strip(" -'")
    cleaned = cleaned.where(cleaned.ne(""), "")
    return cleaned
