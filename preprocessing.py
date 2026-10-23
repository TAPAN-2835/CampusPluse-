"""
CampusPulse - Text Preprocessing Module
Provides robust, explainable text cleaning and normalization for campus complaints.
Designed to work seamlessly offline with resilient fallbacks.

Config-driven design:
- Campus abbreviations  → data/campus_abbreviations.json  (add terms without touching code)
- Negation words        → data/negation_words.json        (preserved during stopword removal)
- Contraction expansion → `contractions` library          (200+ patterns, fallback regex dict)
- Stopwords             → NLTK english corpus             (fallback: DEFAULT_STOPWORDS below)
"""

import json
import os
import re
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Helper: load a JSON config file relative to this module
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_json(filename: str, fallback, label: str):
    """
    Loads a JSON file from the data/ directory.
    Returns `fallback` if the file is missing or malformed.
    """
    path = os.path.join(_DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[CampusPulse WARNING] {filename} not found — {label} will use fallback.")
        return fallback
    except (json.JSONDecodeError, Exception) as e:
        print(f"[CampusPulse WARNING] Failed to load {filename}: {e} — {label} will use fallback.")
        return fallback


# ---------------------------------------------------------------------------
# Negation Words — loaded from data/negation_words.json
# Preserved during stopword removal to retain complaint semantics.
# e.g. "wifi not working", "no water supply"
# ---------------------------------------------------------------------------
_negation_config = _load_json(
    "negation_words.json",
    fallback={"words": ["no", "not", "none", "neither", "never", "nobody", "nowhere", "hardly", "barely"]},
    label="negation words",
)
NEGATION_WORDS: Set[str] = set(_negation_config.get("words", []))


# ---------------------------------------------------------------------------
# Campus Abbreviations — loaded from data/campus_abbreviations.json
# Add new terms there without touching this file.
# ---------------------------------------------------------------------------
_raw_abbrevs = _load_json(
    "campus_abbreviations.json",
    fallback={},
    label="campus abbreviations",
)
CAMPUS_EXPANSIONS: Dict[str, str] = {k: v for k, v in _raw_abbrevs.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# DEFAULT_STOPWORDS — minimal hardcoded emergency fallback ONLY.
# Primary source is NLTK. This is only used when NLTK is completely unavailable.
# ---------------------------------------------------------------------------
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves"
}


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
def get_stopwords() -> Set[str]:
    """
    Retrieves English stopwords from NLTK (primary source).
    Falls back to DEFAULT_STOPWORDS if NLTK is unavailable.
    Always excludes NEGATION_WORDS so complaint context is preserved.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            nltk_stops = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk_stops = set(stopwords.words("english"))
        return nltk_stops - NEGATION_WORDS
    except Exception:
        return DEFAULT_STOPWORDS - NEGATION_WORDS


# ---------------------------------------------------------------------------
# Contraction Expansion
# Primary: `contractions` library (200+ English contractions)
# Fallback: small handcrafted regex dict (offline environments)
# ---------------------------------------------------------------------------
_CONTRACTION_FALLBACK: Dict[str, str] = {
    r"\bdoesn['']?t\b": "does not",
    r"\bdon['']?t\b":   "do not",
    r"\bisn['']?t\b":   "is not",
    r"\baren['']?t\b":  "are not",
    r"\bcan['']?t\b":   "cannot",
    r"\bwon['']?t\b":   "will not",
    r"\bwouldn['']?t\b":"would not",
    r"\bcouldn['']?t\b":"could not",
    r"\bshouldn['']?t\b":"should not",
    r"\bhaven['']?t\b": "have not",
    r"\bhadn['']?t\b":  "had not",
    r"\bhasn['']?t\b":  "has not",
    r"\bwasn['']?t\b":  "was not",
    r"\bweren['']?t\b": "were not",
    r"\bdid['']?t\b":   "did not",
}


def expand_contractions(text: str) -> str:
    """
    Expands English contractions to preserve negation words.
    Primary: uses the `contractions` library (e.g. "doesn't" → "does not").
    Fallback: applies a compact regex dict for offline environments.
    """
    lowered = text.lower()
    try:
        import contractions as ct
        return ct.fix(lowered)
    except ImportError:
        for pattern, replacement in _CONTRACTION_FALLBACK.items():
            lowered = re.sub(pattern, replacement, lowered)
        return lowered


# ---------------------------------------------------------------------------
# Campus Abbreviation Expansion
# ---------------------------------------------------------------------------
def expand_campus_abbreviations(text: str) -> str:
    """
    Expands domain-specific campus abbreviations loaded from
    data/campus_abbreviations.json. Applies contraction expansion first.
    """
    expanded = expand_contractions(text)
    for abbr, full in CAMPUS_EXPANSIONS.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        expanded = re.sub(pattern, full, expanded, flags=re.IGNORECASE)
    return expanded


# ---------------------------------------------------------------------------
# Main Text Cleaning Function
# ---------------------------------------------------------------------------
def clean_text(
    text: Optional[str],
    remove_stopwords: bool = True,
    lemmatize: bool = False,
) -> str:
    """
    Cleans and normalizes raw complaint text through a multi-step pipeline:
    1. Type & edge check (handles None, empty strings, non-strings)
    2. Contraction expansion via `contractions` library (with regex fallback)
    3. Campus abbreviation expansion from data/campus_abbreviations.json
    4. URL, email, and phone number removal
    5. Non-alphabetic character stripping
    6. Tokenization
    7. Optional stopword removal (negation words always retained)
    8. Optional lemmatization via NLTK WordNetLemmatizer
    9. Whitespace normalization
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""

    # Steps 2 & 3
    text = expand_campus_abbreviations(text)

    # Step 4: remove URLs, emails, phone numbers
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d{10}\b", " ", text)

    # Step 5: strip non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s\-]", " ", text)
    text = re.sub(r"-+", " ", text)

    # Step 6: tokenize
    tokens = [tok.strip() for tok in text.split() if tok.strip()]
    tokens = [t for t in tokens if len(t) > 1 or t in {"a", "i"}]

    # Step 7: stopword removal (negation words always kept)
    if remove_stopwords:
        stops = get_stopwords()
        tokens = [t for t in tokens if t not in stops]

    # Step 8: optional lemmatization
    if lemmatize:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            pass

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Explainability Helper
# ---------------------------------------------------------------------------
def explain_preprocessing(raw_text: str) -> Dict[str, object]:
    """
    Returns a step-by-step trace of how raw input was transformed.
    Designed for the CampusPulse dashboard explainability view and viva demos.
    """
    step1_contractions = expand_contractions(raw_text or "")
    step2_abbrev = expand_campus_abbreviations(raw_text or "")
    step3_cleaned = re.sub(r"[^a-zA-Z\s\-]", " ", step2_abbrev)
    tokens_raw = [t.strip().lower() for t in step3_cleaned.split() if len(t.strip()) > 1]

    stops = get_stopwords()
    retained_tokens = [t for t in tokens_raw if t not in stops]
    removed_stops = [t for t in tokens_raw if t in stops]

    return {
        "raw_text": raw_text,
        "contraction_expanded": step1_contractions,
        "abbreviation_expanded": step2_abbrev,
        "tokens_extracted": tokens_raw,
        "stopwords_removed": list(set(removed_stops)),
        "retained_tokens": retained_tokens,
        "final_cleaned": " ".join(retained_tokens),
        "token_count": len(retained_tokens),
        "campus_expansions_loaded": len(CAMPUS_EXPANSIONS),
        "negation_words_loaded": len(NEGATION_WORDS),
    }
