"""
CampusPulse - Text Preprocessing Module
Provides robust, explainable text cleaning and normalization for campus complaints.
Designed to work seamlessly offline with resilient fallbacks.
"""

import re
import string
from typing import Dict, List, Optional, Set

# Fallback standard English stopwords (excluding critical negation words)
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

# Negation words explicitly preserved to retain complaint context ("not working", "no water")
NEGATION_WORDS: Set[str] = {"no", "not", "none", "neither", "never", "nobody", "nowhere", "hardly", "barely"}

# Common campus abbreviations and normalizations
CAMPUS_EXPANSIONS: Dict[str, str] = {
    "wi-fi": "wifi",
    "wi fi": "wifi",
    "ac": "air conditioner",
    "a/c": "air conditioner",
    "prof": "professor",
    "dept": "department",
    "admin": "administration",
    "lab": "laboratory",
    "washroom": "washroom",
    "restroom": "washroom",
    "toilet": "washroom",
    "bathroom": "washroom",
    "mess": "mess food",
    "canteen": "cafeteria",
    "hostel": "hostel",
    "w/o": "without",
    "b/w": "between",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "r": "are",
    "urgnt": "urgent",
    " asap": " as soon as possible",
}


def get_stopwords() -> Set[str]:
    """Retrieves English stopwords, attempting NLTK with a dependable fallback."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            nltk_stops = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk_stops = set(stopwords.words("english"))
        # Remove negation words so context is preserved
        return nltk_stops - NEGATION_WORDS
    except Exception:
        return DEFAULT_STOPWORDS - NEGATION_WORDS


# Contraction patterns to ensure negation words ('not', 'cannot') are preserved
CONTRACTION_EXPANSIONS: Dict[str, str] = {
    r"\bdoesn['’]?t\b": "does not",
    r"\bdon['’]?t\b": "do not",
    r"\bdid['’]?t\b": "did not",
    r"\bisn['’]?t\b": "is not",
    r"\baren['’]?t\b": "are not",
    r"\bwasn['’]?t\b": "was not",
    r"\bweren['’]?t\b": "were not",
    r"\bhasn['’]?t\b": "has not",
    r"\bhaven['’]?t\b": "have not",
    r"\bhadn['’]?t\b": "had not",
    r"\bwon['’]?t\b": "will not",
    r"\bwouldn['’]?t\b": "would not",
    r"\bcan['’]?t\b": "cannot",
    r"\bcouldn['’]?t\b": "could not",
    r"\bshouldn['’]?t\b": "should not",
    r"\bdoesnt\b": "does not",
    r"\bdont\b": "do not",
    r"\bisnt\b": "is not",
    r"\bwont\b": "will not",
    r"\bcant\b": "cannot",
}


def expand_contractions(text: str) -> str:
    """Expands negative and modal contractions so negation words are preserved."""
    lowered = text.lower()
    for pattern, replacement in CONTRACTION_EXPANSIONS.items():
        lowered = re.sub(pattern, replacement, lowered)
    return lowered


def expand_campus_abbreviations(text: str) -> str:
    """Expands domain-specific campus contractions and abbreviations."""
    lowered = expand_contractions(text)
    for abbr, full in CAMPUS_EXPANSIONS.items():
        # Match whole words or boundary patterns
        pattern = r"\b" + re.escape(abbr) + r"\b"
        lowered = re.sub(pattern, full, lowered)
    return lowered


def clean_text(
    text: Optional[str],
    remove_stopwords: bool = True,
    lemmatize: bool = False,
) -> str:
    """
    Cleans and normalizes raw complaint text.
    Steps:
    1. Type and edge check (handles None, empty strings, numbers)
    2. Lowercasing
    3. Campus acronym expansion
    4. URL, email, and special symbol removal
    5. Tokenization & punctuation stripping
    6. Optional stopword removal (with negation retention)
    7. Whitespace normalization
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return ""

    # Normalization
    text = expand_campus_abbreviations(text)

    # Remove URLs, emails, and phone-like patterns
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d{10}\b", " ", text)

    # Retain only alphabetic characters, spaces, and essential hyphens
    text = re.sub(r"[^a-zA-Z\s\-]", " ", text)
    text = re.sub(r"-+", " ", text)

    # Tokenize by word boundary
    tokens = [tok.strip() for tok in text.split() if tok.strip()]

    # Filter short single-letter tokens except meaningful ones
    tokens = [t for t in tokens if len(t) > 1 or t in {"a", "i"}]

    if remove_stopwords:
        stops = get_stopwords()
        tokens = [t for t in tokens if t not in stops]

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
            pass  # Graceful fallback without crashing

    return " ".join(tokens)


def explain_preprocessing(raw_text: str) -> Dict[str, object]:
    """
    Provides step-by-step transparency into how the raw input string was transformed.
    Ideal for faculty demonstration and explainability.
    """
    step1_expanded = expand_campus_abbreviations(raw_text or "")
    step2_cleaned = re.sub(r"[^a-zA-Z\s\-]", " ", step1_expanded)
    tokens_raw = [t.strip().lower() for t in step2_cleaned.split() if len(t.strip()) > 1]

    stops = get_stopwords()
    retained_tokens = [t for t in tokens_raw if t not in stops]
    removed_stops = [t for t in tokens_raw if t in stops]

    final_text = " ".join(retained_tokens)

    return {
        "raw_text": raw_text,
        "abbreviation_expanded": step1_expanded,
        "tokens_extracted": tokens_raw,
        "stopwords_removed": list(set(removed_stops)),
        "retained_tokens": retained_tokens,
        "final_cleaned": final_text,
        "token_count": len(retained_tokens),
    }
