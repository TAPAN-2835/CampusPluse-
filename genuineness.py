"""
CampusPulse - Complaint Genuineness and Mischief Detection Engine
Filters out spam, gibberish/keyboard mashing, profanity, dummy test submissions,
and student threats before complaints flood admin queues.

Transparent, rule-and-NLP-governed validation scoring:
- Returns a 0-100 Genuineness Score
- Determines a Verdict: GENUINE, NEEDS_REVIEW, or SPAM_REJECTED
- Provides transparent factor breakdowns and security/mischief flags
- Backed by data/genuineness_config.json for dynamic tuning
"""

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

# Try loading NLTK words for dictionary validation
_NLTK_WORDS_CACHE: Optional[Set[str]] = None


def _get_dictionary_words() -> Set[str]:
    """Lazy-loads NLTK English words corpus, with graceful fallback."""
    global _NLTK_WORDS_CACHE
    if _NLTK_WORDS_CACHE is not None:
        return _NLTK_WORDS_CACHE

    try:
        import nltk
        from nltk.corpus import words as nltk_words
        _NLTK_WORDS_CACHE = set(w.lower() for w in nltk_words.words())
    except Exception:
        # Fallback common vocabulary in case NLTK words corpus is unavailable
        _NLTK_WORDS_CACHE = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "water", "room", "floor", "class",
            "lab", "wifi", "network", "fan", "light", "exam", "marks", "fee",
            "broken", "dirty", "slow", "down", "help", "please", "issue",
            "problem", "fix", "repair", "service", "ac", "hostel", "mess"
        }
    return _NLTK_WORDS_CACHE


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CONFIG_FILE = os.path.join(_DATA_DIR, "genuineness_config.json")


def load_genuineness_config() -> Dict[str, Any]:
    """Loads genuineness scoring configuration from data/genuineness_config.json."""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Built-in fallbacks if file is missing or invalid
    return {
        "thresholds": {
            "min_chars": 15,
            "min_meaningful_tokens": 3,
            "min_char_diversity_ratio": 0.35,
            "min_dictionary_word_ratio": 0.40,
            "classifier_confidence_floor": 0.25,
            "max_word_repetition_ratio": 0.60,
            "verdict_genuine_floor": 60,
            "verdict_review_floor": 35,
        },
        "spam_patterns": [
            "test", "testing", "hello", "hi", "hey", "lorem ipsum", "sample",
            "abcd", "xyz", "asdf", "qwerty", "1234", "dummy", "placeholder",
            "foo bar", "blah", "bla bla", "random", "na", "n/a", "nothing",
            "idk", "idc", "whatever", "ok ok", "lol", "haha", "hehe",
        ],
        "threat_keywords": [
            "i will kill", "you are dead", "bomb", "i know where you live",
            "ill find you", "i will hurt", "i will destroy", "you will pay",
            "going to attack", "bring a weapon", "threaten", "blackmail",
        ],
        "profanity_keywords": [
            "fuck", "shit", "bastard", "bitch", "asshole", "damn you",
            "motherfucker", "idiot admin", "stupid staff", "useless staff",
            "crap", "bullshit", "wtf", "screw you", "go to hell",
        ],
        "campus_relevance_keywords": [
            "wifi", "water", "electricity", "lights", "lab", "library",
            "hostel", "mess", "canteen", "cafeteria", "classroom", "toilet",
            "washroom", "ac", "fan", "projector", "exam", "fee", "portal",
            "internet", "server", "network", "printer", "elevator", "lift",
            "parking", "road", "campus", "department", "professor", "teacher",
            "timetable", "schedule", "attendance", "marks", "grade", "result",
            "assignment", "submission", "deadline", "room", "floor", "block",
            "computer", "software", "hardware", "geyser", "heater", "pipe",
            "leaking", "broken", "repair", "maintenance", "security", "gate",
        ],
        "scoring_weights": {
            "length_ok": 15,
            "char_diversity_ok": 10,
            "dictionary_words_ok": 15,
            "campus_relevance": 20,
            "classifier_confidence": 15,
            "no_repetition": 10,
            "no_spam_pattern": 10,
            "no_profanity_penalty": -30,
            "threat_penalty": -60,
            "spam_pattern_penalty": -30,
        },
    }


# ---------------------------------------------------------------------------
# Mischief / Gibberish Heuristics
# ---------------------------------------------------------------------------
KEYBOARD_MASH_PATTERNS = [
    r"asdf[a-z]*",
    r"qwer[a-z]*",
    r"zxcv[a-z]*",
    r"hjkl[a-z]*",
    r"uiop[a-z]*",
    r"[0-9]{5,}",
    r"(.)\1{3,}",  # 4+ repeated characters, e.g. "aaaaa", "zzzzz"
]


def _detect_keyboard_mash(text: str) -> Tuple[bool, List[str]]:
    """Detects obvious keyboard mashing, repeated character streaks, and row sweeps."""
    lower = text.lower().strip()
    detected = []

    # Repeated characters like "aaaa", "zzzz", "!!!!!!"
    rep_match = re.findall(r"(.)\1{3,}", lower)
    if rep_match:
        detected.append(f"Repeated character streak ('{rep_match[0]}' x4+)")

    # Common keyboard sequences
    for pattern in KEYBOARD_MASH_PATTERNS[:5]:
        if re.search(pattern, lower):
            detected.append("Keyboard row sequence detected")
            break

    # Long token with no vowels
    tokens = re.findall(r"[a-zA-Z]{5,}", lower)
    vowels = set("aeiouy")
    for t in tokens:
        if not any(char in vowels for char in t):
            detected.append(f"Unpronounceable consonant cluster ('{t}')")
            break

    return (len(detected) > 0, detected)


# ---------------------------------------------------------------------------
# Core Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_genuineness(
    text: str,
    classifier_confidence: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates the genuineness and legitimacy of a student complaint description.

    Returns:
        dict containing:
            - score (int): 0-100 score indicating legitimacy
            - verdict (str): 'GENUINE', 'NEEDS_REVIEW', or 'SPAM_REJECTED'
            - is_genuine (bool): True if complaint should pass directly to queue
            - is_threat (bool): True if threat or intimidation was flagged
            - recommended_status (str): 'Pending', 'Under Review', or 'Spam/Blocked'
            - flags (list of str): specific warning or quality flags
            - factors (list of dict): transparent scoring breakdown
            - summary_reason (str): human-friendly explanation for admin
    """
    if config is None:
        config = load_genuineness_config()

    thresholds = config.get("thresholds", {})
    weights = config.get("scoring_weights", {})
    spam_patterns = [p.lower() for p in config.get("spam_patterns", [])]
    threat_keywords = [k.lower() for k in config.get("threat_keywords", [])]
    profanity_keywords = [p.lower() for p in config.get("profanity_keywords", [])]
    campus_keywords = set(k.lower() for k in config.get("campus_relevance_keywords", []))

    raw_text = text.strip() if text else ""
    lower_text = raw_text.lower()
    factors: List[Dict[str, Any]] = []
    flags: List[str] = []
    current_score = 0
    is_threat = False

    # 1. Length & Token Adequacy Check
    min_chars = thresholds.get("min_chars", 15)
    min_tokens = thresholds.get("min_meaningful_tokens", 3)
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", lower_text)
    meaningful_tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]

    too_short = len(raw_text) < min_chars or len(meaningful_tokens) < min_tokens
    if not too_short:
        pts = weights.get("length_ok", 15)
        current_score += pts
        factors.append({
            "factor": "Length & Detail",
            "points": f"+{pts}",
            "passed": True,
            "details": f"{len(raw_text)} chars, {len(meaningful_tokens)} descriptive tokens (min {min_chars} chars, {min_tokens} tokens).",
        })
    else:
        factors.append({
            "factor": "Length & Detail",
            "points": "0",
            "passed": False,
            "details": f"Too short ({len(raw_text)} chars, {len(meaningful_tokens)} tokens). Lacks sufficient context.",
        })
        flags.append("Description is too brief / insufficient context")

    # 2. Character Diversity / Gibberish Check
    clean_chars = [c for c in lower_text if c.isalnum()]
    char_diversity = (len(set(clean_chars)) / max(1, len(clean_chars))) if clean_chars else 0.0
    min_div = thresholds.get("min_char_diversity_ratio", 0.35)

    mash_detected, mash_details = _detect_keyboard_mash(raw_text)
    if mash_detected:
        flags.extend(mash_details)

    if char_diversity >= min_div and not mash_detected and not too_short:
        pts = weights.get("char_diversity_ok", 10)
        current_score += pts
        factors.append({
            "factor": "Character Diversity",
            "points": f"+{pts}",
            "passed": True,
            "details": f"Character diversity {char_diversity:.2f} (above {min_div:.2f} threshold). Natural distribution.",
        })
    else:
        factors.append({
            "factor": "Character Diversity",
            "points": "0",
            "passed": False,
            "details": f"Low character variety ({char_diversity:.2f}) or keyboard mash pattern detected.",
        })
        if not mash_detected and not too_short:
            flags.append("Low character diversity / repetitive letters")

    # 3. Word Repetition Check (e.g. "help help help help")
    max_rep = thresholds.get("max_word_repetition_ratio", 0.60)
    word_counts = Counter(meaningful_tokens)
    top_word_ratio = (max(word_counts.values()) / max(1, len(meaningful_tokens))) if meaningful_tokens else 0.0

    if top_word_ratio <= max_rep and len(meaningful_tokens) >= min_tokens:
        pts = weights.get("no_repetition", 10)
        current_score += pts
        factors.append({
            "factor": "Natural Phrasing (No Spam Repetition)",
            "points": f"+{pts}",
            "passed": True,
            "details": f"Top word represents {top_word_ratio * 100:.0f}% of tokens (below {max_rep * 100:.0f}% ceiling).",
        })
    else:
        factors.append({
            "factor": "Natural Phrasing (No Spam Repetition)",
            "points": "0",
            "passed": False,
            "details": f"Excessive single-word repetition detected ({top_word_ratio * 100:.0f}%).",
        })
        if len(meaningful_tokens) >= min_tokens:
            flags.append("High single-word repetition spam")

    # 4. Dictionary Word Ratio Check
    dict_words = _get_dictionary_words()
    alpha_tokens = [t for t in meaningful_tokens if t.isalpha()]
    recognized_words = [t for t in alpha_tokens if t in dict_words or t in campus_keywords]
    dict_ratio = (len(recognized_words) / max(1, len(alpha_tokens))) if alpha_tokens else 0.0
    min_dict_ratio = thresholds.get("min_dictionary_word_ratio", 0.40)

    if dict_ratio >= min_dict_ratio and len(alpha_tokens) >= 2 and not too_short:
        pts = weights.get("dictionary_words_ok", 15)
        current_score += pts
        factors.append({
            "factor": "Valid English / Campus Vocabulary",
            "points": f"+{pts}",
            "passed": True,
            "details": f"{dict_ratio * 100:.0f}% recognized vocabulary ({len(recognized_words)}/{len(alpha_tokens)} words).",
        })
    else:
        factors.append({
            "factor": "Valid English / Campus Vocabulary",
            "points": "0",
            "passed": False,
            "details": f"Only {dict_ratio * 100:.0f}% recognized words (below {min_dict_ratio * 100:.0f}% floor).",
        })
        if not too_short:
            flags.append("Unrecognized vocabulary / potential gibberish")

    # 5. Campus Context & Relevance Keywords
    matched_campus = [k for k in campus_keywords if re.search(rf"\b{re.escape(k)}\b", lower_text)]
    if matched_campus:
        pts = weights.get("campus_relevance", 20)
        current_score += pts
        factors.append({
            "factor": "Campus Domain Relevance",
            "points": f"+{pts}",
            "passed": True,
            "details": f"Matched campus operational keywords: {', '.join(matched_campus[:5])}.",
        })
    else:
        factors.append({
            "factor": "Campus Domain Relevance",
            "points": "0",
            "passed": False,
            "details": "No campus operational context keywords found in description.",
        })
        flags.append("Lacks specific campus operational keywords")

    # 6. Classifier Confidence Signal
    conf_floor = thresholds.get("classifier_confidence_floor", 0.25)
    if classifier_confidence is not None:
        if classifier_confidence >= conf_floor and not too_short:
            pts = weights.get("classifier_confidence", 15)
            current_score += pts
            factors.append({
                "factor": "ML Model Confidence Alignment",
                "points": f"+{pts}",
                "passed": True,
                "details": f"Classifier predicted category with {classifier_confidence * 100:.1f}% confidence (above {conf_floor * 100:.0f}% floor).",
            })
        else:
            factors.append({
                "factor": "ML Model Confidence Alignment",
                "points": "0",
                "passed": False,
                "details": f"Classifier confidence low ({classifier_confidence * 100:.1f}%). Text is ambiguous to model.",
            })
    else:
        if not too_short:
            current_score += 10
            factors.append({
                "factor": "ML Model Confidence Alignment",
                "points": "+10",
                "passed": True,
                "details": "Classifier confidence not provided; assigned default neutral weight.",
            })

    # 7. Spam / Dummy Submission Pattern Match
    exact_spam_hit = lower_text in spam_patterns
    # Check if multiple tokens are spam patterns or text consists mainly of dummy words
    matched_spam_patterns = [p for p in spam_patterns if p in lower_text]
    spam_tokens = [t for t in meaningful_tokens if t in spam_patterns]
    spam_token_ratio = (len(spam_tokens) / max(1, len(meaningful_tokens))) if meaningful_tokens else 0.0

    is_spam_dummy = (
        exact_spam_hit
        or spam_token_ratio >= 0.40
        or (len(spam_tokens) >= 2 and len(meaningful_tokens) <= 6)
        or any(p in lower_text for p in ["lorem ipsum", "foo bar", "test 123", "dummy placeholder", "sample text"])
    )

    if not is_spam_dummy:
        pts = weights.get("no_spam_pattern", 10)
        current_score += pts
        factors.append({
            "factor": "Spam / Dummy Filter",
            "points": f"+{pts}",
            "passed": True,
            "details": "No known dummy or placeholder test strings detected.",
        })
    else:
        pen = weights.get("spam_pattern_penalty", -30)
        current_score += pen
        factors.append({
            "factor": "Spam / Dummy Filter",
            "points": f"{pen}",
            "passed": False,
            "details": f"Matches dummy/test submission pattern ({', '.join(spam_tokens or matched_spam_patterns[:3])}).",
        })
        flags.append("Identified as dummy/test submission pattern")

    # 8. Profanity & Abusive Language Penalty
    matched_profanities = [p for p in profanity_keywords if re.search(rf"\b{re.escape(p)}\b", lower_text)]
    if matched_profanities:
        pen = weights.get("no_profanity_penalty", -30)
        current_score += pen
        factors.append({
            "factor": "Respectful Language & Conduct",
            "points": f"{pen}",
            "passed": False,
            "details": f"Profane or abusive language detected ({', '.join(matched_profanities)}).",
        })
        flags.append(f"Abusive / profane language detected ({', '.join(matched_profanities)})")
    else:
        factors.append({
            "factor": "Respectful Language & Conduct",
            "points": "0",
            "passed": True,
            "details": "No inappropriate profanity or abusive language found.",
        })

    # 9. Threat & Intimidation Detection (Security Critical)
    matched_threats = [t for t in threat_keywords if t in lower_text]
    if matched_threats:
        is_threat = True
        pen = weights.get("threat_penalty", -60)
        current_score += pen
        factors.append({
            "factor": "⚠️ Security Threat / Intimidation Check",
            "points": f"{pen}",
            "passed": False,
            "details": f"Direct threat / intimidation language detected: '{matched_threats[0]}'.",
        })
        flags.append(f"🚨 SECURITY ALERT: Threatening language detected ({', '.join(matched_threats)})")

    # Clamp score between 0 and 100
    final_score = max(0, min(100, current_score))

    # Verdict Determination
    floor_genuine = thresholds.get("verdict_genuine_floor", 60)
    floor_review = thresholds.get("verdict_review_floor", 35)

    if is_threat or too_short or is_spam_dummy or (mash_detected and not matched_campus) or final_score < floor_review:
        verdict = "SPAM_REJECTED"
        is_genuine = False
        recommended_status = "Spam/Blocked"
        if is_threat:
            summary_reason = "🚨 Blocked: Security threat / hostile intimidation detected."
        elif too_short:
            summary_reason = "🚫 Blocked: Insufficient detail (< 15 characters). Please provide a descriptive complaint."
        elif is_spam_dummy:
            summary_reason = "🚫 Blocked: Detected as dummy/test or placeholder text."
        else:
            summary_reason = "🚫 Blocked: Frivolous/mischief/spam submission filtered out from main queue."
    elif final_score >= floor_genuine:
        verdict = "GENUINE"
        is_genuine = True
        recommended_status = "Pending"
        summary_reason = "✅ Verified: Genuine student complaint with clear context and operational relevance."
    else:
        verdict = "NEEDS_REVIEW"
        is_genuine = False
        recommended_status = "Under Review"
        summary_reason = "⚠️ Ambiguous: Submission has partial context or edge quality; placed in review queue."

    return {
        "score": final_score,
        "verdict": verdict,
        "is_genuine": is_genuine,
        "is_threat": is_threat,
        "recommended_status": recommended_status,
        "flags": flags,
        "factors": factors,
        "summary_reason": summary_reason,
        "matched_campus_keywords": matched_campus,
    }
