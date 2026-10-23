"""
CampusPulse - Sentiment and Priority Engine
Implements explainable sentiment estimation (via VADER) and transparent,
rule-based priority scoring with full factor breakdown.

Config-driven design:
- All keyword lists (safety, academic urgency, operational urgency)
  and category baselines are loaded from data/priority_keywords.json.
- Add or modify keywords there without touching this file.
"""

import json
import os
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Load priority keywords from external JSON config
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_KEYWORDS_PATH = os.path.join(_DATA_DIR, "priority_keywords.json")


def _load_priority_config() -> Dict[str, Any]:
    """Loads priority_keywords.json. Returns empty dict on failure."""
    try:
        with open(_KEYWORDS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"[CampusPulse WARNING] priority_keywords.json not found at {_KEYWORDS_PATH}. "
            "Priority engine will use hardcoded fallback defaults."
        )
        return {}
    except (json.JSONDecodeError, Exception) as e:
        print(f"[CampusPulse WARNING] Failed to load priority_keywords.json: {e}.")
        return {}


_cfg = _load_priority_config()


def _strip_comments(d: dict) -> dict:
    """Removes internal _comment keys from a loaded JSON dict."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Category baselines — loaded from JSON
# Format: { "CategoryName": [score, "reason string"] }
# Fallback mirrors the original hardcoded values.
# ---------------------------------------------------------------------------
_raw_baselines = _strip_comments(_cfg.get("category_baselines", {}))
CATEGORY_BASELINES: Dict[str, Tuple[int, str]] = {
    cat: (int(vals[0]), vals[1])
    for cat, vals in _raw_baselines.items()
} if _raw_baselines else {
    "Infrastructure":               (20, "Infrastructure and maintenance issues carry physical & safety implications"),
    "Technical":                    (15, "Technical disruptions impact labs, connectivity, and digital learning"),
    "Academic":                     (15, "Academic disruptions impact coursework, evaluations, and grading"),
    "Finance":                      (15, "Financial discrepancies impact registrations and administrative clearance"),
    "Administrative":               (10, "Administrative processes and operational coordination"),
    "Electricity and Infrastructure":(20, "Electrical/infrastructure issues carry higher physical & safety risks"),
    "Water and Sanitation":         (20, "Water supply & hygiene concerns directly impact student health"),
    "Hostel and Accommodation":     (15, "Living condition issues affect daily student welfare and rest"),
    "Academics and Administration": (15, "Administrative blockages may disrupt exams or course registration"),
    "Internet and Wi-Fi":           (15, "Connectivity disruptions impact study routines and submissions"),
    "Other":                        (10, "General inquiry or uncategorized operational issue"),
}

# Critical safety / hazard keywords — +30 points
SAFETY_KEYWORDS: Dict[str, str] = _strip_comments(
    _cfg.get("safety_keywords", {})
) or {
    "spark": "Electrical sparking detected",
    "fire": "Fire or burn hazard mentioned",
    "smoke": "Smoke observed",
    "shock": "Electric shock risk identified",
    "flooding": "Water flooding reported",
    "collapse": "Structural collapse risk",
    "danger": "Explicit safety danger flagged",
    "hazard": "Hazardous condition reported",
    "gas leak": "Gas leak risk",
    "food poisoning": "Food poisoning or severe health illness",
    "contaminated": "Contaminated water/food supply",
    "theft": "Security/theft breach",
}

# Academic deadline keywords — +15 points
ACADEMIC_URGENCY_KEYWORDS: Dict[str, str] = _strip_comments(
    _cfg.get("academic_urgency_keywords", {})
) or {
    "exam": "Upcoming examination deadline constraint",
    "viva": "Imminent viva assessment",
    "deadline": "Time-sensitive assignment or portal deadline",
    "submission": "Project or homework submission affected",
    "placement": "Placement drive or interview affected",
    "interview": "Interview preparation hindered",
}

# Complainant urgency markers — +10 points
OPERATIONAL_URGENCY_KEYWORDS: Dict[str, str] = _strip_comments(
    _cfg.get("operational_urgency_keywords", {})
) or {
    "urgent": "Explicitly marked urgent by complainant",
    "immediately": "Requires prompt intervention",
    "emergency": "Student declared emergency condition",
    "critical": "Critical system downtime",
    "blackout": "Complete power blackout",
    "severe": "Severe impact reported",
}

# Sentiment fallback lexicons (used when NLTK VADER is unavailable)
_SENTIMENT_FALLBACK_NEG: List[str] = _cfg.get("sentiment_fallback_negative", [
    "broken", "terrible", "worst", "awful", "bad", "horrible", "leak",
    "failing", "unacceptable", "dirty", "smelly", "disgusting", "angry",
    "frustrated", "suffering", "useless", "danger", "hazardous", "shock"
])
_SENTIMENT_FALLBACK_POS: List[str] = _cfg.get("sentiment_fallback_positive", [
    "good", "great", "excellent", "working", "resolved", "fine", "helpful", "thanks"
])


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Falls back to a lexicon loaded from priority_keywords.json if NLTK is unavailable.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0,
            "label": "Neutral", "method": "Empty input default",
        }

    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(cleaned)
        compound = scores["compound"]
        label = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")
        return {
            "compound": round(compound, 4),
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4),
            "label": label,
            "method": "VADER Lexicon",
        }

    except Exception:
        # Rule-based fallback using config-loaded lexicons
        negative_words = set(_SENTIMENT_FALLBACK_NEG)
        positive_words = set(_SENTIMENT_FALLBACK_POS)
        tokens = set(cleaned.lower().split())
        neg_count = len(tokens.intersection(negative_words))
        pos_count = len(tokens.intersection(positive_words))

        if neg_count > pos_count:
            compound = max(-1.0, -0.3 * neg_count)
            label = "Negative"
        elif pos_count > neg_count:
            compound = min(1.0, 0.3 * pos_count)
            label = "Positive"
        else:
            compound = 0.0
            label = "Neutral"

        return {
            "compound": round(compound, 4),
            "pos": 0.0, "neu": 1.0, "neg": 0.0,
            "label": label,
            "method": "Rule-based fallback lexicon (priority_keywords.json)",
        }


# ---------------------------------------------------------------------------
# Priority Scoring
# ---------------------------------------------------------------------------
def calculate_priority(
    category: str,
    text: str,
    sentiment_compound: float,
    similar_count: int = 0,
    is_unresolved: bool = True,
) -> Dict[str, Any]:
    """
    Computes an explainable priority score (0–100) and priority level
    based on documented, transparent campus rules loaded from priority_keywords.json.

    Scoring breakdown:
    - Category base urgency       : up to +20 pts
    - Physical safety keywords    : +30 pts
    - Academic deadline keywords  : +15 pts
    - Complainant urgency markers : +10 pts
    - Sentiment distress (VADER)  : +5 to +10 pts
    - Recurrence cluster          : +5 to +20 pts
    - Unresolved status           : +5 pts
    Total clamped to [5, 100].
    """
    score = 0
    factors: List[Dict[str, Any]] = []
    text_lower = (text or "").lower()

    # 1. Base category weight
    base_pts, base_reason = CATEGORY_BASELINES.get(
        category, (10, "Standard campus service baseline")
    )
    score += base_pts
    factors.append({
        "factor": f"Category Base Urgency ({category})",
        "points": f"+{base_pts}",
        "reason": base_reason,
    })

    # 2. Physical Safety Indicators (+30 max)
    matched_safety = [
        f"'{kw}' ({desc})" for kw, desc in SAFETY_KEYWORDS.items() if kw in text_lower
    ]
    if matched_safety:
        safety_pts = 30
        score += safety_pts
        factors.append({
            "factor": "Physical Safety / Hazard Detected",
            "points": f"+{safety_pts}",
            "reason": f"Keywords: {', '.join(matched_safety[:3])}",
        })

    # 3. Academic Deadlines (+15 max)
    matched_academic = [
        f"'{kw}' ({desc})" for kw, desc in ACADEMIC_URGENCY_KEYWORDS.items() if kw in text_lower
    ]
    if matched_academic:
        acad_pts = 15
        score += acad_pts
        factors.append({
            "factor": "Academic Deadline Constraint",
            "points": f"+{acad_pts}",
            "reason": f"Keywords: {', '.join(matched_academic[:2])}",
        })

    # 4. Complainant Urgency Markers (+10 max — skipped if safety already triggered)
    matched_urgency = [
        f"'{kw}'" for kw, _ in OPERATIONAL_URGENCY_KEYWORDS.items() if kw in text_lower
    ]
    if matched_urgency and not matched_safety:
        urg_pts = 10
        score += urg_pts
        factors.append({
            "factor": "Complainant Urgency Indicator",
            "points": f"+{urg_pts}",
            "reason": f"Expressed urgency terms: {', '.join(matched_urgency[:2])}",
        })

    # 5. Sentiment Distress (+5 to +10)
    if sentiment_compound <= -0.5:
        sent_pts = 10
        score += sent_pts
        factors.append({
            "factor": "High Complainant Distress (Sentiment)",
            "points": f"+{sent_pts}",
            "reason": f"VADER compound polarity is strongly negative ({sentiment_compound:.2f})",
        })
    elif sentiment_compound <= -0.1:
        sent_pts = 5
        score += sent_pts
        factors.append({
            "factor": "Moderate Distress (Sentiment)",
            "points": f"+{sent_pts}",
            "reason": f"VADER compound polarity is moderately negative ({sentiment_compound:.2f})",
        })

    # 6. Recurrence Factor
    if similar_count >= 4:
        rec_pts, rec_label = 20, "Widespread Recurrent Incident"
    elif similar_count >= 2:
        rec_pts, rec_label = 10, "Recurring Complaint Pattern"
    elif similar_count == 1:
        rec_pts, rec_label = 5, "Potential Related Issue"
    else:
        rec_pts = 0

    if rec_pts > 0:
        score += rec_pts
        factors.append({
            "factor": rec_label,
            "points": f"+{rec_pts}",
            "reason": f"Correlates with {similar_count} similar existing complaint(s) in records",
        })

    # 7. Unresolved Status Boost
    if is_unresolved:
        unres_pts = 5
        score += unres_pts
        factors.append({
            "factor": "Status Unresolved",
            "points": f"+{unres_pts}",
            "reason": "Complaint is currently active/open in ticket queue",
        })

    final_score = max(5, min(100, score))
    level = (
        "Critical" if final_score >= 80 else
        "High"     if final_score >= 60 else
        "Medium"   if final_score >= 35 else
        "Low"
    )

    return {
        "priority_score": final_score,
        "priority_level": level,
        "factors": factors,
        "disclaimer": (
            "Priority score is a rule-based triage heuristic designed to aid ticket prioritization. "
            "It does not replace human inspection or autonomous safety assessment."
        ),
    }
