"""
CampusPulse - Sentiment and Priority Engine
Implements explainable sentiment estimation (via VADER) and transparent,
rule-based priority scoring with full factor breakdown.
"""

from typing import Any, Dict, List, Tuple


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Includes a built-in fallback lexicon if NLTK is unavailable.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neu": 1.0,
            "neg": 0.0,
            "label": "Neutral",
            "method": "Empty input default",
        }

    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(cleaned)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "compound": round(compound, 4),
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4),
            "label": label,
            "method": "VADER Lexicon",
        }
    except Exception:
        # Transparent rule-based fallback
        negative_words = {
            "broken", "terrible", "worst", "awful", "bad", "horrible", "leak",
            "failing", "unacceptable", "dirty", "smelly", "disgusting", "angry",
            "frustrated", "suffering", "useless", "danger", "hazardous", "shock"
        }
        positive_words = {
            "good", "great", "excellent", "working", "resolved", "fine", "helpful", "thanks"
        }

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
            "pos": 0.0,
            "neu": 1.0,
            "neg": 0.0,
            "label": label,
            "method": "Rule-based fallback lexicon",
        }


# Category intrinsic baseline urgency points (scale 0-25)
CATEGORY_BASELINES: Dict[str, Tuple[int, str]] = {
    # Dataset native categories
    "Infrastructure": (20, "Infrastructure and maintenance issues carry physical & safety implications"),
    "Technical": (15, "Technical disruptions impact labs, connectivity, and digital learning"),
    "Academic": (15, "Academic disruptions impact coursework, evaluations, and grading"),
    "Finance": (15, "Financial discrepancies impact registrations and administrative clearance"),
    "Administrative": (10, "Administrative processes and operational coordination"),
    # Standard campus taxonomy aliases
    "Electricity and Infrastructure": (20, "Electrical/infrastructure issues carry higher physical & safety risks"),
    "Water and Sanitation": (20, "Water supply & hygiene concerns directly impact student health"),
    "Hostel and Accommodation": (15, "Living condition issues affect daily student welfare and rest"),
    "Academics and Administration": (15, "Administrative blockages may disrupt exams or course registration"),
    "Internet and Wi-Fi": (15, "Connectivity disruptions impact study routines and submissions"),
    "Other": (10, "General inquiry or uncategorized operational issue"),
}

# Critical safety / hazard keywords (+30 points)
SAFETY_KEYWORDS: Dict[str, str] = {
    "spark": "Electrical sparking detected",
    "sparks": "Electrical sparking detected",
    "fire": "Fire or burn hazard mentioned",
    "smoke": "Smoke observed",
    "shock": "Electric shock risk identified",
    "short circuit": "Short circuit hazard",
    "flooding": "Water flooding reported",
    "collapse": "Structural collapse risk",
    "collapsed": "Structural collapse risk",
    "danger": "Explicit safety danger flagged",
    "hazard": "Hazardous condition reported",
    "hazardous": "Hazardous condition reported",
    "gas leak": "Gas leak risk",
    "food poisoning": "Food poisoning or severe health illness",
    "vomiting": "Severe gastrointestinal illness reported",
    "contaminated": "Contaminated water/food supply",
    "theft": "Security/theft breach",
    "broken lock": "Hostel room security compromise",
}

# Academic urgency keywords (+15 points)
ACADEMIC_URGENCY_KEYWORDS: Dict[str, str] = {
    "exam": "Upcoming examination deadline constraint",
    "exams": "Upcoming examination deadline constraint",
    "viva": "Imminent viva assessment",
    "deadline": "Time-sensitive assignment or portal deadline",
    "submission": "Project or homework submission affected",
    "placement": "Placement drive or interview affected",
    "interview": "Interview preparation hindered",
}

# Urgent operational keywords (+10 points)
OPERATIONAL_URGENCY_KEYWORDS: Dict[str, str] = {
    "urgent": "Explicitly marked urgent by complainant",
    "immediately": "Requires prompt intervention",
    "emergency": "Student declared emergency condition",
    "critical": "Critical system downtime",
    "blackout": "Complete power blackout",
    "severe": "Severe impact reported",
}


def calculate_priority(
    category: str,
    text: str,
    sentiment_compound: float,
    similar_count: int = 0,
    is_unresolved: bool = True,
) -> Dict[str, Any]:
    """
    Computes an explainable priority score (0 to 100) and priority level
    based on documented, transparent campus rules.
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

    # 2. Hazard & Physical Safety Indicators (+30 max)
    matched_safety = []
    for kw, desc in SAFETY_KEYWORDS.items():
        if kw in text_lower:
            matched_safety.append(f"'{kw}' ({desc})")
    if matched_safety:
        safety_pts = 30
        score += safety_pts
        factors.append({
            "factor": "Physical Safety / Hazard Detected",
            "points": f"+{safety_pts}",
            "reason": f"Keywords: {', '.join(matched_safety[:3])}",
        })

    # 3. Academic Deadlines & Constraints (+15 max)
    matched_academic = []
    for kw, desc in ACADEMIC_URGENCY_KEYWORDS.items():
        if kw in text_lower:
            matched_academic.append(f"'{kw}' ({desc})")
    if matched_academic:
        acad_pts = 15
        score += acad_pts
        factors.append({
            "factor": "Academic Deadline Constraint",
            "points": f"+{acad_pts}",
            "reason": f"Keywords: {', '.join(matched_academic[:2])}",
        })

    # 4. Complainant Urgency Markers (+10 max)
    matched_urgency = []
    for kw, desc in OPERATIONAL_URGENCY_KEYWORDS.items():
        if kw in text_lower:
            matched_urgency.append(f"'{kw}'")
    if matched_urgency and not matched_safety:  # don't double count if safety already triggered
        urg_pts = 10
        score += urg_pts
        factors.append({
            "factor": "Complainant Urgency Indicator",
            "points": f"+{urg_pts}",
            "reason": f"Expressed urgency terms: {', '.join(matched_urgency[:2])}",
        })

    # 5. Sentiment Frustration Factor (+5 to +10)
    # Note: Kept separate from physical severity
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

    # 6. Recurrence Factor (Cluster / Similar Complaints)
    if similar_count >= 4:
        rec_pts = 20
        score += rec_pts
        factors.append({
            "factor": "Widespread Recurrent Incident",
            "points": f"+{rec_pts}",
            "reason": f"Correlates with {similar_count} similar existing complaints in this block/category",
        })
    elif similar_count >= 2:
        rec_pts = 10
        score += rec_pts
        factors.append({
            "factor": "Recurring Complaint Pattern",
            "points": f"+{rec_pts}",
            "reason": f"Correlates with {similar_count} similar existing complaints in this block/category",
        })
    elif similar_count == 1:
        rec_pts = 5
        score += rec_pts
        factors.append({
            "factor": "Potential Related Issue",
            "points": f"+{rec_pts}",
            "reason": "1 similar existing complaint found in records",
        })

    # 7. Unresolved status boost
    if is_unresolved:
        unres_pts = 5
        score += unres_pts
        factors.append({
            "factor": "Status Unresolved",
            "points": f"+{unres_pts}",
            "reason": "Complaint is currently active/open in ticket queue",
        })

    # Clamp total score between 0 and 100
    final_score = max(5, min(100, score))

    # Map to priority tier
    if final_score >= 80:
        level = "Critical"
    elif final_score >= 60:
        level = "High"
    elif final_score >= 35:
        level = "Medium"
    else:
        level = "Low"

    return {
        "priority_score": final_score,
        "priority_level": level,
        "factors": factors,
        "disclaimer": (
            "Priority score is a rule-based triage heuristic designed to aid ticket prioritization. "
            "It does not replace human inspection or autonomous safety assessment."
        ),
    }
