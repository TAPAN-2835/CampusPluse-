"""
Unit tests for VADER sentiment analysis and explainable priority scoring.
"""

import pytest
from priority import analyze_sentiment, calculate_priority


def test_sentiment_analysis_negative():
    text = "Drinking water is dirty, contaminated and smelling awful. Students falling sick!"
    result = analyze_sentiment(text)
    assert result["compound"] < -0.05
    assert result["label"] == "Negative"


def test_sentiment_analysis_positive():
    text = "Thank you so much, the technician resolved the AC issue quickly and it is working great."
    result = analyze_sentiment(text)
    assert result["compound"] > 0.05
    assert result["label"] == "Positive"


def test_priority_score_hazard_escalation():
    # Dangerous electrical issue
    hazard_text = "Dangerous electrical spark and smoke coming from switchboard in room 204"
    res = calculate_priority(
        category="Infrastructure",
        text=hazard_text,
        sentiment_compound=-0.6,
        similar_count=1,
        is_unresolved=True,
    )
    assert res["priority_score"] >= 65
    assert res["priority_level"] in ("High", "Critical")

    # Verify factor explainability list
    factors = [f["factor"] for f in res["factors"]]
    assert any("Safety" in f or "Hazard" in f for f in factors)
    assert "disclaimer" in res


def test_priority_routine_request():
    routine_text = "Curtain rod came slightly loose on the wall"
    res = calculate_priority(
        category="Infrastructure",
        text=routine_text,
        sentiment_compound=0.0,
        similar_count=0,
        is_unresolved=True,
    )
    assert res["priority_score"] < 40
    assert res["priority_level"] in ("Low", "Medium")


def test_priority_recurrence_boost():
    text = "WiFi dropping during online test"
    res_single = calculate_priority("Technical", text, 0.0, similar_count=0)
    res_clustered = calculate_priority("Technical", text, 0.0, similar_count=5)
    assert res_clustered["priority_score"] > res_single["priority_score"]
