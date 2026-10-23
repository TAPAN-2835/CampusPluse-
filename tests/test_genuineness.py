"""
Unit tests for complaint genuineness and mischief detection engine.
"""

import pytest
from genuineness import evaluate_genuineness, load_genuineness_config


def test_config_loads_properly():
    config = load_genuineness_config()
    assert "thresholds" in config
    assert "spam_patterns" in config
    assert "threat_keywords" in config
    assert "profanity_keywords" in config
    assert "campus_relevance_keywords" in config
    assert "scoring_weights" in config


def test_genuine_complaint():
    desc = "The water cooler in 4th floor mechanical block is leaking dirty water and smells bad. Please send someone to fix it."
    res = evaluate_genuineness(desc, classifier_confidence=0.88)
    assert res["is_genuine"] is True
    assert res["verdict"] == "GENUINE"
    assert res["score"] >= 60
    assert res["is_threat"] is False
    assert res["recommended_status"] == "Pending"
    assert len(res["factors"]) > 0


def test_keyboard_mashing_gibberish():
    desc = "asdfghjkl qwertyuiop zxcvbnm asdf"
    res = evaluate_genuineness(desc)
    assert res["is_genuine"] is False
    assert res["verdict"] == "SPAM_REJECTED"
    assert res["recommended_status"] == "Spam/Blocked"
    assert any("Keyboard" in f or "consonant" in f or "diversity" in f for f in res["flags"])


def test_too_short_description():
    desc = "fix it"
    res = evaluate_genuineness(desc)
    assert res["is_genuine"] is False
    assert res["verdict"] == "SPAM_REJECTED"
    assert any("too brief" in f for f in res["flags"])


def test_dummy_submission():
    desc = "test testing dummy placeholder"
    res = evaluate_genuineness(desc)
    assert res["is_genuine"] is False
    assert res["verdict"] == "SPAM_REJECTED"
    assert any("dummy" in f or "test" in f for f in res["flags"])


def test_threat_detection():
    desc = "I will kill the professor if my assignment marks are not increased immediately"
    res = evaluate_genuineness(desc)
    assert res["is_threat"] is True
    assert res["is_genuine"] is False
    assert res["verdict"] == "SPAM_REJECTED"
    assert any("SECURITY ALERT" in f or "Threatening" in f for f in res["flags"])


def test_profanity_penalty():
    desc = "Fuck the stupid staff they never turn on the generator when power goes out"
    res = evaluate_genuineness(desc)
    assert any("profane" in f or "Profane" in f for f in res["flags"])
    # Score should suffer from profanity penalty
    assert res["score"] < 60


def test_repetition_spam():
    desc = "wifi wifi wifi wifi wifi wifi wifi wifi wifi wifi wifi wifi wifi wifi"
    res = evaluate_genuineness(desc)
    assert any("repetition" in f for f in res["flags"])
