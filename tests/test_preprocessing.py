"""
Unit tests for text preprocessing module.
"""

import pytest
from preprocessing import clean_text, expand_campus_abbreviations, explain_preprocessing, get_stopwords


def test_clean_text_normal():
    raw = "The campus Wi-Fi on 3rd floor is NOT working properly! Please help asap."
    cleaned = clean_text(raw)
    assert "wifi" in cleaned
    assert "not" in cleaned  # Negation must be preserved
    assert "!" not in cleaned
    assert "asap" not in cleaned or "possible" in cleaned


def test_clean_text_edge_cases():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""
    assert clean_text("1234567890") == ""
    assert clean_text("!@#$%^&*()") == ""


def test_abbreviation_expansion():
    text = "prof from cse dept said ac is broken in lab"
    expanded = expand_campus_abbreviations(text)
    assert "professor" in expanded
    assert "department" in expanded
    assert "air conditioner" in expanded
    assert "laboratory" in expanded


def test_negation_retention():
    stopwords = get_stopwords()
    assert "not" not in stopwords
    assert "no" not in stopwords
    assert "never" not in stopwords


def test_explain_preprocessing():
    raw = "Wi-Fi is not connecting in hostel room 202"
    result = explain_preprocessing(raw)
    assert result["raw_text"] == raw
    assert "wifi" in result["abbreviation_expanded"]
    assert isinstance(result["tokens_extracted"], list)
    assert "not" in result["retained_tokens"]
    assert result["token_count"] > 0


def test_negation_contractions_expansion():
    # Negative contractions must retain 'not' or 'cannot'
    samples = [
        ("The hostel Wi-Fi doesn't connect", "not"),
        ("Projector isn't turning on in hall", "not"),
        ("Water won't dispense from cooler", "not"),
        ("Students can't access exam portal", "cannot"),
        ("The classroom projectors doesnt work well", "not"),
    ]
    for text, expected in samples:
        cleaned = clean_text(text)
        assert expected in cleaned, f"Expected '{expected}' in '{cleaned}' for input '{text}'"

