"""
Unit tests for cosine similarity duplicate detection engine.
"""

import pandas as pd
import pytest
from similarity import ComplaintSimilarityEngine, evaluate_duplicate_thresholds


@pytest.fixture
def sample_candidates_df():
    return pd.DataFrame([
        {
            "complaint_id": "CP-101",
            "title": "Wi-Fi down on 3rd floor",
            "description": "Campus Wi-Fi is disconnected on 3rd floor hostel",
            "location": "Hostel 3rd floor",
            "category": "Technical",
            "status": "Pending",
        },
        {
            "complaint_id": "CP-102",
            "title": "Water leakage in washroom",
            "description": "Continuous water leaking under the washbasin in bathroom",
            "location": "Washroom 2nd floor",
            "category": "Infrastructure",
            "status": "In Progress",
        },
        {
            "complaint_id": "CP-103",
            "title": "Exam fee pending",
            "description": "Exam fee deducted but portal shows pending status",
            "location": "Admin",
            "category": "Finance",
            "status": "Pending",
        },
    ])


def test_similarity_engine_finds_match(sample_candidates_df):
    engine = ComplaintSimilarityEngine()
    query = "Wi-Fi is not working on 3rd floor of hostel"
    matches = engine.find_similar(query, sample_candidates_df, threshold=0.30, top_k=3)

    assert len(matches) > 0
    top_match = matches[0]
    assert top_match["complaint_id"] == "CP-101"
    assert top_match["similarity_score"] > 0.30
    assert len(top_match["shared_terms"]) > 0


def test_similarity_engine_excludes_self(sample_candidates_df):
    engine = ComplaintSimilarityEngine()
    query = "Campus Wi-Fi is disconnected on 3rd floor hostel"
    matches = engine.find_similar(
        query, sample_candidates_df, threshold=0.30, exclude_id="CP-101"
    )
    for m in matches:
        assert m["complaint_id"] != "CP-101"


def test_similarity_empty_candidates():
    engine = ComplaintSimilarityEngine()
    empty_df = pd.DataFrame()
    matches = engine.find_similar("Any complaint text", empty_df)
    assert matches == []


def test_duplicate_threshold_evaluation():
    pairs_df = pd.DataFrame([
        {"text_1": "WiFi down in library", "text_2": "Library WiFi not working", "is_duplicate": 1},
        {"text_1": "WiFi down in library", "text_2": "Fee receipt pending", "is_duplicate": 0},
    ])
    results_df = evaluate_duplicate_thresholds(pairs_df, thresholds=[0.3, 0.6])
    assert not results_df.empty
    assert "precision" in results_df.columns
    assert "recall" in results_df.columns
    assert "f1_score" in results_df.columns
