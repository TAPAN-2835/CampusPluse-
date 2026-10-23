"""
Unit tests for SQLite database persistence layer.
"""

import os
import pytest
from database import (
    init_db,
    insert_complaint,
    get_all_complaints,
    get_complaint_by_id,
    update_complaint_status,
    delete_complaint,
    get_summary_statistics,
    generate_complaint_id,
)

TEST_DB = "test_campuspulse.db"


@pytest.fixture(autouse=True)
def cleanup_test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_insert_and_retrieve_complaint():
    init_db(TEST_DB)
    cid = insert_complaint(
        title="Test Wi-Fi Issue",
        description="Wi-Fi is not connecting in lab 3",
        category="Technical",
        sentiment_label="Negative",
        sentiment_score=-0.45,
        priority_score=65,
        priority_level="High",
        location="Lab 3",
        genuineness_score=85,
        genuineness_verdict="GENUINE",
        genuineness_flags=["Verified vocabulary"],
        db_path=TEST_DB,
    )
    assert cid.startswith("CP-")
    item = get_complaint_by_id(cid, db_path=TEST_DB)
    assert item is not None
    assert item["title"] == "Test Wi-Fi Issue"
    assert item["category"] == "Technical"
    assert item["status"] == "Pending"
    assert item["genuineness_score"] == 85
    assert item["genuineness_verdict"] == "GENUINE"
    assert "Verified vocabulary" in item["genuineness_flags"]


def test_status_update_and_summary_stats():
    init_db(TEST_DB)
    cid = insert_complaint(
        title="Test Sparking Switch",
        description="Dangerous electrical spark in room 101",
        category="Infrastructure",
        sentiment_label="Negative",
        sentiment_score=-0.7,
        priority_score=85,
        priority_level="Critical",
        db_path=TEST_DB,
    )
    stats1 = get_summary_statistics(TEST_DB)
    assert stats1["total"] == 1
    assert stats1["pending"] == 1
    assert stats1["urgent_open"] == 1

    # Update to Under Review
    ok = update_complaint_status(cid, "Under Review", db_path=TEST_DB)
    assert ok is True
    stats2 = get_summary_statistics(TEST_DB)
    assert stats2["under_review"] == 1
    assert stats2["pending"] == 0

    # Update to Resolved
    ok = update_complaint_status(cid, "Resolved", db_path=TEST_DB)
    assert ok is True
    stats3 = get_summary_statistics(TEST_DB)
    assert stats3["resolved"] == 1
    assert stats3["urgent_open"] == 0
    assert stats3["resolution_rate"] == 100.0


def test_spam_quarantine_and_delete():
    init_db(TEST_DB)
    cid = insert_complaint(
        title="Spam Test",
        description="asdfghjkl qwertyuiop",
        category="Technical",
        sentiment_label="Neutral",
        sentiment_score=0.0,
        priority_score=10,
        priority_level="Low",
        status="Spam/Blocked",
        genuineness_score=15,
        genuineness_verdict="SPAM_REJECTED",
        genuineness_flags=["Keyboard row sequence detected"],
        db_path=TEST_DB,
    )
    stats = get_summary_statistics(TEST_DB)
    assert stats["spam_blocked"] == 1
    assert stats["active_open"] == 0

    # Delete spam
    deleted = delete_complaint(cid, db_path=TEST_DB)
    assert deleted is True
    assert get_complaint_by_id(cid, db_path=TEST_DB) is None


def test_sequential_id_generation():
    init_db(TEST_DB)
    id1 = generate_complaint_id(TEST_DB)
    insert_complaint(
        title="T1", description="D1", category="Technical",
        sentiment_label="Neutral", sentiment_score=0.0,
        priority_score=20, priority_level="Low", db_path=TEST_DB
    )
    id2 = generate_complaint_id(TEST_DB)
    assert id1 != id2
