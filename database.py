"""
CampusPulse - SQLite Local Persistence Layer
Handles storage, retrieval, and status updates for student complaints.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

DEFAULT_DB_PATH = "campuspulse.db"


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a database connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the database schema if it doesn't already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT DEFAULT 'General Campus',
            category TEXT NOT NULL,
            probabilities TEXT,
            sentiment_label TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            priority_score INTEGER NOT NULL,
            priority_level TEXT NOT NULL,
            priority_factors TEXT,
            status TEXT DEFAULT 'Pending',
            submitted_at TEXT NOT NULL,
            duplicate_cluster_id TEXT
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON complaints(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON complaints(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON complaints(priority_level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_submitted ON complaints(submitted_at)")

    conn.commit()
    conn.close()


def generate_complaint_id(db_path: str = DEFAULT_DB_PATH) -> str:
    """Generates a collision-safe sequential complaint identifier, e.g. CP-2026-1042."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(id), 0) FROM complaints")
    max_id = cursor.fetchone()[0]
    conn.close()
    year = datetime.now().year
    return f"CP-{year}-{1001 + max_id}"


def insert_complaint(
    title: str,
    description: str,
    category: str,
    sentiment_label: str,
    sentiment_score: float,
    priority_score: int,
    priority_level: str,
    location: str = "General Campus",
    probabilities: Optional[Dict[str, float]] = None,
    priority_factors: Optional[List[str]] = None,
    status: str = "Pending",
    duplicate_cluster_id: Optional[str] = None,
    submitted_at: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Inserts a new complaint record and returns the assigned complaint ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    complaint_id = generate_complaint_id(db_path)
    if not submitted_at:
        submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    probs_json = json.dumps(probabilities or {})
    factors_json = json.dumps(priority_factors or [])

    cursor.execute(
        """
        INSERT INTO complaints (
            complaint_id, title, description, location, category,
            probabilities, sentiment_label, sentiment_score,
            priority_score, priority_level, priority_factors,
            status, submitted_at, duplicate_cluster_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            complaint_id,
            title.strip(),
            description.strip(),
            location.strip() or "General Campus",
            category,
            probs_json,
            sentiment_label,
            round(sentiment_score, 4),
            int(priority_score),
            priority_level,
            factors_json,
            status,
            submitted_at,
            duplicate_cluster_id,
        ),
    )

    conn.commit()
    conn.close()
    return complaint_id


def get_all_complaints(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves all complaints as a pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM complaints ORDER BY submitted_at DESC, id DESC", conn
    )
    conn.close()
    return df


def get_complaint_by_id(complaint_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetches full details for a specific complaint."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        data = dict(row)
        data["probabilities"] = json.loads(data["probabilities"]) if data.get("probabilities") else {}
        data["priority_factors"] = json.loads(data["priority_factors"]) if data.get("priority_factors") else []
        return data
    return None


def update_complaint_status(complaint_id: str, new_status: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Updates the resolution status of an existing complaint."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE complaints SET status = ? WHERE complaint_id = ?",
        (new_status, complaint_id),
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def get_summary_statistics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Computes summary metrics for the executive dashboard."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM complaints")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Pending'")
    pending = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'In Progress'")
    in_progress = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Under Review'")
    under_review = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'")
    resolved = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE priority_level IN ('High', 'Critical') AND status != 'Resolved'")
    urgent_open = cursor.fetchone()[0]

    conn.close()

    return {
        "total": total,
        "pending": pending,
        "in_progress": in_progress,
        "under_review": under_review,
        "active_open": pending + in_progress + under_review,
        "resolved": resolved,
        "urgent_open": urgent_open,
        "resolution_rate": round((resolved / total * 100), 1) if total > 0 else 0.0,
    }
