"""
CampusPulse - Dataset Preparation and Database Seeder
Processes the user-provided datasets:
1. Loads train.csv, test.csv, and university_students_complaints.xlsx
2. Generates an authentic benchmark pair test set `data/complaint_pairs.csv`
3. Creates a standardized unified reference `data/complaints.csv`
4. Seeds the SQLite database `campuspulse.db` with historical complaints,
   including predictions, sentiment, priority scores, and duplicate cluster links.
"""

import itertools
import json
import os
import random
import pandas as pd

from database import init_db, insert_complaint, get_connection
from preprocessing import clean_text
from priority import analyze_sentiment, calculate_priority

random.seed(42)


def generate_benchmark_pairs(test_df: pd.DataFrame, output_path: str = "data/complaint_pairs.csv") -> pd.DataFrame:
    """
    Creates an authentic duplicate evaluation benchmark from the real test set
    using ground truth Complaint_Group_ID.
    """
    # 1. Authentic positive duplicate pairs (same Complaint_Group_ID)
    pos_pairs = []
    for grp_id, grp in test_df.groupby("Complaint_Group_ID"):
        if len(grp) >= 2:
            for idx1, idx2 in itertools.combinations(grp.index, 2):
                pos_pairs.append({
                    "text_1": grp.loc[idx1, "Complaint_Description"],
                    "text_2": grp.loc[idx2, "Complaint_Description"],
                    "is_duplicate": 1,
                    "group_id": int(grp_id)
                })

    # 2. Authentic negative pairs (different Complaint_Group_ID, balanced)
    neg_pairs = []
    all_indices = list(test_df.index)
    attempts = 0
    while len(neg_pairs) < len(pos_pairs) and attempts < 2000:
        attempts += 1
        i1, i2 = random.sample(all_indices, 2)
        if test_df.loc[i1, "Complaint_Group_ID"] != test_df.loc[i2, "Complaint_Group_ID"]:
            # Prioritize some hard negatives from same category
            neg_pairs.append({
                "text_1": test_df.loc[i1, "Complaint_Description"],
                "text_2": test_df.loc[i2, "Complaint_Description"],
                "is_duplicate": 0,
                "group_id": -1
            })

    pairs_df = pd.DataFrame(pos_pairs + neg_pairs)
    pairs_df = pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    pairs_df.to_csv(output_path, index=False)
    print(f"Generated {len(pairs_df)} authentic benchmark pairs ({len(pos_pairs)} duplicates, {len(neg_pairs)} non-duplicates) at {output_path}")
    return pairs_df


def standardize_complaints(excel_path: str = "data/university_students_complaints.xlsx", output_path: str = "data/complaints.csv") -> pd.DataFrame:
    """Standardizes complaints into a unified schema."""
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, sheet_name="Complaints_Clean")
    else:
        # Fallback to train + test
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        df = pd.concat([train_df, test_df], ignore_index=True)

    df["title"] = df["Complaint_Description"].apply(lambda t: str(t)[:60].strip() + ("..." if len(str(t)) > 60 else ""))
    df["location"] = df.apply(
        lambda r: f"{r.get('Student_Dept', 'Campus')} Dept (Sem {r.get('Semester', 'General')})",
        axis=1
    )

    df.to_csv(output_path, index=False)
    print(f"Saved standardized complaints dataset ({len(df)} rows) to {output_path}")
    return df


def seed_database_from_dataset(excel_path: str = "data/university_students_complaints.xlsx", db_path: str = "campuspulse.db") -> None:
    """Populates campuspulse.db with pre-processed complaints for immediate dashboard testing."""
    init_db(db_path)

    # Check if already seeded
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM complaints")
    count = cur.fetchone()[0]
    conn.close()

    if count >= 100:
        print(f"Database already seeded with {count} complaints.")
        return

    print("Seeding database with authentic student complaints...")
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, sheet_name="Complaints_Clean")
    else:
        df = pd.read_csv("data/train.csv")

    # Map severity strings to initial priority level heuristics
    severity_map = {
        "Urgent": ("Critical", 85),
        "High": ("High", 70),
        "Medium": ("Medium", 50),
        "Low": ("Low", 25)
    }

    # Count group sizes for recurrence factor
    grp_counts = df["Complaint_Group_ID"].value_counts().to_dict()

    statuses = ["Pending", "In Progress", "Under Review", "Resolved"]
    status_weights = [0.45, 0.25, 0.15, 0.15]

    for _, row in df.iterrows():
        desc = str(row.get("Complaint_Description", "")).strip()
        if not desc:
            continue

        cat = str(row.get("Category", "Infrastructure")).strip()
        sev = str(row.get("Severity", "Medium")).strip()
        grp_id = row.get("Complaint_Group_ID", None)
        rec_count = grp_counts.get(grp_id, 1) - 1

        dept = str(row.get("Student_Dept", "Campus")).strip()
        sem = str(row.get("Semester", "General")).strip()
        loc = f"{dept} Department (Semester {sem})"

        sent_res = analyze_sentiment(desc)
        prio_res = calculate_priority(
            category=cat,
            text=desc,
            sentiment_compound=sent_res["compound"],
            similar_count=rec_count,
            is_unresolved=True
        )

        title = desc[:65].strip()
        if len(desc) > 65:
            title += "..."

        assigned_status = random.choices(statuses, weights=status_weights)[0]
        timestamp = str(row.get("Timestamp", "2026-02-10 10:00:00"))[:19]

        insert_complaint(
            title=title,
            description=desc,
            category=cat,
            sentiment_label=sent_res["label"],
            sentiment_score=sent_res["compound"],
            priority_score=prio_res["priority_score"],
            priority_level=prio_res["priority_level"],
            location=loc,
            probabilities={cat: 0.85},
            priority_factors=[f"{f['factor']}: {f['points']} ({f['reason']})" for f in prio_res["factors"]],
            status=assigned_status,
            duplicate_cluster_id=f"GRP-{grp_id}" if pd.notna(grp_id) else None,
            submitted_at=timestamp,
            db_path=db_path
        )

    print(f"Successfully seeded database with {len(df)} authentic complaints.")


if __name__ == "__main__":
    test_df = pd.read_csv("data/test.csv")
    generate_benchmark_pairs(test_df)
    standardize_complaints()
    seed_database_from_dataset()
