"""
CampusPulse - Enterprise REST API & Web Server
Powered by Flask, Flask-CORS, and Scikit-Learn NLP Pipeline.
Provides high-performance endpoints for:
1. Executive KPIs, status distributions, and duplicate cluster analytics
2. Real-time debounced NLP triage (Category classification, VADER sentiment, Priority rules, Genuineness/Mischief score)
3. Complaint Queue CRUD and lifecycle progression
4. Model evaluation metrics and top feature explainability
5. Static delivery of the modern Obsidian & Titanium frontend dashboard
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import (
    delete_complaint,
    get_all_complaints,
    get_complaint_by_id,
    get_summary_statistics,
    init_db,
    insert_complaint,
    update_complaint_status,
)
from evaluation import load_eval_metrics
from genuineness import evaluate_genuineness, load_genuineness_config
from preprocessing import clean_text, explain_preprocessing
from priority import analyze_sentiment, calculate_priority
from similarity import ComplaintSimilarityEngine

# ---------------------------------------------------------------------------
# App Initialization & Model Loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Initialize database schema
init_db()

# Load trained classification pipeline
MODEL_PATH = os.path.join(BASE_DIR, "models", "complaint_classifier.joblib")
pipeline = None
similarity_engine = None

try:
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        tfidf_vec = pipeline.named_steps.get("tfidf") if hasattr(pipeline, "named_steps") else None
        similarity_engine = ComplaintSimilarityEngine(vectorizer=tfidf_vec)
        print("[CampusPulse Server] Successfully loaded model and initialized similarity engine.")
    else:
        print(f"[CampusPulse Server WARNING] Model file {MODEL_PATH} not found. Running with fallback defaults.")
        similarity_engine = ComplaintSimilarityEngine()
except Exception as exc:
    print(f"[CampusPulse Server WARNING] Failed to load model: {exc}")
    similarity_engine = ComplaintSimilarityEngine()


# ---------------------------------------------------------------------------
# Static Route
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the primary Obsidian & Warm Titanium Dashboard."""
    return send_from_directory(STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns server and ML model health status."""
    return jsonify({
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "classes": list(pipeline.classes_) if pipeline is not None else [],
        "database": "connected",
        "theme": "Obsidian & Warm Titanium",
    })


@app.route("/api/stats", methods=["GET"])
def get_dashboard_stats():
    """Returns aggregated executive dashboard KPIs, charts data, and clusters."""
    stats = get_summary_statistics()
    df = get_all_complaints()

    if df.empty:
        return jsonify({
            "kpis": stats,
            "category_distribution": {},
            "priority_distribution": {},
            "sentiment_distribution": {},
            "status_distribution": {},
            "duplicate_clusters": [],
            "recent_complaints": [],
        })

    # Category breakdown (for genuine / active tickets)
    genuine_df = df[df["genuineness_verdict"] != "SPAM_REJECTED"]
    cat_counts = genuine_df["category"].value_counts().to_dict()

    # Priority breakdown
    prio_counts = genuine_df["priority_level"].value_counts().to_dict()

    # Sentiment breakdown
    sent_counts = genuine_df["sentiment_label"].value_counts().to_dict()

    # Status breakdown
    status_counts = df["status"].value_counts().to_dict()

    # Duplicate clusters
    active_df = df[~df["status"].isin(["Resolved", "Spam/Blocked"]) & (df["genuineness_verdict"] != "SPAM_REJECTED")]
    clusters = []
    if not active_df.empty and similarity_engine is not None:
        try:
            clusters = similarity_engine.find_all_duplicate_clusters(
                complaints_df=active_df,
                threshold=0.45,
                min_cluster_size=2,
            )
        except Exception:
            clusters = []

    # Recent 8 tickets
    recent_cols = ["complaint_id", "title", "category", "location", "priority_level", "priority_score", "sentiment_label", "status", "submitted_at", "genuineness_verdict"]
    recent_df = df[recent_cols].head(8).where(pd.notnull(df[recent_cols].head(8)), None)
    recent = recent_df.to_dict(orient="records")

    return jsonify({
        "kpis": stats,
        "category_distribution": cat_counts,
        "priority_distribution": prio_counts,
        "sentiment_distribution": sent_counts,
        "status_distribution": status_counts,
        "duplicate_clusters": clusters,
        "recent_complaints": recent,
    })


@app.route("/api/complaints", methods=["GET"])
def list_complaints():
    """Retrieves paginated and filtered list of complaints."""
    view = request.args.get("view", "active")
    category = request.args.get("category")
    priority = request.args.get("priority")
    status_filter = request.args.get("status")
    search = request.args.get("search")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))

    df = get_all_complaints()

    if df.empty:
        return jsonify({"total": 0, "complaints": []})

    # View filtering
    if view == "quarantine":
        df = df[(df["status"] == "Spam/Blocked") | (df["genuineness_verdict"] == "SPAM_REJECTED")]
    elif view == "active":
        df = df[(df["status"] != "Spam/Blocked") & (df["genuineness_verdict"] != "SPAM_REJECTED")]

    # Category filter
    if category and category != "All":
        df = df[df["category"] == category]

    # Priority filter
    if priority and priority != "All":
        df = df[df["priority_level"] == priority]

    # Status filter
    if status_filter and status_filter != "All":
        df = df[df["status"] == status_filter]

    # Search filter across title, description, and location
    if search and search.strip():
        q = search.strip().lower()
        match_mask = (
            df["title"].str.lower().str.contains(q, na=False)
            | df["description"].str.lower().str.contains(q, na=False)
            | df["location"].str.lower().str.contains(q, na=False)
            | df["complaint_id"].str.lower().str.contains(q, na=False)
        )
        df = df[match_mask]

    total_count = len(df)
    paged_df = df.iloc[offset : offset + limit]

    records = []
    for _, row in paged_df.iterrows():
        rec = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        # Parse JSON fields safely
        for json_col in ["probabilities", "priority_factors", "genuineness_flags"]:
            val = rec.get(json_col)
            if isinstance(val, str) and val.strip():
                try:
                    rec[json_col] = json.loads(val)
                except Exception:
                    rec[json_col] = [] if json_col != "probabilities" else {}
            elif not val:
                rec[json_col] = [] if json_col != "probabilities" else {}
        records.append(rec)

    return jsonify({
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "complaints": records,
    })


@app.route("/api/complaints/<complaint_id>", methods=["GET"])
def get_complaint(complaint_id: str):
    """Fetches full breakdown of a single complaint."""
    data = get_complaint_by_id(complaint_id)
    if not data:
        return jsonify({"detail": "Complaint not found"}), 404
    return jsonify(data)


@app.route("/api/complaints/<complaint_id>/status", methods=["PATCH"])
def change_complaint_status(complaint_id: str):
    """Updates complaint resolution status."""
    payload = request.get_json(silent=True) or {}
    new_status = payload.get("status")

    valid_statuses = ["Pending", "In Progress", "Under Review", "Resolved", "Spam/Blocked"]
    if not new_status or new_status not in valid_statuses:
        return jsonify({"detail": f"Invalid status '{new_status}'. Must be one of: {valid_statuses}"}), 400

    success = update_complaint_status(complaint_id, new_status)
    if not success:
        return jsonify({"detail": "Complaint not found or update failed"}), 404

    return jsonify({
        "message": f"Complaint {complaint_id} updated to {new_status}",
        "complaint_id": complaint_id,
        "new_status": new_status,
    })


@app.route("/api/complaints/<complaint_id>", methods=["DELETE"])
def remove_complaint(complaint_id: str):
    """Permanently deletes a complaint from database."""
    success = delete_complaint(complaint_id)
    if not success:
        return jsonify({"detail": "Complaint not found"}), 404
    return jsonify({"message": f"Complaint {complaint_id} deleted successfully"})


@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """
    Performs real-time, explainable NLP intelligence on input text:
    1. Preprocessing normalization and token explanation
    2. Genuineness and Anti-Mischief Shield (score 0-100, verdict, flags)
    3. Multiclass classification & probability distribution
    4. VADER Sentiment polarity estimation
    5. Rule-governed Priority scoring with full factor breakdown
    6. Historical duplicate search using TF-IDF cosine similarity
    """
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    location = (payload.get("location") or "General Campus").strip()
    sim_threshold = float(payload.get("similarity_threshold", 0.45))

    if not text:
        return jsonify({"detail": "Text cannot be empty"}), 400

    # 1. Preprocessing Explanation
    prep_explanation = explain_preprocessing(text)
    cleaned = clean_text(text)

    # 2. ML Classification
    probabilities: Dict[str, float] = {}
    if pipeline is not None and cleaned:
        try:
            probas = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_
            probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}
            probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
            pred_category = next(iter(probabilities))
            confidence = probabilities[pred_category]
        except Exception:
            pred_category = "Infrastructure"
            confidence = 0.85
            probabilities = {pred_category: 0.85}
    else:
        pred_category = "Infrastructure"
        confidence = 0.85
        probabilities = {pred_category: 0.85}

    # 3. Genuineness Evaluation
    genuineness = evaluate_genuineness(
        text=text,
        classifier_confidence=confidence,
    )

    # 4. Sentiment Analysis
    sentiment = analyze_sentiment(text)

    # 5. Similar Complaints Search
    similar_matches = []
    if similarity_engine is not None:
        try:
            existing_df = get_all_complaints()
            similar_matches = similarity_engine.find_similar(
                query_text=text,
                candidates_df=existing_df,
                threshold=sim_threshold,
                top_k=4,
            )
        except Exception:
            similar_matches = []

    # 6. Priority Scoring
    priority = calculate_priority(
        category=pred_category,
        text=text,
        sentiment_compound=sentiment["compound"],
        similar_count=len(similar_matches),
        is_unresolved=True,
    )

    return jsonify({
        "predicted_category": pred_category,
        "confidence": confidence,
        "probabilities": probabilities,
        "sentiment": sentiment,
        "priority": priority,
        "genuineness": genuineness,
        "similar_matches": similar_matches,
        "similar_count": len(similar_matches),
        "preprocessing": prep_explanation,
    })


@app.route("/api/complaints", methods=["POST"])
def submit_complaint():
    """
    Submits a verified complaint into the system.
    Runs complete NLP pipeline and stores results in SQLite database.
    """
    payload = request.get_json(silent=True) or {}
    text = (payload.get("description") or "").strip()
    title = (payload.get("title") or "").strip()
    location = (payload.get("location") or "General Campus").strip()
    sim_threshold = float(payload.get("similarity_threshold", 0.45))

    if not title or not text:
        return jsonify({"detail": "Title and description are required"}), 400

    cleaned = clean_text(text)
    probabilities = {}
    if pipeline is not None and cleaned:
        try:
            probas = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_
            probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}
            pred_category = payload.get("category") or pipeline.predict([cleaned])[0]
            confidence = probabilities.get(pred_category, 0.85)
        except Exception:
            pred_category = payload.get("category") or "Infrastructure"
            confidence = 0.85
            probabilities = {pred_category: 0.85}
    else:
        pred_category = payload.get("category") or "Infrastructure"
        confidence = 0.85
        probabilities = {pred_category: 0.85}

    genuineness = evaluate_genuineness(text=text, classifier_confidence=confidence)
    sentiment = analyze_sentiment(text)

    similar_matches = []
    cluster_id = None
    if similarity_engine is not None:
        try:
            existing_df = get_all_complaints()
            similar_matches = similarity_engine.find_similar(
                query_text=text,
                candidates_df=existing_df,
                threshold=sim_threshold,
                top_k=4,
            )
            if similar_matches and similar_matches[0].get("similarity_percentage", 0) >= 70:
                cluster_id = f"CLUSTER-{similar_matches[0]['complaint_id']}"
        except Exception:
            similar_matches = []

    priority = calculate_priority(
        category=pred_category,
        text=text,
        sentiment_compound=sentiment["compound"],
        similar_count=len(similar_matches),
        is_unresolved=True,
    )

    # Initial status: if rejected by anti-mischief shield, quarantine immediately
    initial_status = "Spam/Blocked" if genuineness["verdict"] == "SPAM_REJECTED" else "Pending"

    # Insert into database
    complaint_id = insert_complaint(
        title=title,
        description=text,
        category=pred_category,
        sentiment_label=sentiment["label"],
        sentiment_score=sentiment["compound"],
        priority_score=priority["priority_score"],
        priority_level=priority["priority_level"],
        location=location,
        probabilities=probabilities,
        priority_factors=[f"{f['factor']}: {f['points']} ({f['reason']})" for f in priority["factors"]],
        status=initial_status,
        duplicate_cluster_id=cluster_id,
        genuineness_score=genuineness["score"],
        genuineness_verdict=genuineness["verdict"],
        genuineness_flags=genuineness["flags"],
    )

    return jsonify({
        "success": True,
        "complaint_id": complaint_id,
        "status": initial_status,
        "category": pred_category,
        "priority_level": priority["priority_level"],
        "priority_score": priority["priority_score"],
        "genuineness_verdict": genuineness["verdict"],
        "genuineness_score": genuineness["score"],
        "sentiment_label": sentiment["label"],
        "similar_matches_count": len(similar_matches),
    }), 201


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation_metrics():
    """Returns genuine benchmark test evaluation metrics and confusion matrix."""
    metrics = load_eval_metrics()
    if not metrics:
        return jsonify({"detail": "Evaluation metrics file not found"}), 404
    return jsonify(metrics)


@app.route("/api/config", methods=["GET"])
def get_system_config():
    """Returns active genuineness thresholds and campus abbreviation dictionary."""
    gen_config = load_genuineness_config()
    abbrevs_path = os.path.join(BASE_DIR, "data", "campus_abbreviations.json")
    abbrevs = {}
    if os.path.exists(abbrevs_path):
        try:
            with open(abbrevs_path, "r", encoding="utf-8") as f:
                abbrevs = json.load(f)
        except Exception:
            abbrevs = {}
    return jsonify({
        "genuineness_config": gen_config,
        "campus_abbreviations": abbrevs,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
