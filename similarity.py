"""
CampusPulse - Similar and Duplicate Complaint Detection Engine
Uses TF-IDF vectorization and Cosine Similarity to detect related complaints.
Provides transparent similarity scores, shared n-grams, and context explanations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


class ComplaintSimilarityEngine:
    """
    Computes cosine similarity between incoming complaints and historical records
    using sublinear TF-IDF vectorization.
    """

    def __init__(self, vectorizer: Optional[TfidfVectorizer] = None):
        """Initializes with an optional pre-fitted vectorizer or creates an internal one."""
        self.vectorizer = vectorizer

    def _ensure_vectorizer(self, corpus: List[str]) -> TfidfVectorizer:
        """Returns the pre-fitted vectorizer or fits a new one on candidate corpus."""
        if self.vectorizer is not None:
            return self.vectorizer
        # Standalone vectorizer fitting
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_features=4000,
        )
        vec.fit(corpus)
        return vec

    def find_similar(
        self,
        query_text: str,
        candidates_df: pd.DataFrame,
        threshold: float = 0.60,
        top_k: int = 5,
        exclude_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-K most similar complaints in `candidates_df` above `threshold`.

        Parameters:
            query_text: Raw description of the incoming complaint.
            candidates_df: Historical complaints DataFrame.
            threshold: Cosine similarity cutoff (0.0 to 1.0).
            top_k: Maximum number of matches to return.
            exclude_id: Complaint ID to exclude (e.g. self-match).

        Returns:
            List of matching complaint dictionaries with similarity metrics.
        """
        if candidates_df.empty or not query_text.strip():
            return []

        cleaned_query = clean_text(query_text)
        if not cleaned_query:
            return []

        df = candidates_df.copy()
        if exclude_id and "complaint_id" in df.columns:
            df = df[df["complaint_id"] != exclude_id].copy()

        if df.empty:
            return []

        # Prepare candidate cleaned texts
        df["cleaned_text"] = df["description"].fillna("").astype(str).apply(clean_text)
        corpus = [cleaned_query] + df["cleaned_text"].tolist()

        try:
            vec = self._ensure_vectorizer(corpus)
            query_vec = vec.transform([cleaned_query])
            candidate_vecs = vec.transform(df["cleaned_text"].tolist())
            similarities = cosine_similarity(query_vec, candidate_vecs)[0]
        except Exception:
            # Fallback if vocabulary mismatch occurs
            fallback_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            all_vecs = fallback_vec.fit_transform(corpus)
            query_vec = all_vecs[0]
            candidate_vecs = all_vecs[1:]
            similarities = cosine_similarity(query_vec, candidate_vecs)[0]

        df["similarity_score"] = similarities

        # Filter by threshold
        matches = df[df["similarity_score"] >= threshold].copy()
        matches = matches.sort_values(by="similarity_score", ascending=False).head(top_k)

        query_tokens = set(cleaned_query.split())
        results: List[Dict[str, Any]] = []

        for _, row in matches.iterrows():
            cand_tokens = set(row["cleaned_text"].split())
            shared_terms = list(query_tokens.intersection(cand_tokens))

            score = float(row["similarity_score"])
            results.append({
                "complaint_id": row.get("complaint_id", "N/A"),
                "title": row.get("title", "Untitled"),
                "description": row.get("description", ""),
                "location": row.get("location", "Unspecified"),
                "category": row.get("category", "Unspecified"),
                "status": row.get("status", "Pending"),
                "submitted_at": row.get("submitted_at", "Unknown"),
                "similarity_score": round(score, 4),
                "similarity_percentage": round(score * 100, 1),
                "shared_terms": shared_terms[:8],
                "is_likely_duplicate": score >= 0.75,
                "confidence_note": (
                    "High lexical similarity detected. Please verify room/time details before merging."
                    if score >= 0.75
                    else "Moderate topic overlap. May be a related symptom rather than an identical ticket."
                ),
            })

        return results


def evaluate_duplicate_thresholds(
    pairs_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Evaluates precision, recall, and F1 across multiple cosine similarity thresholds
    on a labeled test set of complaint pairs (is_duplicate: 0 or 1).
    """
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    if pairs_df.empty or "text_1" not in pairs_df or "text_2" not in pairs_df or "is_duplicate" not in pairs_df:
        return pd.DataFrame()

    cleaned_1 = pairs_df["text_1"].astype(str).apply(clean_text)
    cleaned_2 = pairs_df["text_2"].astype(str).apply(clean_text)
    y_true = np.array(pairs_df["is_duplicate"].tolist(), dtype=int)

    # Fit vectorizer across all pair texts
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    all_texts = list(cleaned_1) + list(cleaned_2)
    vec.fit(all_texts)

    v1 = vec.transform(cleaned_1)
    v2 = vec.transform(cleaned_2)

    # Calculate pairwise dot products (row by row cosine since L2 normalized)
    sims = np.asarray(v1.multiply(v2).sum(axis=1)).ravel()

    records = []
    for th in thresholds:
        y_pred = (sims >= th).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        records.append({
            "threshold": th,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return pd.DataFrame(records)
