"""
CampusPulse - Model Training and Evaluation Pipeline
Trains TF-IDF + Multiclass Logistic Regression using the authentic train.csv and test.csv partitions.
Fits vectorizer ONLY on training partition to eliminate data leakage.
Evaluates against Majority Baseline and tests duplicate detection benchmark.
Saves model artifacts and genuine metrics to `models/`.
"""

import json
import os
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import clean_text
from similarity import evaluate_duplicate_thresholds


def train_and_evaluate(
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
    pairs_path: str = "data/complaint_pairs.csv",
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes end-to-end training and evaluation using train.csv and test.csv.
    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("train.csv or test.csv not found in data/ directory.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean text
    train_df["cleaned_text"] = train_df["Complaint_Description"].fillna("").apply(clean_text)
    test_df["cleaned_text"] = test_df["Complaint_Description"].fillna("").apply(clean_text)

    # Ensure no empty strings
    train_df = train_df[train_df["cleaned_text"].str.strip().str.len() > 0].copy()
    test_df = test_df[test_df["cleaned_text"].str.strip().str.len() > 0].copy()

    X_train = np.array(train_df["cleaned_text"].tolist(), dtype=object)
    y_train = np.array(train_df["Category"].tolist(), dtype=object)

    X_test = np.array(test_df["cleaned_text"].tolist(), dtype=object)
    y_test = np.array(test_df["Category"].tolist(), dtype=object)

    categories = sorted(list(set(y_train.tolist()).union(set(y_test.tolist()))))

    print(f"Loaded {len(X_train)} training complaints and {len(X_test)} testing complaints.")
    print(f"Categories ({len(categories)}): {categories}")

    # Build Pipeline: TF-IDF (unigrams + bigrams) + Multiclass Logistic Regression
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_features=4000,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs",
            ),
        ),
    ])

    # Fit pipeline ONLY on training data
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Baseline 1: Majority Class Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    # Baseline 2: Multinomial Naive Bayes Classifier
    nb_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_features=4000,
            ),
        ),
        ("nb", MultinomialNB(alpha=1.0)),
    ])
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)

    # Compute genuine performance metrics for primary model (Logistic Regression)
    test_acc = accuracy_score(y_test, y_pred)
    test_macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    test_macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Baseline metrics
    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_macro_f1 = f1_score(y_test, dummy_pred, average="macro", zero_division=0)

    nb_acc = accuracy_score(y_test, nb_pred)
    nb_macro_prec = precision_score(y_test, nb_pred, average="macro", zero_division=0)
    nb_macro_rec = recall_score(y_test, nb_pred, average="macro", zero_division=0)
    nb_macro_f1 = f1_score(y_test, nb_pred, average="macro", zero_division=0)

    # Per-class report
    cls_report = classification_report(
        y_test, y_pred, labels=categories, target_names=categories, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=categories)

    # Extract top keywords per category from Logistic Regression coefficients
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_keywords_per_category: Dict[str, List[str]] = {}
    for idx, cat_name in enumerate(classifier.classes_):
        top_indices = np.argsort(classifier.coef_[idx])[-8:][::-1]
        top_keywords_per_category[cat_name] = feature_names[top_indices].tolist()

    # Evaluate Duplicate Detection with Validation / Test Split
    duplicate_results: List[Dict[str, Any]] = []
    duplicate_val_test_report: Dict[str, Any] = {}

    if os.path.exists(pairs_path):
        pairs_df = pd.read_csv(pairs_path)

        # Full threshold sweep across benchmark for Plotly PR curve
        eval_pairs_df = evaluate_duplicate_thresholds(
            pairs_df, thresholds=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        )
        duplicate_results = eval_pairs_df.to_dict(orient="records")

        # Academic rigor: 50/50 Validation vs Untouched Test Split
        if len(pairs_df) >= 20 and "is_duplicate" in pairs_df.columns:
            val_pairs, test_pairs = train_test_split(
                pairs_df, test_size=0.5, stratify=pairs_df["is_duplicate"], random_state=random_state
            )
            val_df = evaluate_duplicate_thresholds(
                val_pairs, thresholds=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            )
            # Choose threshold with best F1 on validation (tie-breaker: highest threshold for precision)
            best_val_row = val_df.sort_values(by=["f1_score", "threshold"], ascending=[False, False]).iloc[0]
            chosen_threshold = float(best_val_row["threshold"])

            # Evaluate chosen threshold on untouched test pairs
            test_res_df = evaluate_duplicate_thresholds(test_pairs, thresholds=[chosen_threshold])
            test_row = test_res_df.iloc[0].to_dict()

            duplicate_val_test_report = {
                "validation_pairs_count": len(val_pairs),
                "test_pairs_count": len(test_pairs),
                "selected_threshold_from_val": chosen_threshold,
                "val_f1_at_chosen_threshold": float(best_val_row["f1_score"]),
                "test_evaluation": {
                    "threshold": float(test_row["threshold"]),
                    "precision": float(test_row["precision"]),
                    "recall": float(test_row["recall"]),
                    "f1_score": float(test_row["f1_score"]),
                    "tp": int(test_row["tp"]),
                    "fp": int(test_row["fp"]),
                    "fn": int(test_row["fn"]),
                    "tn": int(test_row["tn"]),
                },
            }

            print(f"\nDuplicate Detection Split: Validation={len(val_pairs)}, Test={len(test_pairs)}")
            print(f"Selected Threshold on Validation: tau={chosen_threshold:.2f} (Val F1={best_val_row['f1_score']:.4f})")
            print(f"Untouched Test Results: Precision={test_row['precision']:.4f}, Recall={test_row['recall']:.4f}, F1={test_row['f1_score']:.4f}")
            print(f"Test Confusion: TP={int(test_row['tp'])}, FP={int(test_row['fp'])}, FN={int(test_row['fn'])}, TN={int(test_row['tn'])}")

    # Structure complete metrics dictionary
    eval_metrics: Dict[str, Any] = {
        "dataset_name": "University Student Complaints Benchmark",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "categories": categories,
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_precision": round(float(test_macro_prec), 4),
        "test_macro_recall": round(float(test_macro_rec), 4),
        "test_macro_f1": round(float(test_macro_f1), 4),
        "baseline_majority_accuracy": round(float(dummy_acc), 4),
        "baseline_majority_f1": round(float(dummy_macro_f1), 4),
        "baseline_nb_accuracy": round(float(nb_acc), 4),
        "baseline_nb_macro_precision": round(float(nb_macro_prec), 4),
        "baseline_nb_macro_recall": round(float(nb_macro_rec), 4),
        "baseline_nb_macro_f1": round(float(nb_macro_f1), 4),
        "accuracy_gain_over_baseline": round(float(test_acc - dummy_acc), 4),
        "accuracy_gain_over_majority_baseline": round(float(test_acc - dummy_acc), 4),
        "accuracy_gain_over_nb_baseline": round(float(test_acc - nb_acc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "top_keywords_per_category": top_keywords_per_category,
        "duplicate_detection_benchmark": duplicate_results,
        "duplicate_detection_split_evaluation": duplicate_val_test_report,
    }

    # Save model pipeline
    model_save_path = os.path.join(models_dir, "complaint_classifier.joblib")
    joblib.dump(pipeline, model_save_path)
    print(f"\nTrained pipeline saved to {model_save_path}")

    # Save metrics JSON
    metrics_save_path = os.path.join(models_dir, "eval_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_save_path}")

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Primary Model Accuracy (LogReg):  {test_acc * 100:.2f}% (Macro F1: {test_macro_f1:.4f})")
    print(f"Baseline 1 (Majority Class):      {dummy_acc * 100:.2f}% (Macro F1: {dummy_macro_f1:.4f})")
    print(f"Baseline 2 (Multinomial NB):      {nb_acc * 100:.2f}% (Macro F1: {nb_macro_f1:.4f})")
    print(f"Net Accuracy Gain over Majority:  +{(test_acc - dummy_acc) * 100:.2f}%")
    print(f"Net Accuracy Gain over NB:        +{(test_acc - nb_acc) * 100:.2f}%")
    print("====================================================")

    return eval_metrics


if __name__ == "__main__":
    train_and_evaluate()
