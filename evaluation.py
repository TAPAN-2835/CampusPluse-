"""
CampusPulse - Model Evaluation and Visualization Module
Loads saved evaluation metrics, computes visualization figures with Plotly,
and provides tabular breakdowns for the evaluation dashboard.
"""

import json
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_eval_metrics(metrics_path: str = "models/eval_metrics.json") -> Optional[Dict[str, Any]]:
    """Loads evaluation metrics from saved JSON artifact."""
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def create_confusion_matrix_figure(metrics: Dict[str, Any]) -> go.Figure:
    """Generates an interactive Plotly heatmap for the test set confusion matrix."""
    cm = np.array(metrics.get("confusion_matrix", []))
    categories = metrics.get("categories", [])

    if len(cm) == 0 or len(categories) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Confusion Matrix Data Unavailable", showarrow=False)
        return fig

    # Text annotations inside heatmap cells
    annotations = []
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=categories[j],
                    y=categories[i],
                    text=str(val),
                    font=dict(color="white" if val > cm.max() / 2 else "#1e293b", size=14),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=categories,
            y=categories,
            colorscale="Tealgrn",
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title="Test Set Confusion Matrix (Normalized Multiclass Evaluation)",
        xaxis=dict(title="Predicted Class", tickangle=25),
        yaxis=dict(title="True Class", autorange="reversed"),
        annotations=annotations,
        margin=dict(l=60, r=40, t=60, b=80),
        height=450,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def create_per_class_metrics_figure(metrics: Dict[str, Any]) -> go.Figure:
    """Creates a grouped bar chart of Precision, Recall, and F1 per category."""
    cls_report = metrics.get("classification_report", {})
    categories = metrics.get("categories", [])

    data = []
    for cat in categories:
        if cat in cls_report:
            item = cls_report[cat]
            data.append({
                "Category": cat,
                "Precision": round(item.get("precision", 0) * 100, 1),
                "Recall": round(item.get("recall", 0) * 100, 1),
                "F1-Score": round(item.get("f1-score", 0) * 100, 1),
                "Support": item.get("support", 0),
            })

    df = pd.DataFrame(data)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Per-Class Report Unavailable", showarrow=False)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Category"], y=df["Precision"], name="Precision (%)", marker_color="#3b82f6"))
    fig.add_trace(go.Bar(x=df["Category"], y=df["Recall"], name="Recall (%)", marker_color="#10b981"))
    fig.add_trace(go.Bar(x=df["Category"], y=df["F1-Score"], name="F1-Score (%)", marker_color="#f59e0b"))

    fig.update_layout(
        title="Per-Class Performance Breakdown (Held-Out Test Set)",
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 105]),
        xaxis=dict(title="Complaint Category", tickangle=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=80),
        height=400,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def create_duplicate_pr_figure(metrics: Dict[str, Any]) -> go.Figure:
    """Generates Precision-Recall trade-off curve across similarity thresholds."""
    bench = metrics.get("duplicate_detection_benchmark", [])
    if not bench:
        fig = go.Figure()
        fig.add_annotation(text="Duplicate Benchmark Data Unavailable", showarrow=False)
        return fig

    df = pd.DataFrame(bench)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["threshold"],
            y=df["precision"],
            mode="lines+markers",
            name="Precision",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["threshold"],
            y=df["recall"],
            mode="lines+markers",
            name="Recall",
            line=dict(color="#16a34a", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["threshold"],
            y=df["f1_score"],
            mode="lines+markers",
            name="F1-Score",
            line=dict(color="#d97706", width=3, dash="dot"),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title="Duplicate Detection Performance Across Cosine Similarity Cutoffs",
        xaxis=dict(title="Cosine Similarity Threshold (τ)", range=[0.15, 0.85]),
        yaxis=dict(title="Metric Score (0.0 - 1.0)", range=[0.0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=60),
        height=380,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def get_per_class_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Constructs a clean summary DataFrame of test set per-class metrics."""
    cls_report = metrics.get("classification_report", {})
    categories = metrics.get("categories", [])

    rows = []
    for cat in categories:
        if cat in cls_report:
            item = cls_report[cat]
            rows.append({
                "Category": cat,
                "Precision": f"{item.get('precision', 0) * 100:.1f}%",
                "Recall": f"{item.get('recall', 0) * 100:.1f}%",
                "F1-Score": f"{item.get('f1-score', 0) * 100:.1f}%",
                "Support (Test Count)": int(item.get("support", 0)),
            })
    return pd.DataFrame(rows)
