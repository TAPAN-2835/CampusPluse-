"""
Unit tests for trained model pipeline and prediction inference.
"""

import os
import joblib
import numpy as np
import pytest
from preprocessing import clean_text


@pytest.fixture(scope="module")
def pipeline():
    model_path = os.path.join("models", "complaint_classifier.joblib")
    assert os.path.exists(model_path), f"Trained model not found at {model_path}"
    return joblib.load(model_path)


def test_model_loads_successfully(pipeline):
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_prediction_output(pipeline):
    sample_text = clean_text("Wi-Fi in library reading room is disconnected and very slow")
    pred = pipeline.predict([sample_text])[0]
    proba = pipeline.predict_proba([sample_text])[0]

    assert isinstance(pred, str)
    assert pred in pipeline.classes_
    assert len(proba) == len(pipeline.classes_)
    assert np.isclose(np.sum(proba), 1.0, atol=1e-3)


def test_multiclass_categories(pipeline):
    classes = list(pipeline.classes_)
    expected_categories = {"Academic", "Administrative", "Finance", "Infrastructure", "Technical"}
    for cat in classes:
        assert cat in expected_categories


def test_empty_string_safe_handling(pipeline):
    # Pipeline should handle clean text gracefully
    empty_clean = clean_text("")
    # Even if empty, transform should not crash
    proba = pipeline.predict_proba([empty_clean])[0]
    assert len(proba) == len(pipeline.classes_)
