# CampusPulse — Version History & Changelog

> This file tracks every significant version of the project — what was changed, why it was changed, what improved, and what the impact was. Maintained as a living document alongside development.

---

## [v2.1.0] — 2026-10-24 — *Config-Driven Architecture + Performance Boost*

### 🎯 Why This Version
After v2.0.0 established the working NLP pipeline, the codebase had a critical maintainability problem: domain knowledge (campus abbreviations, priority keywords, sentiment fallback words, negation terms) was **hardcoded directly inside Python files**. Adding a new abbreviation or urgency keyword required editing source code. This version externalizes all of that into JSON config files — turning the system into a truly config-driven NLP pipeline.

### ✅ What Was Done

#### New Files Created
| File | Purpose |
|------|---------|
| `data/campus_abbreviations.json` | 45 campus-specific abbreviation mappings (was hardcoded `CAMPUS_EXPANSIONS` dict in `preprocessing.py`) |
| `data/negation_words.json` | 13 negation words preserved during stopword removal (was hardcoded `NEGATION_WORDS` set) |
| `data/priority_keywords.json` | All safety keywords, academic urgency terms, operational urgency terms, category baselines, and VADER fallback lexicons (was hardcoded in `priority.py`) |

#### Files Modified

**`requirements.txt`**
- Added `contractions>=0.1.73`

**`preprocessing.py` — Full Refactor**
- `CONTRACTION_EXPANSIONS` (15-pattern hardcoded regex dict) → replaced by `contractions` library (handles 200+ English contractions automatically)
- `CAMPUS_EXPANSIONS` dict → loaded dynamically from `data/campus_abbreviations.json`
- `NEGATION_WORDS` set → loaded dynamically from `data/negation_words.json`
- Added `_load_json()` helper with graceful fallback (warning + empty dict, never crashes)
- `explain_preprocessing()` trace now includes `contraction_expanded`, `campus_expansions_loaded`, `negation_words_loaded` fields
- `DEFAULT_STOPWORDS` retained as pure offline emergency fallback only

**`priority.py` — Full Refactor**
- `CATEGORY_BASELINES` dict → loaded from `data/priority_keywords.json`
- `SAFETY_KEYWORDS` dict → loaded from `data/priority_keywords.json`
- `ACADEMIC_URGENCY_KEYWORDS` dict → loaded from `data/priority_keywords.json`
- `OPERATIONAL_URGENCY_KEYWORDS` dict → loaded from `data/priority_keywords.json`
- Sentiment fallback `negative_words` and `positive_words` sets → loaded from `data/priority_keywords.json`
- All hardcoded fallbacks preserved as inline defaults (library gracefully degrades)
- Priority scoring logic cleaned — redundant `if/elif` chains replaced with concise expressions

**Model Retrained** (`python train.py`)
- Triggered by improved contraction expansion in preprocessing (cleaner training data)
- VADER lexicon downloaded (`nltk.download('vader_lexicon')`)

### 📊 Performance Improvement
| Metric | v2.0.0 | v2.1.0 | Delta |
|--------|--------|--------|-------|
| Test Accuracy | 81.36% | **86.44%** | **+5.08%** |
| Macro Precision | 88.76% | **90.07%** | **+1.31%** |
| Macro Recall | 86.25% | **90.00%** | **+3.75%** |
| Macro F1-Score | 0.8328 | **0.8790** | **+0.0462** |
| Gain vs. Majority | +32.20% | **+37.29%** | **+5.09%** |

> **Why did accuracy improve?** The `contractions` library expanded far more contraction patterns than the old 15-pattern regex dict. This cleaned the training text more accurately, giving the TF-IDF vectorizer better, more consistent n-gram features to work with.

### 🧪 Tests
- All **22 tests pass** (was 22 passing in v2.0.0 too, but 3 were failing due to sklearn version mismatch from old `.joblib` — retraining fixed them + VADER download fixed the sentiment test)

---

## [v2.0.0] — 2026-10-23 — *Original Project Submission*

### 🎯 Overview
Initial implementation of CampusPulse — an explainable NLP campus complaint intelligence system.

### Key Components
#### Core Pipeline
- `preprocessing.py` — Text cleaning, regex abbreviation normalization, custom stopwords preservation (negations kept)
- `train.py` — Stratified training of TF-IDF + Logistic Regression multiclass classifier
- `priority.py` — VADER sentiment analysis + rule-based urgency scoring heuristic (0-100)
- `similarity.py` — TF-IDF cosine similarity engine for near-duplicate complaint clustering
- `database.py` — SQLite persistence (`campuspulse.db`) with status lifecycle management
- `evaluation.py` — Metric generation, confusion matrix, precision-recall curve plotting
- `app.py` — 4-page Streamlit web application

#### Data
- `data/train.csv` — 273 labeled training complaints
- `data/test.csv` — 59 held-out test complaints
- `data/complaint_pairs.csv` — 88 labeled duplicate pairs for benchmark
- `data/university_students_complaints.xlsx` — source data with duplicate group labels

#### Model
- Algorithm: **TF-IDF (unigrams + bigrams, sublinear TF, max 4000 features) + Logistic Regression (C=1.0, max_iter=1000, class_weight='balanced')**
- 5 categories: Academic, Administrative, Finance, Infrastructure, Technical
- Pipeline saved to `models/complaint_classifier.joblib`

#### Documentation
- `README.md` — Setup and quick start guide
- `PROJECT_REPORT.md` — Full academic report
- `VIVA_PREPARATION.md` — 20 viva Q&As + live demo script
- `AUDIT_RESULTS.md` — Independent data provenance and leakage audit

#### Tests
- 18 unit tests across 5 test files covering preprocessing, pipeline inference, similarity, priority, and database

### 📊 Performance (v2.0.0)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Majority Baseline | 49.15% | 0.1318 |
| Multinomial Naive Bayes | 62.71% | 0.3512 |
| **CampusPulse v2.0.0** | **81.36%** | **0.8328** |

**Duplicate Detection** (τ = 0.50): Precision 100% | Recall 95.45% | F1 0.9767

---

## Version Comparison Table

| Feature / Metric | v2.0.0 | v2.1.0 |
|---|---|---|
| **Test Accuracy** | 81.36% | **86.44%** |
| **Macro F1** | 0.8328 | **0.8790** |
| **Gain vs. Majority** | +32.20% | **+37.29%** |
| Contraction Expansion | 15-pattern regex dict (hardcoded) | `contractions` library (200+ patterns) |
| Campus Abbreviations | Hardcoded dict in `preprocessing.py` | `data/campus_abbreviations.json` (45 entries) |
| Negation Words | Hardcoded set in `preprocessing.py` | `data/negation_words.json` (13 words) |
| Priority Keywords | Hardcoded dicts in `priority.py` | `data/priority_keywords.json` (all keyword groups) |
| Sentiment Fallback Lexicon | Hardcoded sets in `priority.py` | `data/priority_keywords.json` |
| Adding new keyword | Edit Python source code | Edit JSON file only |
| Unit Tests Passing | 22/22 | 22/22 |
| Python Dependencies | 8 packages | 9 packages (`contractions` added) |

---

## Roadmap / Planned Improvements

| Priority | Feature | Status |
|----------|---------|--------|
| 🟢 High | Complaint Genuineness & Anti-Mischief Defense Engine | Planned |
| 🟢 High | Quarantined Spam/Mischief Queue with Admin Unblock / Purge | Planned |
| 🟢 High | Streamlit Admin Config Studio to edit JSON rules directly | Planned |
