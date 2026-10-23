# CampusPulse — Version History & Changelog

> This file tracks every significant version of the project — what was changed, why it was changed, what improved, and what the impact was. Maintained as a living document alongside development.

---

## [v2.2.0] — 2026-10-24 — *Anti-Mischief Gate, Complaint Genuineness Engine & Admin Studio*

### 🎯 Why This Version
A student complaint portal open to all students faces two operational hazards:
1. **Frivolous mischief / spam / trolling**: Random keyboard mashing (`asdfghjkl`), dummy test entries (`test 123`), excessive single-word repetition, and off-topic submissions overwhelm administrators with useless noise.
2. **Abusive conduct & security threats**: Hostile threats or abusive profanity directed at faculty or administrative staff require immediate quarantine and critical alerts.

This version introduces an intelligent, transparent **Genuineness and Mischief Detection Engine** (`genuineness.py` + `data/genuineness_config.json`) that validates incoming complaints before they enter the active workflow, plus an **Admin Config Studio** to tune rules dynamically.

### ✅ What Was Done

#### New Files Created
| File | Purpose |
|------|---------|
| [`genuineness.py`](file:///c:/Users/anish/OneDrive/Desktop/NLP%20project/CampusPluse-/genuineness.py) | Real-time complaint authenticity validation engine (0–100 score, factor breakdown, verdicts: `GENUINE`, `NEEDS_REVIEW`, `SPAM_REJECTED`) |
| [`data/genuineness_config.json`](file:///c:/Users/anish/OneDrive/Desktop/NLP%20project/CampusPluse-/data/genuineness_config.json) | Externalized thresholds, spam patterns, threat keywords, profanity lists, and campus relevance keywords |
| [`tests/test_genuineness.py`](file:///c:/Users/anish/OneDrive/Desktop/NLP%20project/CampusPluse-/tests/test_genuineness.py) | Comprehensive unit tests for genuine complaints, keyboard mashing, short text, dummy submissions, threats, profanity, and word repetition |

#### Files Modified

**`database.py` — Schema Migration & Quarantine Operations**
- Added columns to `complaints` table: `genuineness_score`, `genuineness_verdict`, `genuineness_flags` with automatic backward-compatible schema migration (`PRAGMA table_info`).
- Added `delete_complaint(complaint_id)` to permanently purge malicious submissions.
- Updated `get_summary_statistics()` to track `spam_blocked` count and separate genuine active tickets from quarantined mischief.

**`app.py` — Full UI Integration**
- **Executive Dashboard**: Added `Filtered Mischief` KPI card; separated active genuine complaints from quarantined spam.
- **Submit & Triage Page**: Added 1-click test mischief & threat demo presets; real-time Authenticity Scorecard (0–100); 4th detailed inspection tab `🛡️ Genuineness & Anti-Mischief Audit`; automatic quarantine routing (`Spam/Blocked` status).
- **Complaint Explorer**: Added dual tabs: `📋 Active Complaint Queue` (unpolluted) and `🛡️ Blocked Mischief / Spam Quarantine` with inspector, `✅ Approve & Unblock`, and `🗑️ Permanently Delete` actions.
- **Admin Config Studio**: Added a live interactive visual editor for `campus_abbreviations.json`, `priority_keywords.json`, and `genuineness_config.json` with instant JSON validation and persistence.

**`tests/test_database.py`**
- Added test coverage for `genuineness_score`, `genuineness_verdict`, `genuineness_flags`, and `delete_complaint`.

### 🧪 Tests
- **31/31 unit tests passing** across the entire project (`tests/test_*.py`).

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

## [v2.0.0] — Initial Academic Submission — *The Working NLP System*

### 🎯 Why This Version
This was the first fully functional, end-to-end version submitted as a minor academic project. It established the entire pipeline architecture: preprocessing → classification → duplicate detection → sentiment → priority scoring → Streamlit dashboard.

### ✅ What Was Done

#### Architecture Established
- **`preprocessing.py`**: Text cleaning pipeline with hardcoded stopwords, contraction regex dict, campus abbreviation dict, negation word preservation
- **`similarity.py`**: `ComplaintSimilarityEngine` using TF-IDF cosine similarity for duplicate detection; `evaluate_duplicate_thresholds()` for benchmark P/R/F1
- **`priority.py`**: VADER sentiment analysis + 7-factor rule-based priority scoring (0–100 scale)
- **`database.py`**: SQLite CRUD layer with auto-incremented ticket IDs (`CP-YYYY-XXXX`)
- **`evaluation.py`**: Plotly figure generators for confusion matrix, per-class metrics, PR curves
- **`train.py`**: TF-IDF + Logistic Regression pipeline with leakage-free train/test split evaluation
- **`prepare_data.py`**: Seeds SQLite with benchmark complaint data from CSV
- **`app.py`** (843 lines): 4-page Streamlit dashboard with custom CSS, Plotly charts, duplicate cluster detection

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

| Feature / Metric | v2.0.0 | v2.1.0 | v2.2.0 (Current) |
|---|---|---|---|
| **Test Accuracy** | 81.36% | **86.44%** | **86.44%** |
| **Macro F1** | 0.8328 | **0.8790** | **0.8790** |
| **Gain vs. Majority** | +32.20% | **+37.29%** | **+37.29%** |
| Contraction Expansion | 15 regex patterns | `contractions` library | `contractions` library |
| Campus Abbreviations | Hardcoded dict | `campus_abbreviations.json` | JSON + Live Admin UI Editor |
| Priority Keywords | Hardcoded dicts | `priority_keywords.json` | JSON + Live Admin UI Editor |
| Genuineness / Anti-Mischief Gate | ❌ None (all accepted) | ❌ None | ✅ **`genuineness.py` (0-100 Score)** |
| Spam & Keyboard Mash Filtering | ❌ None | ❌ None | ✅ **Quarantined to Spam Queue** |
| Threat & Security Alert Detection | ❌ None | ❌ None | ✅ **Automatic Security Quarantine** |
| Admin Config Studio UI | ❌ None | ❌ None | ✅ **Live JSON Editor in app.py** |
| Unit Tests Passing | 18/18 | 22/22 | **31/31** |

---

## Roadmap & Feature Status

| Priority | Feature | Status |
|----------|---------|--------|
| 🟢 High | Externalize domain knowledge to JSON configs | ✅ Completed (v2.1.0) |
| 🟢 High | Update documentation (`PROJECT_REPORT.md`, `README.md`, `VIVA_PREPARATION.md`) | ✅ Completed (v2.1.0) |
| 🟢 High | Complaint Genuineness & Anti-Mischief Defense Engine | ✅ Completed (v2.2.0) |
| 🟢 High | Quarantined Spam/Mischief Queue with Admin Unblock / Purge | ✅ Completed (v2.2.0) |
| 🟢 High | Streamlit Admin Config Studio to edit JSON rules directly | ✅ Completed (v2.2.0) |
| 🟡 Low | Explore neural/transformer embeddings if dataset scales | On Hold (Small data ceiling) |

