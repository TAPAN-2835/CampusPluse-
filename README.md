# CampusPulse 🎓
### Explainable NLP-Based Campus Complaint Intelligence System

CampusPulse is an intermediate-level, locally runnable Natural Language Processing system for university complaint triage, duplicate ticket detection, sentiment analysis, and priority intelligence.

Built with **Python**, **Streamlit**, **Scikit-Learn**, **NLTK**, **Pandas**, and **Plotly**.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment
Ensure Python 3.10+ is installed on your machine.

Clone or open the project folder in your terminal:
```bash
cd CampusPulse
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

*(Optional: Download NLTK lexicons if running offline for the first time)*:
```bash
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('stopwords', quiet=True)"
```

---

### 2. Prepare Data & Seed Database
Prepare the authentic benchmark pair test set and seed the local SQLite database (`campuspulse.db`):
```bash
python prepare_data.py
```

---

### 3. Train & Evaluate the NLP Model
Train the TF-IDF + Multiclass Logistic Regression pipeline on `data/train.csv` and evaluate on held-out `data/test.csv`:
```bash
python train.py
```
*Outputs:*
- Trained model saved to `models/complaint_classifier.joblib`
- Comprehensive evaluation metrics saved to `models/eval_metrics.json`
- Test accuracy: **81.36%** | Macro F1-score: **0.8328** | Baseline Gain: **+32.20%**

---

### 4. Run Automated Unit Tests
Run the test suite covering preprocessing, pipeline inference, similarity matching, and priority scoring:
```bash
python -m pytest tests/ -v
```
*(All 18 unit tests pass)*

---

### 5. Launch the Web Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your web browser at: `http://localhost:8501`

---

## 📂 Project Architecture

```
CampusPulse/
├── app.py                      # Multi-page Streamlit application & interactive UI
├── train.py                    # Model training pipeline & leakage-free evaluation
├── prepare_data.py             # Prepares pair benchmark & seeds SQLite database
├── preprocessing.py            # Text normalization, abbreviation expansion & negation filtering
├── similarity.py               # TF-IDF cosine similarity duplicate detection engine
├── priority.py                 # VADER sentiment analysis & explainable priority rules
├── evaluation.py               # Plotly figure generators & metric tables
├── database.py                 # SQLite local persistence layer (campuspulse.db)
├── data/
│   ├── train.csv               # Authentic training complaints (N=273)
│   ├── test.csv                # Held-out testing complaints (N=59)
│   ├── complaint_pairs.csv     # Labeled benchmark pairs for duplicate detection (N=88)
│   ├── complaints.csv          # Standardized reference complaints dataset
│   └── university_students_complaints.xlsx # Complete reference dataset with duplicate groups
├── models/
│   ├── complaint_classifier.joblib  # Fitted TF-IDF + Logistic Regression pipeline
│   └── eval_metrics.json            # Genuine test evaluation results & confusion matrix
├── tests/
│   ├── test_preprocessing.py   # Unit tests for text cleaning & edge cases
│   ├── test_pipeline.py        # Unit tests for classifier inference
│   ├── test_similarity.py      # Unit tests for cosine similarity matching
│   └── test_priority.py        # Unit tests for sentiment & priority rules
├── PROJECT_REPORT.md           # Formal academic minor project report
├── VIVA_PREPARATION.md         # Comprehensive viva defense Q&A & live demo script
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

---

## 🎯 Core Features

1. **Executive Dashboard**:
   - Live KPI cards: Total Complaints, Pending Action, In Progress, Resolved, and Urgent/Critical Issues.
   - Interactive Plotly visualizations for Category Distribution and Priority Triage.
   - **Recurring Issues & Duplicate Clusters**: Surfaces multi-ticket clusters sharing identical root problems.
2. **Submit & Triage Complaint**:
   - 1-Click Faculty Demo Buttons for instant viva testing.
   - Real-time TF-IDF multiclass category classification with estimated class probabilities.
   - Transparent preprocessing trace (abbreviation expansion, stopword filtering).
   - VADER sentiment estimation (Compound polarity from -1.0 to +1.0).
   - Explainable priority scoring based on documented campus rules (physical hazards, academic deadlines, recurrence).
   - Real-time Cosine Similarity Duplicate Alert against existing tickets with configurable cutoff ($\tau$).
3. **Complaint Explorer**:
   - Searchable and filterable complaint database (by category, priority, status, department).
   - Ticket inspector and status updater (`Pending` $\rightarrow$ `Under Review` $\rightarrow$ `In Progress` $\rightarrow$ `Resolved`).
4. **Model Evaluation & Explainability**:
   - Zero-leakage held-out test evaluation scorecard.
   - Interactive Confusion Matrix heatmap.
   - Per-Class Precision, Recall, and F1 bar charts.
   - Duplicate Detection Precision-Recall trade-off curve across similarity thresholds.
   - Top informative TF-IDF n-grams per category extracted from model coefficients.

---

## 📊 Evaluation Summary

| Model / Benchmark | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Net Gain vs. Majority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 49.15% | 9.83% | 20.00% | 0.1318 | — |
| **Multinomial Naive Bayes** | 62.71% | 46.60% | 34.82% | 0.3512 | +13.56% |
| **CampusPulse (Held-Out Test N=59)** | **81.36%** | **88.76%** | **86.25%** | **0.8328** | **+32.20%** |
| **CampusPulse (Unleaked Subset N=47)** | **76.60%** | **80.89%** | **85.39%** | **0.7637** | **+27.45%** |

### Duplicate Detection (Untouched Test Partition at $\tau = 0.50$)
- **Test Precision**: **100.0%** | **Test Recall**: **95.45%** | **Test F1-Score**: **0.9767**
- **Test Confusion**: TP: **21**, FP: **0**, FN: **1**, TN: **22**

---

## 🛠️ Troubleshooting

- **ModuleNotFoundError for Streamlit or Plotly**:
  Ensure you installed requirements in the active Python environment:
  ```bash
  python -m pip install -r requirements.txt
  ```
- **NLTK Download Errors in Offline Environments**:
  CampusPulse includes built-in fallback dictionaries for stopwords and sentiment. If offline, the application automatically uses embedded lexicons without crashing.
- **Port 8501 Already in Use**:
  Launch Streamlit on an alternate port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

## 📚 Academic Documentation
For complete technical write-ups and viva defense materials:
- [AUDIT_RESULTS.md](file:///c:/Users/Tapan/Desktop/m/AUDIT_RESULTS.md): Rigorous independent audit of dataset provenance, leakage analysis, baseline reproductions, duplicate benchmark, and bug fixes.
- [PROJECT_REPORT.md](file:///c:/Users/Tapan/Desktop/m/PROJECT_REPORT.md): Complete minor project report with mathematical formulations, architecture diagrams, and experimental methodology.
- [VIVA_PREPARATION.md](file:///c:/Users/Tapan/Desktop/m/VIVA_PREPARATION.md): 2-minute elevator pitch, 20 faculty viva questions with answers, and step-by-step live demo script.
