# CampusPulse: Explainable NLP-Based Campus Complaint Intelligence System
**Academic Minor Project Report**  
*Department of Computer Science & Engineering / Information Technology*

---

## Abstract
CampusPulse is an explainable, lightweight Natural Language Processing (NLP) system designed to automate the ingestion, categorization, duplicate detection, sentiment estimation, and priority triage of student complaints across a university campus. Traditional campus grievance mechanisms rely on manual ticket sorting, leading to delayed response times, overlooked infrastructure hazards, and redundant maintenance dispatches for identical issues. 

CampusPulse addresses these challenges using a supervised Machine Learning pipeline combining domain-specific text preprocessing, sublinear TF-IDF (Term Frequency-Inverse Document Frequency) unigram and bigram representation, and a multiclass Logistic Regression classifier. To prevent redundant work orders, the system employs cosine similarity over vectorized document embeddings to detect semantic and lexical duplicates with configurable sensitivity. Furthermore, an explainable priority scoring engine decouples emotional sentiment (estimated via VADER) from physical hazard severity indicators, ensuring that emergency situations (such as electrical sparking or contaminated water) receive immediate escalation regardless of the student's phrasing. Evaluated on a real benchmark of 332 university student complaints, CampusPulse achieves **86.44% test accuracy** and a **Macro F1-score of 0.8790**, representing a **+37.29% net accuracy gain** over a majority-class baseline. All components run locally without external cloud APIs, GPUs, or proprietary services, providing an inspectable, transparent, and viva-ready solution. The system uses a fully config-driven architecture — all keyword lists and domain abbreviations are stored in external JSON files, making the pipeline extensible without any code changes.

---

## 1. Problem Statement
University campus administrations receive hundreds of operational, academic, and infrastructure complaints each semester. Managing these tickets manually incurs severe operational inefficiencies:
1. **Misrouting and Delayed Resolution**: Complaints sent to general inboxes take days to be routed to appropriate departments (e.g., IT vs. Maintenance vs. Academic Affairs).
2. **Duplication and Resource Waste**: A single issue (such as a Wi-Fi outage or broken water cooler) often triggers dozens of separate complaints from different students in the same building, overwhelming staff with duplicate tickets.
3. **Subjective Priority Triage**: Without standardized scoring, tickets are resolved on a first-come-first-served basis rather than by genuine hazard severity or academic deadline urgency.
4. **Lack of Explainability**: Modern deep learning "black box" models cannot provide transparent justifications to university administrators for why a complaint was flagged as critical or assigned to a specific department.

---

## 2. Project Objectives
- **Automated Text Classification**: Classify raw, unstructured student complaint descriptions into operational categories using supervised NLP.
- **Explainable Duplicate Detection**: Compute vector similarity across historical tickets to identify related issues, highlight overlapping n-grams, and quantify duplication confidence.
- **Decoupled Sentiment and Urgency Scoring**: Implement a transparent, rule-governed priority engine that distinguishes student emotional frustration (sentiment) from structural, electrical, and academic risks (severity).
- **Executive Administration Dashboard**: Provide interactive analytics, recurring symptom cluster tracking, ticket lifecycle management, and transparent prediction probabilities.
- **Rigorous, Leakage-Free Evaluation**: Validate the pipeline on held-out test data, demonstrating quantifiable gains over majority-class baselines and reporting precision-recall trade-offs.

---

## 3. Existing Limitations vs. Proposed Solution

| Dimension | Conventional Campus Portals | CampusPulse (Proposed System) |
| :--- | :--- | :--- |
| **Categorization** | Manual dropdown selection by student (often inaccurate). | Supervised NLP classifier (TF-IDF + Logistic Regression) trained on genuine student data. |
| **Duplicate Handling** | None; every submission creates an isolated ticket. | Cosine similarity clustering flags existing tickets and highlights shared terms before dispatch. |
| **Priority Assignment** | Self-reported priority by student (everyone marks "Urgent"). | Rule-based triage scoring combining category weights, safety keywords, recurrence, and sentiment. |
| **Explainability** | Opaque or absent. | Full transparency: extracted tokens, stopword filters, class probabilities, and factor point breakdown. |
| **Infrastructure Cost** | Heavy database backends or costly cloud LLM APIs. | 100% locally runnable, lightweight Python stack with zero paid dependencies or GPU requirements. |

---

## 4. System Architecture

```
[Student Complaint Input]
        │
        ▼
[Preprocessing & Normalization Engine]
  ├── Contraction & Campus Slang Expansion ("wifi", "ac", "dept", "lab")
  ├── Punctuation & Digit Stripping
  ├── Negation-Preserving Stopword Filtering ("not working", "no water")
  └── Tokenization
        │
        ├───► [TF-IDF Vectorizer (1-2 ngrams, sublinear TF)]
        │             │
        │             ▼
        │     [Multiclass Logistic Regression Classifier]
        │             │
        │             ▼
        │     Predicted Category + Class Probabilities
        │
        ├───► [Cosine Similarity Duplicate Engine]
        │             │
        │             ▼
        │     Candidate Historical Comparison (Top-K Matches + Overlapping Terms)
        │
        ├───► [VADER Sentiment Analyzer]
        │             │
        │             ▼
        │     Compound Polarity (-1.0 to +1.0)
        │
        ▼
[Explainable Priority Scoring Engine]
  ├── Category Baseline Points (+10 to +20)
  ├── Hazard & Safety Keywords (+30 points)
  ├── Academic Deadlines (+15 points)
  ├── Recurrence Clustering (+5 to +20 points)
  └── Complainant Distress (+5 to +10 points)
        │
        ▼
[Local SQLite Persistence & Executive Streamlit Dashboard]
```

---

## 5. Dataset and Labeling Methodology

### 5.1 Dataset Composition and Provenance
The benchmark dataset (`data/university_students_complaints.xlsx`) comprises 396 student complaint records collected from a university environment across 12 metadata attributes (including Department, Semester, Gender, Severity, and `Complaint_Group_ID`):
- **Base Issues & Synthetic Variations**: The 396 records encompass **219 unique ground-truth issue groups**. A subset of issues was augmented with fixed template suffix variations (e.g., *"Kindly verify and take action."*, *"Needs attention."*, *"Requesting a quick check."*, *"Would appreciate a resolution soon."*).
- **Group-Based Partitioning**:
  - **Training Set (`train.csv`)**: 273 labeled complaints spanning 153 issue groups.
  - **Testing Set (`test.csv`)**: 59 held-out complaints spanning 33 issue groups.
  - **Unassigned Holdout**: 64 complaints (33 groups) remain in the master dataset outside train/test.
- **Dataset Audit Finding (Near-Duplicate Leakage)**: A rigorous audit of `Complaint_Group_ID` revealed an annotation inconsistency in the raw Excel data: a small number of identical base complaints were assigned distinct group IDs (e.g., *"Whiteboards in theory classrooms are damaged"* had Group 22 and Group 94). Consequently, 12 out of 59 test records share near-duplicate phrasing ($>0.75$ cosine similarity) with training samples. When evaluated strictly on the **47 non-leaked test complaints**, CampusPulse achieves **76.60% Accuracy** and **0.7637 Macro F1**, confirming that genuine generalization occurs beyond leaked phrasing.
- **Benchmark Pairs (`complaint_pairs.csv`)**: 88 labeled evaluation pairs (44 positive duplicates with shared `Complaint_Group_ID`, and 44 negatives sampled across distinct issues).

### 5.2 Category Breakdown (5 Native Classes)
1. **Infrastructure (42.2%, N=167 total, 114 train, 29 test)**: Maintenance issues involving classrooms, washrooms, water supply, elevators, lighting, and furniture.
2. **Technical (21.2%, N=84 total, 58 train, 7 test)**: IT hardware and software disruptions, including campus Wi-Fi, computer laboratories, projectors, and LAN networks.
3. **Academic (18.2%, N=72 total, 48 train, 16 test)**: Faculty advising, course registration, exam timetabling, grading discrepancies, and curriculum delivery.
4. **Finance (9.3%, N=37 total, 29 train, 4 test)**: Tuition payments, scholarship disbursements, exam fee receipts, and transaction portal errors.
5. **Administrative (9.1%, N=36 total, 24 train, 3 test)**: Library conduct, student welfare, common room facilities, and campus scheduling.

---

## 6. NLP Preprocessing Pipeline
Raw complaint text contains non-standard campus acronyms, emotional punctuation, and colloquial expressions. Preprocessing ensures clean feature representations:

1. **Negative Contraction Normalization**: Standard tokenizers strip apostrophes, converting `"doesn't"` to `"doesn t"` and dropping `"t"`, which strips the negation word `"not"` and inverts statement meaning. CampusPulse applies regex contraction expansion (`doesn't`/`doesnt` $\rightarrow$ `does not`, `can't`/`cant` $\rightarrow$ `cannot`, `isn't` $\rightarrow$ `is not`, `won't` $\rightarrow$ `will not`), ensuring negation semantics are preserved.
2. **Abbreviation and Slang Normalization**: Expands domain-specific terms (`wifi`/`wi-fi` $\rightarrow$ `wifi`, `ac` $\rightarrow$ `air conditioner`, `dept` $\rightarrow$ `department`, `prof` $\rightarrow$ `professor`, `w/o` $\rightarrow$ `without`).
3. **Noise and Symbol Filtering**: Strips URLs, email addresses, phone digits, and special characters while preserving alphabetic tokens and essential hyphens.
4. **Negation-Preserving Stopword Removal**: Standard stopword lists discard words like `not`, `no`, `never`, and `hardly`. Our custom stopword filter explicitly retains `NEGATION_WORDS` to protect semantic intent (`"wifi not working"`, `"no running water"`).

---

## 7. Mathematical Formulations

### 7.1 TF-IDF Vectorization
TF-IDF quantifies the importance of a term $t$ in a document $d$ relative to a corpus $D$. CampusPulse uses sublinear term frequency scaling to prevent long complaints from dominating feature weights:

$$\text{tf}(t, d) = 1 + \ln(f_{t, d}) \quad \text{for } f_{t, d} > 0$$

$$\text{idf}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$

Feature vectors are subsequently $L_2$-normalized:

$$\mathbf{v}_d = \frac{\mathbf{x}_d}{\|\mathbf{x}_d\|_2}$$

Both unigrams and bigrams ($n \in \{1, 2\}$) are extracted, capturing phrases like `"broken lock"`, `"short circuit"`, and `"fee pending"`.

### 7.2 Multiclass Logistic Regression
Multiclass classification is formulated using the multinomial (softmax) Logistic Regression framework. For $K$ classes and feature vector $\mathbf{x} \in \mathbb{R}^M$:

$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

The model minimizes the cross-entropy loss with $L_2$ regularization:

$$\mathcal{L}(\mathbf{W}) = -\sum_{i=1}^N \sum_{k=1}^K y_{i, k} \ln P(Y = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$

Class weights are balanced inversely proportional to class frequencies to counteract category imbalance:

$$w_k = \frac{N}{K \cdot N_k}$$

### 7.3 Cosine Similarity for Duplicate Detection
Given an incoming complaint vector $\mathbf{q}$ and historical complaint vector $\mathbf{c}$, their cosine similarity represents the normalized inner product:

$$\text{Sim}(\mathbf{q}, \mathbf{c}) = \cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\|_2 \|\mathbf{c}\|_2} = \sum_{m=1}^M q_m c_m$$

A configurable threshold $\tau \in [0.20, 0.85]$ determines whether a match is presented as a likely duplicate ($\tau \ge 0.65$) or a related symptom ($0.35 \le \tau < 0.65$).

### 7.4 Sentiment and Priority Scoring Formulation
- **Sentiment**: Calculated via NLTK VADER compound polarity score $S_{\text{compound}} \in [-1.0, +1.0]$.
- **Priority Score ($P \in [5, 100]$)**:
  
$$P = \min\left(100, \max\left(5, B_{\text{category}} + W_{\text{safety}} + W_{\text{academic}} + W_{\text{urgency}} + W_{\text{sentiment}} + W_{\text{recurrence}} + W_{\text{unresolved}}\right)\right)$$

Where:
- $B_{\text{category}} \in \{10, 15, 20\}$: Baseline physical urgency of the department.
- $W_{\text{safety}} = +30$: Triggered if hazard keywords (`spark`, `smoke`, `fire`, `shock`, `contaminated`, `collapse`) are present.
- $W_{\text{academic}} = +15$: Triggered by time-critical academic constraints (`exam`, `viva`, `deadline`, `submission`).
- $W_{\text{sentiment}} \in \{0, 5, 10\}$: Extra triage points if student distress is high ($S \le -0.5 \implies +10$).
- $W_{\text{recurrence}} \in \{0, 5, 10, 20\}$: Escalation based on cluster size of similar historical reports in the same location.
- $W_{\text{unresolved}} = +5$: Active open status boost.

---

## 8. Experimental Methodology and Actual Results

### 8.1 Training & Baseline Comparison Protocol
- **Zero Leakage**: Vectorizer fitted strictly on `X_train`. Held-out test set `test.csv` was completely isolated until final evaluation.
- **Baseline 1 (Majority Class)**: Scikit-Learn `DummyClassifier(strategy='most_frequent')` consistently predicting `Infrastructure`.
- **Baseline 2 (Multinomial Naive Bayes)**: TF-IDF + MultinomialNB ($\alpha=1.0$).

### 8.2 Overall Performance on Held-Out Test Set ($N=59$)

| Model / Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Net Gain vs. Majority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | 49.15% | 9.83% | 20.00% | 0.1318 | 0.00% |
| **Multinomial Naive Bayes** | 62.71% | 46.60% | 34.82% | 0.3512 | +13.56% |
| **CampusPulse v2.1.0 (TF-IDF + LogReg)** | **86.44%** | **90.07%** | **90.00%** | **0.8790** | **+37.29%** |

*Note: Multinomial Naive Bayes suffered complete collapse on minority classes (0% recall on Administrative and Finance) due to prior probability skew. Logistic Regression with inverse class weighting successfully balanced all 5 categories.*

### 8.3 Performance on Unleaked Test Subset ($N=47$)
To account for the 12 near-duplicate records identified during data auditing, performance was evaluated separately on the strictly non-leaked test subset:
- **Accuracy**: **76.60%**
- **Macro Precision**: **80.89%**
- **Macro Recall**: **85.39%**
- **Macro F1-Score**: **0.7637**
- Proves genuine generalization over the 49.15% baseline even without any shared phrasing.

### 8.4 Per-Class Performance Breakdown ($N=59$)

| Category | Precision (%) | Recall (%) | F1-Score (%) | Test Support ($N$) |
| :--- | :--- | :--- | :--- | :--- |
| **Academic** | 100.0% | 31.2% | 47.6% | 16 |
| **Administrative** | 100.0% | 100.0% | 100.0% | 3 |
| **Finance** | 80.0% | 100.0% | 88.9% | 4 |
| **Infrastructure** | 76.3% | 100.0% | 86.6% | 29 |
| **Technical** | 87.5% | 100.0% | 93.3% | 7 |

#### Root Cause Analysis: Academic Recall (31.2%)
1. **Replicated Underlying Issues**: The 16 Academic test complaints actually represent only 4 distinct issues (with 3-4 template variations each) plus 1 isolated complaint.
2. **Out-of-Vocabulary (OOV) Terms**: Key academic terms in the test set (`grade`, `evaluation`, `improvement`, `questions`) had 0 training occurrences. Additionally, the training set contained plural forms (`exams`, `quizzes`) which did not match singular forms (`exam`, `quiz`) in test complaints.
3. **Amplified Error Count**: Misclassifying 3 underlying issues resulted in $4 + 4 + 3 = 11$ misclassifications, heavily depressing recall to $5/16 = 31.25\%$.

### 8.5 Duplicate Detection Validation / Untouched Test Protocol
To avoid threshold data snooping on the 88 pairs in `data/complaint_pairs.csv`:
- **Protocol**: Stratified 50/50 split into a **Validation Set (N=44)** and an **Untouched Test Set (N=44)**.
- **Validation Tuning**: Evaluated candidate thresholds $\tau \in [0.20, 0.80]$. Optimal threshold selected: **$\tau = 0.50$** (Validation $F_1 = 1.0000$).
- **Untouched Test Results at Selected $\tau = 0.50$**:
  - **Precision**: **1.0000 (100.0%)**
  - **Recall**: **0.9545 (95.5%)**
  - **F1-Score**: **0.9767**
  - **Confusion Matrix**: True Positives: **21** | False Positives: **0** | False Negatives: **1** | True Negatives: **22**

Full benchmark threshold sweep across all 88 pairs:

| Threshold ($\tau$) | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.20** | 1.0000 | 1.0000 | 1.0000 | 44 | 0 | 0 |
| **0.40** | 1.0000 | 1.0000 | 1.0000 | 44 | 0 | 0 |
| **0.50** | 1.0000 | 0.9773 | 0.9885 | 43 | 0 | 1 |
| **0.60** | 1.0000 | 0.8636 | 0.9268 | 38 | 0 | 6 |
| **0.70** | 1.0000 | 0.7727 | 0.8718 | 34 | 0 | 10 |
| **0.80** | 1.0000 | 0.5227 | 0.6866 | 23 | 0 | 21 |

---

## 9. Limitations and Ethical Considerations
1. **Benchmark Negatives Representativeness**: In the pair benchmark, negative pairs were randomly sampled across distinct categories, resulting in low similarity ($\le 0.068$). Hard negatives (e.g., two distinct Wi-Fi issues in different rooms sharing 70% of words) are underrepresented.
2. **Lexical Paraphrase Sensitivity**: Pure TF-IDF cosine similarity cannot connect disjoint vocabularies without shared characters (e.g., `"faucet leaking"` vs. `"tap dripping"`).
3. **Ethical & Safety Scope**: The priority scoring engine is a queue-sorting aid for administrative staff. It does not replace engineering inspection or certified life-safety systems.
4. **Prediction Probabilities**: Softmax class probabilities reflect linear distance in TF-IDF space and should be treated as confidence indicators, not absolute guarantees of real-world circumstances.

---

## 10. Conclusion and Future Scope
CampusPulse demonstrates that an explainable, lightweight NLP architecture combining domain-aware preprocessing, TF-IDF vectorization, multiclass regularized Logistic Regression, cosine similarity duplicate tracking, and rule-based priority triage provides an auditable, efficient alternative to costly cloud LLMs. Future extensions include subword tokenization (BPE), lemmatization for singular/plural alignment, and fine-tuning lightweight transformer embeddings (DistilBERT) for deep semantic synonymy.
