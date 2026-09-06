# 🛡️ Sentinel — Fraud Detection & Transaction Risk Analysis

> **Rewind. Investigate. Defend.**

Sentinel is an end-to-end fraud investigation and transaction risk analysis system built around the IEEE-CIS Fraud Detection dataset. It combines **machine learning risk scoring**, **behavioral transaction timelines**, **evidence synthesis**, and **human decision-making** into a single investigation workflow.

---

## 🚨 Problem

Fraud detection systems often stop at a binary prediction: *"Fraud" or "Not Fraud."* But a real investigation requires more:

- How risky is this transaction?
- How does it compare with historical behavior?
- Is there unusual transaction velocity?
- What evidence supports the risk assessment?
- Should the case be contested, reviewed, or accepted as a loss?
- Can the final decision be audited?

**Sentinel bridges the gap between prediction and investigation.**

---

## 💡 What Sentinel Does

```text
Transaction
     │
     ▼
┌─────────────────────┐
│  ML Fraud Scoring    │
│    XGBoost Model     │
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│  Behavioral History  │
│  Fraud Time Machine  │
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│  Evidence Synthesis  │
│    AI + Rule Logic   │
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│  Merchant Decision   │
│ Contest / Review /   │
│     Accept Loss      │
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│    Audit Logging     │
└─────────────────────┘
```

---

## 🔍 Key Features

### 1. 🤖 Machine Learning Fraud Detection

An XGBoost classifier estimates fraud probability using transaction and behavioral attributes: transaction amount, card attributes, address attributes, distance information, and historical transaction behavior.

Two approaches were compared on the validation set before final training:

| Model | Validation F1 |
|---|---|
| Class-Weighted XGBoost | 0.2787 |
| **SMOTE + XGBoost** | **0.4537** |

SMOTE + XGBoost achieved the higher validation F1 and was selected for final training.

### 2. 🎯 Cost-Aware Decision Threshold

Fraud datasets are highly imbalanced, so a default 0.50 cutoff isn't necessarily appropriate. Thresholds were evaluated on the validation set:

| Threshold | Precision | Recall | F1-Score | % Fraud Caught | % False Positives |
|---|---|---|---|---|---|
| 0.30 | 0.1764 | 0.6738 | 0.2796 | 67.4% | 11.40% |
| 0.50 | 0.4062 | 0.5138 | 0.4537 | 51.4% | 2.72% |
| **0.70 (Selected)** | **0.6266** | **0.3671** | **0.4630** | **36.7%** | **0.79%** |
| 0.85 | 0.8004 | 0.2795 | 0.4143 | 28.0% | 0.25% |
| 0.90 | 0.8467 | 0.2485 | 0.3843 | 24.9% | 0.16% |

**Selected threshold: 0.70** — best F1 balance while keeping false positives under 1%. The cost table below shows why a lower threshold could also be justified depending on business priorities.

### 3. ⏳ Fraud Time Machine

A transaction isn't investigated in isolation. Sentinel reconstructs the historical timeline of the entity using a privacy-conscious proxy. Two candidate identifiers were evaluated on real cardinality and history coverage before choosing one:

| Candidate | Unique Entities | Entities with ≥3 Transactions | Coverage |
|---|---|---|---|
| `card1` | 13,553 | 8,419 | 62.1% |
| **`card1` + `addr1`** | 39,974 | **18,334** | 45.9% |

**Selected proxy: `card1` + `addr1`** — chosen for the larger absolute number of entities with enough history to build meaningful timelines, even though its percentage coverage is lower; `card1` alone groups more transactions together, inflating its per-entity hit rate at the cost of merging unrelated cardholders.

The system analyzes previous transaction timing, frequency, historical amounts, amount deviation, and transactions within the prior 24 hours — computed strictly from transactions before the current one, with no future-data leakage.

### 4. 📊 Behavioral Risk Indicators

- **24h Transaction Count** — transactions associated with the entity in the prior 24 hours.
- **Spike Ratio** — current amount ÷ the entity's historical average.
- **Time Since Last Transaction** — seconds elapsed since the entity's previous transaction.

Cutoffs for what counts as a "spike" or "high velocity" are derived empirically from the actual fraud-vs-legitimate medians in the training data, not fixed arbitrarily.

### 5. 🧠 AI Evidence Synthesis

Converts model outputs and behavioral signals into investigation-oriented explanations. Uses an LLM when an API key is available; falls back to a deterministic rule-based explanation otherwise, so the full demo runs locally without any external service.

**Design principle:** the system is instructed to describe only the evidence actually supplied to it — it does not invent transaction history or investigative facts. The LLM never independently decides whether a transaction is fraudulent; that judgment stays with the ML model and the human investigator.

### 6. ⚖️ Human-in-the-Loop Decision

The merchant/investigator chooses the final action: **Contest Dispute**, **Send to Review**, or **Accept Loss**. The ML system provides evidence and a recommendation; the decision stays with the human.

### 7. 📜 Audit Trail

Every submitted decision is recorded in an append-only audit log containing: transaction ID, amount, ML score, decision threshold, AI recommendation, rule-based recommendation, merchant's final decision, disagreement flag, and evidence summary.

---

## 📈 Model Performance

Evaluated on an untouched 20% test split saved **before** any model tuning or resampling.

| Metric | Result |
|---|---|
| ROC-AUC | 0.8837 |
| Precision | 0.6695 |
| Recall | 0.4038 |
| F1-Score | 0.5038 |
| Decision Threshold | 0.70 |

These are internal held-out test results, not Kaggle leaderboard results.

### 📊 Confusion Matrix (at threshold 0.70)

| | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | 113,151 (True Negatives) | 824 (False Positives) |
| **Actual Fraud** | 2,464 (False Negatives) | 1,669 (True Positives) |

### 💰 Business Impact Analysis

| | Transactions | Estimated Dollar Impact |
|---|---|---|
| False Positives | 824 | $87,457.26 friction |
| False Negatives | 2,464 | $428,786.38 estimated fraud loss |

These are illustrative estimates based on transaction amounts in the evaluated dataset, not measured real-world losses. The roughly 5:1 ratio of missed-fraud cost to false-positive friction is a reasonable starting point for arguing a lower threshold in a production setting, depending on how the business weighs fraud loss against review overhead.

### 🔍 Failure Case Analysis

A dynamically extracted false-negative example:

```
Transaction ID:              3502995
Transaction Amount:          $35.00
Model Probability Score:     0.1053
Threshold:                   0.70
Amount vs Rolling Avg Ratio: 0.27x
24h Velocity Count:          1
```

This transaction was missed because its amount and spending pattern matched the entity's own baseline behavior — a low-value, low-velocity transaction doesn't trip the model even when it's fraud. Sentinel routes borderline cases to a **Review** path rather than relying solely on automated rejection.

---

## 🧪 Data & Methodology

Sentinel uses the **IEEE-CIS Fraud Detection** dataset, with an entity proxy constructed from available transaction attributes to support historical behavioral analysis.

```text
Raw IEEE-CIS Data
        │
        ▼
Column Selection
        │
        ▼
Entity Proxy Analysis
        │
        ▼
Missing Value Handling
        │
        ▼
Canonical Timeline Features
        │
        ▼
Processed Parquet Dataset
```

### 🧩 Feature Engineering

- **`time_since_last_tx`** — time since the entity's previous transaction.
- **`amt_vs_rolling_avg`** — current amount relative to the average of previous transactions.
- **`tx_count_last_24h`** — prior transactions in the last 24 hours.

All features are generated using only transactions that occurred before the current one, preserving temporal ordering.

---

## 🏗️ Project Structure

```
Razorpay/
│
├── sentinel/
│   ├── app/
│   │   └── streamlit_app.py
│   │
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   │
│   └── src/
│       ├── data/
│       │   └── clean_data.py
│       ├── model/
│       │   ├── train.py
│       │   ├── xgb_fraud_model.joblib
│       │   └── model_meta.json
│       ├── features/
│       │   └── timeline.py
│       ├── evidence/
│       │   └── synthesizer.py
│       └── audit/
│           └── logger.py
│
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Running Sentinel Locally

**1. Clone the repository**
```bash
git clone https://github.com/nithya16o1/Fraud-detection-and-transaction-risk-analysis-project.git
cd Fraud-detection-and-transaction-risk-analysis-project
```

**2. Install dependencies**
```bash
py -m pip install -r requirements.txt
```

**3. Prepare the dataset**

Download the IEEE-CIS Fraud Detection dataset from Kaggle and place the transaction file under `sentinel/data/raw/`. Raw CSVs are excluded from Git via `.gitignore`.

**4. Run the data pipeline**
```bash
py sentinel/src/data/clean_data.py
```
Generates `sentinel/data/processed/cleaned_transactions.parquet`.

**5. Train the model**
```bash
py sentinel/src/model/train.py
```
Creates `xgb_fraud_model.joblib` and `model_meta.json`.

**6. Launch the application**
```bash
py -m streamlit run sentinel/app/streamlit_app.py
```

---

## 🖥️ Application Workflow

1. **Select Transaction** — choose a disputed transaction from the investigation selector.
2. **Review ML Risk** — inspect the fraud probability against the decision threshold.
3. **Rewind History** — use the Fraud Time Machine to inspect historical behavior.
4. **Review Evidence** — examine the synthesized primary and secondary concerns.
5. **Make a Decision** — Contest Dispute / Send to Review / Accept Loss.
6. **Audit** — submit the decision and record it in the audit trail.

---

## 🧪 Testing

Includes a test validating that timeline features never use future transactions when constructing historical context (no data leakage).

```bash
pytest
```

---

## 🔐 Responsible Design

- **No fabricated evidence** — the evidence layer only uses information actually present in the transaction data.
- **No future leakage in timelines** — historical features respect strict transaction ordering.
- **Untouched test evaluation** — final evaluation runs on a test set saved before any tuning or resampling.
- **Human oversight** — the system recommends; the investigator decides.
- **Privacy-conscious entity proxy** — uses a dataset-derived proxy rather than claiming access to a real customer identity.

---

## ⚠️ Limitations

This is a prototype, not a production fraud prevention system.

- The IEEE-CIS dataset is historical and competition-oriented.
- `card1 + addr1` is an entity proxy, not a verified customer identifier.
- Model performance may differ on live production data.
- Business-cost figures are estimates based on transaction values, not measured losses.
- The LLM evidence layer is not a substitute for human investigation.
- No integration with a live payment processor.
- Internal test results should not be read as production performance.

---

## 🔮 Future Improvements

Real-time transaction scoring · production payment gateway integration · more advanced behavioral features · SHAP-based model explanations · cost-sensitive threshold optimization · investigator feedback loops · case prioritization · production monitoring and drift detection · role-based investigator access · database-backed audit storage · real-time alerting.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical computation |
| XGBoost | Fraud classification |
| Scikit-learn | Evaluation and splitting |
| imbalanced-learn | SMOTE experimentation |
| Joblib | Model serialization |
| Plotly | Interactive visualizations |
| Streamlit | Investigation dashboard |
| OpenAI API | Optional evidence synthesis |
| Pytest | Testing |

---

## 🎯 Why Sentinel?

Traditional fraud detection answers *"Is this transaction fraudulent?"* Sentinel answers a more useful question:

*"Why does this transaction look risky, what does its history tell us, and what should an investigator do next?"*

ML prediction + behavioral context + evidence synthesis + human decision-making + auditability — that combination is the core idea behind Sentinel.

---

## 👩‍💻 Author

**Sreenithya Garudammagari** — built as an end-to-end AI/ML fraud investigation project for the Razorpay AI Builder Internship 2026 (Track: AI Risk Manager).

---

## ⭐ Project Highlights

🤖 XGBoost Fraud Detection · 🎯 Validation-Based Threshold Selection · ⏳ Behavioral Fraud Time Machine · 📊 Interactive Risk Visualization · 🧠 AI Evidence Synthesis · ⚖️ Human-in-the-Loop Decisions · 📜 Audit Logging · 💰 Cost-Aware Evaluation · 🧪 Leakage-Aware Testing

---

**Sentinel — Rewind. Investigate. Defend.**