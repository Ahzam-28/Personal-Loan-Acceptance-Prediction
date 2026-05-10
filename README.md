# 📞Personal Loan Acceptance Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> A machine-learning project that predicts which customers will **accept a personal loan / term-deposit offer** so the bank's marketing team can target the right people, save call-centre time, and lift conversion rates.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Objective](#-objective)
3. [Dataset](#-dataset)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#%EF%B8%8F-installation--setup)
6. [How to Run](#-how-to-run)
7. [Workflow](#-workflow)
8. [Exploratory Data Analysis](#-exploratory-data-analysis)
9. [Modelling](#-modelling)
10. [Results](#-results)
11. [Feature Importance](#-feature-importance)
12. [Key Insights](#-key-insights)
13. [Business Recommendations](#-business-recommendations)
14. [Skills Demonstrated](#-skills-demonstrated)
15. [Future Improvements](#-future-improvements)
16. [Author](#-author)

---

## 🎯 Project Overview

Marketing departments waste enormous resources calling customers who will **never** say yes. If a bank could predict — *before* picking up the phone — which customers are most likely to accept a personal-loan or term-deposit offer, it could:

- 📞 Reduce wasted call-centre minutes
- 💰 Lower customer-acquisition cost per conversion
- 😊 Stop annoying uninterested customers
- 📈 Lift overall campaign conversion rate

This project builds a **binary classifier** that predicts campaign acceptance (`y = yes / no`) and surfaces the **customer profiles** the bank should prioritise.

---

## 🎯 Objective

- Train a model to **predict who will accept** a personal-loan / term-deposit offer.
- Identify the **demographics, financial profile, and contact patterns** linked to acceptance.
- Compare **Logistic Regression** vs **Decision Tree** classifiers on accuracy *and* ROC-AUC.
- Translate model output into **actionable targeting rules** for the marketing team.

---

## 📊 Dataset

**Source:** [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) — UCI Machine Learning Repository

| Property | Value |
|---|---|
| Rows | 4,521 |
| Columns | 17 |
| Target | `y` (binary: `yes` / `no`) |
| Class balance | ≈ 18 % yes / 82 % no |
| Type | Tabular, mixed (numeric + categorical) |
| Separator | **Semicolon (`;`)** — UCI standard format |

> ⚠️ **Important:** The file is **semicolon-separated**, so it is loaded with `pd.read_csv(..., sep=";")`.

### Feature Dictionary

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Customer age in years |
| `job` | Categorical | Job category (admin, blue-collar, technician, retired, student, …) |
| `marital` | Categorical | `married` / `single` / `divorced` |
| `education` | Categorical | `primary` / `secondary` / `tertiary` / `unknown` |
| `default` | Binary | Has credit in default? (`yes` / `no`) |
| `balance` | Numeric | Average yearly account balance (€) |
| `housing` | Binary | Has a housing loan? (`yes` / `no`) |
| `loan` | Binary | Has a personal loan? (`yes` / `no`) |
| `contact` | Categorical | Contact type (`cellular` / `telephone` / `unknown`) |
| `day` | Numeric | Day of the month of last contact |
| `month` | Categorical | Month of last contact (`jan` – `dec`) |
| `duration` | Numeric | Last contact duration in seconds ⚠️ *leakage risk — see below* |
| `campaign` | Numeric | Contacts performed during this campaign |
| `pdays` | Numeric | Days since previous contact (−1 = never) |
| `previous` | Numeric | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous campaign (`success` / `failure` / `other` / `unknown`) |
| **`y`** | **Target** | **`yes` = subscribed, `no` = did not** |

> ⚠️ **A note on `duration`:** call duration is *partially a result* of acceptance — long calls happen because the customer is interested. For a real-world targeting model `duration` should be **excluded** to avoid data leakage. It is kept here for completeness so the EDA matches the standard UCI version of the project.

---

## 📂 Project Structure

```
Task_5_Loan_Acceptance/
│
├── 📓 Task_5_Loan_Acceptance.ipynb   # Main notebook with full analysis
├── 📊 bank_marketing.csv             # Bank Marketing dataset (4.5k rows, ; separated)
├── 📄 README.md                      # You are here
└── 📋 requirements.txt               # Python dependencies (optional)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip / conda

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or use a `requirements.txt`:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter
```

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# Launch Jupyter
jupyter notebook Task_5_Loan_Acceptance.ipynb
```

Then **Run All Cells** ( `Cell → Run All` ).
Make sure `bank_marketing.csv` is in the same directory as the notebook.

---

## 🔄 Workflow

```
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│  1. Load Data      │ ─► │  2. Inspect Data   │ ─► │  3. EDA — who      │
│  (sep=';')         │    │     (4.5k rows)    │    │     accepts?       │
└────────────────────┘    └────────────────────┘    └─────────┬──────────┘
                                                              │
┌────────────────────┐    ┌────────────────────┐    ┌─────────▼──────────┐
│ 7. Visualise Tree  │ ◄─ │  6. Train Models   │ ◄─ │ 4. One-hot Encode  │
│ + Importances      │    │  (LR + Decision    │    │ 5. Scale + Split   │
│                    │    │   Tree, balanced)  │    │    (80 / 20)       │
└──────────┬─────────┘    └────────────────────┘    └────────────────────┘
           │
┌──────────▼──────────┐
│ 8. Targeting Rules  │
│ for Marketing       │
└─────────────────────┘
```

### Pipeline steps
1. **Load** the `bank_marketing.csv` file (semicolon-separated).
2. **Inspect** for nulls, dtypes, target balance.
3. **Explore** demographics (age, job, marital), finances (balance, housing/personal loans), and campaign features (`poutcome`, `contact`, `month`).
4. **One-hot encode** all categorical columns; encode the binary target.
5. **Standard-scale** numeric features for Logistic Regression and **stratified-split** 80 / 20.
6. **Train two models** with `class_weight="balanced"` to handle the 18/82 imbalance:
   - Logistic Regression (max_iter=2000)
   - Decision Tree (max_depth=5)
7. **Visualise the top 3 levels** of the tree and the **top-15 feature importances**.
8. **Translate** model output into **targeting rules** for the marketing team.

---

## 📈 Exploratory Data Analysis

Highlights from the EDA section:

- **Class imbalance:** ~18 % acceptance — the bank converts roughly 1 in 6 calls.
- **Age:** customers **60+** accept at a much higher rate (retired customers have time, savings, and an interest in fixed-income products).
- **Job:** **students** and **retired** customers top the acceptance list; **blue-collar** workers sit near the bottom.
- **Marital status:** **single** customers accept at a slightly higher rate than married.
- **Education:** **tertiary-educated** customers accept more than primary.
- **Housing/personal loan:** customers **without** an existing housing or personal loan accept far more — they have less competing debt service.
- **Previous campaign:** customers whose previous campaign ended in **`success`** are by far the most likely to accept again — past behaviour is the best signal of future behaviour.
- **Contact method:** **cellular** contact strongly outperforms `unknown` and `telephone`.
- **Balance:** customers with healthy positive balances accept more.

Visualisations in the notebook:
- Age distribution split by outcome + acceptance rate by age band
- Acceptance rate by **job** (horizontal bar chart, sorted)
- Outcome by **marital status** and **education**
- Box plot of **balance** by outcome (with quantile-clipped y-axis to handle outliers)
- Outcome by **housing** loan and **personal loan**
- Outcome by **previous campaign result (`poutcome`)**
- Confusion matrices for both models (side-by-side)
- Top-3 levels of the **Decision Tree** visualised with `plot_tree`
- Top-15 **Decision Tree feature importances**
- Top positive & negative **Logistic Regression coefficients**

---

## 🧠 Modelling

| Step | Detail |
|---|---|
| Encoding | `pd.get_dummies(drop_first=True)` for all categoricals; binary encoding for target `y` |
| Scaling | `StandardScaler` (used for Logistic Regression only) |
| Split | `train_test_split(test_size=0.2, random_state=42, stratify=y)` |
| Imbalance | `class_weight="balanced"` on **both** models (compensates for 18 / 82 split) |
| Model 1 | `LogisticRegression(max_iter=2000, class_weight="balanced")` |
| Model 2 | `DecisionTreeClassifier(max_depth=5, class_weight="balanced")` |
| Metrics | `accuracy_score`, `roc_auc_score`, `confusion_matrix`, `classification_report` |

### Why `class_weight="balanced"`?

Without it, both models would learn to predict "no" for almost everyone (because that's correct 82 % of the time) and miss most of the actual yes-customers — exactly the people the marketing team needs to find. Balanced weights force the model to pay equal attention to both classes.

---

## 🏆 Results

| Model | Accuracy | ROC-AUC | Notes |
|---|:---:|:---:|---|
| **Logistic Regression** | **≈ 0.74** | **≈ 0.76** | Highest AUC, best ranker, slightly lower accuracy due to balanced weights |
| **Decision Tree (depth 5)** | **≈ 0.76** | **≈ 0.70** | Slightly higher accuracy, easier to visualise as targeting rules |

> Lower accuracy here is **a good thing** — both models are deliberately trading some accuracy on the easy "no" class to catch more of the rare "yes" customers, which is what the business actually needs.

### Why ROC-AUC matters more than accuracy here

With an 18 / 82 split, a "predict no for everyone" baseline already gets **82 % accuracy** while catching **zero** subscribers. **ROC-AUC** measures the model's ability to *rank* likely-yes customers above likely-no ones — the ranking is exactly what the call list will be sorted by.

### Confusion-matrix interpretation

|  | Predicted no | Predicted yes |
|---|---|---|
| **Actual no** | True Negatives ✅ | False Positives (a few wasted calls) |
| **Actual yes** | False Negatives ❗ (missed conversion) | True Positives ✅ |

False negatives are the costly error — every missed yes-customer is revenue left on the table.

---

## 📊 Feature Importance

Top drivers from the Decision Tree (typical ranking, excluding the leakage-prone `duration`):

| Rank | Feature | Why it matters |
|:---:|---|---|
| 🥇 1 | `poutcome_success` | Past acceptance is the single best predictor of future acceptance |
| 🥈 2 | `housing_yes` (negative) | Customers **without** a housing loan are much more likely to accept |
| 🥉 3 | `loan_yes` (negative) | Customers **without** a personal loan are much more likely to accept |
| 4 | `contact_unknown` (negative) | Unknown contact method underperforms cellular |
| 5 | `age` | Older customers (60+) accept more |
| 6 | `balance` | Healthy positive balances correlate with acceptance |
| 7 | `job_retired` / `job_student` | Both segments over-index on acceptance |
| 8 | `month` | Seasonal patterns (Mar, Sep, Oct, Dec do well) |

---

## 💡 Key Insights

1. 🎯 **Past success predicts future success.** Customers who accepted a previous campaign are the lowest-cost, highest-conversion target. Always call them first.
2. 🏠 **Existing loans crush conversion.** Customers already burdened with a housing or personal loan rarely take on more debt — deprioritise them.
3. 📱 **Cellular > telephone > unknown.** The contact channel itself is a meaningful predictor — invest in keeping mobile numbers up to date.
4. 👴 **Retired and student customers convert disproportionately well.** Retirees have savings to deposit; students respond to education-themed loan offers.
5. 🎓 **Tertiary education** customers are more receptive than primary — likely correlated with income and financial literacy.
6. ⚖️ **Class imbalance must be handled.** Without `class_weight="balanced"`, the model would simply predict "no" for everyone and pass the project's accuracy bar while being completely useless for the marketing team.

---

## 🎯 Business Recommendations

| Priority | Action | Target Segment |
|:---:|---|---|
| 🔴 **High** | Re-target last campaign's converters first | `poutcome = success` |
| 🔴 **High** | Build a focused list of **debt-free customers** | `housing = no` AND `loan = no` |
| 🟡 **Medium** | Run a **retiree-focused** term-deposit campaign | `job = retired` AND `age ≥ 60` |
| 🟡 **Medium** | Always call from a **cellular** number / make sure mobile numbers are current | All segments |
| 🟡 **Medium** | Tertiary-educated, single, healthy-balance customers | `education = tertiary`, `marital = single`, top-quartile balance |
| 🟢 **Low** | Skip cold-calls to customers with active housing + personal loans | `housing = yes` AND `loan = yes` |
| 🟢 **Low** | Avoid blue-collar segment for term-deposit pitches; tailor a different product instead | `job = blue-collar` |

> **Expected impact:** focusing the call list on the top-scoring segments should lift acceptance well above the baseline ~18 % seen across the whole population — and free up call-centre capacity for follow-ups.

---

## 🛠 Skills Demonstrated

- 🐍 **Python** — pandas, NumPy
- 📊 **Data visualisation** — matplotlib, seaborn (count plots, box plots, decision-tree visualisation, importance bars, coefficient bars)
- 🧹 **Data preparation** — semicolon-separated CSV handling, target encoding
- 🔢 **Feature encoding** — one-hot encoding with `pd.get_dummies(drop_first=True)`
- ⚖️ **Feature scaling** — `StandardScaler` (and knowing when *not* to use it for trees)
- 🧠 **Machine learning** — Logistic Regression, Decision Tree classifiers
- ⚖️ **Imbalanced classification** — `class_weight="balanced"` and the accuracy-vs-AUC trade-off
- 📐 **Model evaluation** — accuracy, ROC-AUC, confusion matrix, classification report
- 🌳 **Decision-tree interpretation** — visualising splits with `plot_tree`
- 🔍 **Feature importance & coefficient analysis** — turning model internals into business rules
- 🚨 **Data-leakage awareness** — flagging `duration` as post-hoc in production scoring
- 📝 **Communication** — translating model output into a prioritised targeting list

---

## 🚀 Future Improvements

- 🚫 **Drop `duration`** and re-run the whole pipeline to get a leakage-free, deployable model.
- 🌲 Try **Random Forest, Gradient Boosting, XGBoost, LightGBM** — usually the best tabular models.
- ⚖️ Compare `class_weight="balanced"` against **SMOTE** / **ADASYN** oversampling.
- 🔍 **Hyperparameter tuning** with `GridSearchCV` or **Optuna**.
- 📈 Add **Precision-Recall curves** — more informative than ROC for imbalanced data.
- 🧪 **Stratified k-fold cross-validation** instead of a single 80/20 split.
- 🔬 Use **SHAP values** for per-customer explanations the call-centre can read.
- 💸 **Cost-sensitive evaluation** — tie predictions to call-cost (€) and conversion-value (€) to optimise lift, not accuracy.
- 📊 **Lift / gain charts** — show marketing the conversion uplift at the top 10 / 20 / 30 % of the call list.
- 🌐 Deploy as a **Streamlit dashboard** that ranks today's call list automatically.

---

## 👤 Author

- Mohammad Ahzam Hassan

---

## 🙏 Acknowledgements

- **UCI Machine Learning Repository** — for hosting the original Bank Marketing dataset.
- **scikit-learn** — for the modelling and evaluation tools used throughout.

---

<p align="center">⭐ If you found this project helpful, consider giving it a star! ⭐</p>
