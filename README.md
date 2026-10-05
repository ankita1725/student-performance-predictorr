# 🎓 Student Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, educational machine learning web application that predicts secondary student academic performance categories based on historical academic history, attendance, and study habits using the UCI Student Performance dataset.

> **⚠️ Responsible AI Notice:** This application is built strictly as an educational machine learning demonstration. Predicted outcomes are probabilistic estimates derived from historical patterns and must never be used for formal academic grading, disciplinary sanctions, or definitive assessments of student ability.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Dataset Attribution & Description](#-dataset-attribution--description)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Application Screenshots](#-application-screenshots)
- [Project Directory Structure](#-project-directory-structure)
- [Local Installation & Setup (Windows)](#-local-installation--setup-windows)
- [Model Training & Testing](#-model-training--testing)
- [Launching the Web Application](#-launching-the-web-application)
- [Deployment to Streamlit Community Cloud](#-deployment-to-streamlit-community-cloud)
- [Evaluation Methodology & Benchmark Results](#-evaluation-methodology--benchmark-results)
- [Ethical Considerations & Limitations](#-ethical-considerations--limitations)
- [Future Improvements](#-future-improvements)

---

## 🌟 Overview

Identifying students who may need additional academic support early in the school year enables educators, counselors, and advisors to intervene with targeted tutoring, mentoring, or study habit adjustments. 

**Student Performance Predictor** translates this real-world educational challenge into a supervised multi-class classification problem. Using audited demographic, attendance, and interim grading records, the system forecasts which performance tier a student is likely to achieve:

- **Needs Support** (Grade 0–9): Below the standard passing grade threshold.
- **Developing** (Grade 10–14): Standard passing performance with foundational competence.
- **Strong** (Grade 15–20): High mastery and academic distinction.

---

## 🚀 Key Features

1. **Interactive Multi-Page Streamlit Dashboard:**
   - **Home Dashboard:** High-level project KPIs, model status, pipeline overview, and ethical disclaimers.
   - **Student Prediction:** Form with boundary-validated inputs, class probability breakdown, and contextual diagnostic notes.
   - **Exploratory Data Analysis (EDA):** Interactive Plotly visualizations depicting grade distributions, attendance patterns, and correlation heatmaps.
   - **Model Evaluation:** Direct side-by-side comparison between the Majority-Class Baseline, Logistic Regression, and Random Forest Classifier, complete with interactive confusion matrices.
   - **About the Project:** Academic literature attribution, feature documentation, and boundary notes.

2. **Audited ML Pipeline with Zero Data Leakage:**
   - Preprocessing (`StandardScaler`) is fitted strictly on the training partition and transformed on test data via scikit-learn `Pipeline`.
   - The target attribute `G3` is strictly isolated from the feature matrix `X`.

3. **Multi-Model Benchmark:**
   - Evaluates a Dummy Majority Baseline against Logistic Regression and Random Forest.
   - Documents evaluation rationale using Macro F1-score to ensure robust classification across imbalanced classes.

4. **Production-Ready Code Quality:**
   - Clean modular architecture (`src/data_loader.py`, `src/model.py`, `train_model.py`, `app.py`).
   - Automated test suite using `pytest` covering schema validation, edge cases, probability invariants, and model benchmarks.

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.10+ (tested on Python 3.13)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `joblib`
- **Interactive Visualization:** `plotly`, `altair`
- **Web Application Framework:** `streamlit`
- **Testing Framework:** `pytest`
- **Network & SSL Security:** `certifi` (ensures reliable downloads across Windows platforms)

---

## 📊 Dataset Attribution & Description

The application utilizes the **Student Performance Data Set** from the **UCI Machine Learning Repository**:

- **Repository Link:** [UCI Student Performance Dataset (ID: 320)](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Course Focus:** Mathematics (`student-mat.csv`, 395 records)
- **Original Authors:** Paulo Cortez and Alice Silva (University of Minho, Portugal)
- **Citation:**
  > P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008. EUROSIS, ISBN 978-9077381-39-7.

### Selected Features

| Feature | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `age` | Numeric | 15–22 | Student age in years |
| `studytime` | Categorical | 1–4 | Weekly study hours (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h) |
| `failures` | Numeric | 0–3 | Count of past class failures |
| `absences` | Numeric | 0–30 | Total recorded days absent from school |
| `G1` | Numeric | 0–20 | First-period examination mark |
| `G2` | Numeric | 0–20 | Second-period examination mark |

---

## 🧠 Machine Learning Architecture

```
[User Input Features]
        │
        ▼
[Schema Validation] (Bounds verification: age, studytime, failures, absences, G1, G2)
        │
        ▼
[StandardScaler Transformer] (Applies training set mean & standard deviation)
        │
        ▼
[Random Forest Classifier] (Ensemble of 100 decision trees, max_depth=5)
        │
   ┌────┴───────────────────────────┐
   ▼                                ▼
[Predicted Class]         [Class Probabilities]
(Needs Support /          (P[Needs Support], P[Developing],
 Developing / Strong)      P[Strong] via predict_proba)
```

---

## 🖼️ Application Screenshots

### 1. Home Dashboard & KPI Overview
*(Screenshot placeholder: `docs/screenshots/dashboard_home.png`)*
> Visualizes summary metrics (395 students, 6 features, 87.3% accuracy, 89.4% Macro F1), how the prediction pipeline functions, and ethical use guidelines.

### 2. Interactive Student Prediction Form & Probability Gauge
*(Screenshot placeholder: `docs/screenshots/prediction_page.png`)*
> Two-column responsive input interface with immediate probability bars, color-coded badges, and human-readable pedagogical explanations.

### 3. Exploratory Data Analysis (EDA)
*(Screenshot placeholder: `docs/screenshots/eda_page.png`)*
> Interactive Plotly charts examining grade distributions, attendance vs. final achievement, and correlation heatmaps.

### 4. Model Evaluation & Confusion Matrices
*(Screenshot placeholder: `docs/screenshots/evaluation_page.png`)*
> Side-by-side benchmark table comparing Majority Baseline, Logistic Regression, and Random Forest, accompanied by heatmaps of prediction matrices.

---

## 📁 Project Directory Structure

```
student-performance-predictor/
├── .streamlit/
│   └── config.toml          # Custom academic blue UI styling
├── data/
│   ├── student-mat.csv      # Local cached copy of UCI Mathematics dataset
│   └── README.md            # Dataset origin, schema, and licensing notes
├── models/
│   ├── best_model.joblib    # Serialized champion model (Random Forest)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── evaluation_results.json # Verified held-out test metrics and confusion matrices
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Safe dataset loading, SSL fallback, schema validation
│   └── model.py             # Scikit-learn pipelines, cross-evaluation, inference
├── tests/
│   ├── __init__.py
│   └── test_model.py        # Automated test suite (9 comprehensive pytest tests)
├── .gitignore               # Python and virtual environment exclusions
├── app.py                   # Streamlit web application entry-point
├── requirements.txt         # Project package dependencies
├── train_model.py           # Standalone model training and evaluation script
└── README.md                # Project documentation and guide
```

---

## 💻 Local Installation & Setup (Windows)

Follow these steps to set up and run the project locally on Windows:

### Step 1: Clone or Navigate to the Repository

Open **PowerShell** or **Command Prompt**:

```powershell
cd C:\Users\INDIAN\.gemini\antigravity\scratch\student-performance-predictor
```

### Step 2: Create a Virtual Environment

It is recommended to use a virtual environment to isolate dependencies:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt
.\venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies

Upgrade `pip` and install packages from `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Model Training & Testing

### Train the Models

Run the standalone training script to validate the dataset, train all classification models, and serialize the champion pipeline:

```powershell
python train_model.py
```

**Expected Console Output:**
```
======================================================================
 STUDENT PERFORMANCE PREDICTOR - MODEL TRAINING PIPELINE
======================================================================

[1/4] Loading and validating dataset from 'data\student-mat.csv'...

[2/4] Dataset statistics:
      Total records: 395
      Training set:  316 samples (80%)
      Held-out test: 79 samples (20%)
      Features:      age, studytime, failures, absences, G1, G2

[3/4] Model Evaluation Comparison (Held-out Test Set):
----------------------------------------------------------------------
Model                    | Accuracy   | Macro Prec  | Macro Rec  | Macro F1  
----------------------------------------------------------------------
Majority Baseline        |    48.10% |     16.03% |    33.33% |    21.65%
Logistic Regression      |    86.08% |     87.11% |    89.14% |    87.81%
Random Forest            |    87.34% |     89.28% |    90.01% |    89.42%
----------------------------------------------------------------------

[4/4] Model Selection Decision:
      Champion Model: Random Forest
      Rationale:      Random Forest achieved higher Macro F1 (0.8942) and Accuracy (0.8734)...
[SUCCESS] Artifacts successfully serialized to 'models/'.
======================================================================
```

### Run Automated Tests

Execute the comprehensive test suite using `pytest`:

```powershell
pytest -v
```

All 9 test cases should pass:
- Dataset verification & schema compliance
- Target grade boundary categorization
- Full target class representation
- Verification of zero data leakage (G3 absent from X)
- Scikit-learn Pipeline construction
- Prediction output schema & probability summing
- Differential prediction sensitivity across student profiles
- Input validation & bounds rejection
- Superiority over the majority-class baseline

---

## 🌐 Launching the Web Application

Launch the Streamlit interactive dashboard with:

```powershell
streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

---

## ☁️ Deployment to Streamlit Community Cloud

This project is structured for 1-click deployment on **Streamlit Community Cloud**:

1. **Push the repository to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: complete student performance predictor project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/student-performance-predictor.git
   git push -u origin main
   ```

2. **Connect to Streamlit Community Cloud:**
   - Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **New app**.
   - Select your repository: `<your-username>/student-performance-predictor`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - Click **Deploy!**

3. **Cloud Runtime Execution:**
   - Streamlit Cloud automatically installs packages from `requirements.txt`.
   - The app dynamically detects if models exist; if missing, it automatically downloads the dataset and trains the models on first startup.

---

## 📈 Evaluation Methodology & Benchmark Results

### Evaluation Strategy
- **Held-out Split:** 80% training (316 students), 20% test (79 students).
- **Stratification:** Splitting was stratified on `y` (`stratify=y`) to maintain identical class ratios across train and test sets.
- **Fixed Random State:** `random_state=42` ensures exact reproducibility.

### Benchmark Results Table

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 48.10% | 16.03% | 33.33% | 21.65% |
| **Logistic Regression** | 86.08% | 87.11% | 89.14% | 87.81% |
| **Random Forest (Champion)** | **87.34%** | **89.28%** | **90.01%** | **89.42%** |

### Confusion Matrix (Random Forest Test Set)

```
                     Predicted
                 Needs Support  Developing  Strong
True Needs Support     23           3          0
     Developing         7          31          0
     Strong             0           0         15
```

- **Strong Category:** 100% recall and precision (15 out of 15 correctly classified).
- **Needs Support Category:** 88.5% recall (23 out of 26 identified correctly), critical for proactive intervention.

---

## 🛡️ Ethical Considerations & Limitations

1. **Heavy Association with Previous Grades (G1 & G2):**
   Interim exam marks are the strongest empirical predictors of final grades. However, assuming that a student with low early grades cannot recover risks creating self-fulfilling prophecies. The tool should be framed as highlighting opportunities for support rather than defining an immutable ceiling.

2. **Geographical & Contextual Specificity:**
   The dataset was collected from two public secondary schools in the Alentejo region of Portugal during the 2005–2006 academic year. Educational norms, grading distributions, and socio-economic variables are distinct to that region and era.

3. **Probability Calibration Notice:**
   Ensemble tree probabilities represent empirical voting frequencies across tree leaves. While informative, they should not be interpreted as mathematically calibrated Bayesian posterior probabilities.

---

## 🔮 Future Improvements

- [ ] Incorporate SHAP (SHapley Additive exPlanations) for per-prediction local feature attribution.
- [ ] Implement probability calibration via `CalibratedClassifierCV`.
- [ ] Add support for the Portuguese Language dataset (`student-por.csv`) for cross-subject transfer learning.
- [ ] Introduce a synthetic data generator to simulate academic progress trajectories over multiple academic semesters.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
The underlying dataset remains attributed to the original authors (P. Cortez and A. Silva, 2008) under the terms of the UCI Machine Learning Repository.
