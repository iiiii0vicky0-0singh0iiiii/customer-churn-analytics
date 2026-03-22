<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=200&section=header&text=Customer%20Churn%20Analytics&fontSize=38&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=End-to-End%20ML%20System%20%7C%20Multi-Model%20Comparison%20%7C%20Explainable%20AI&descAlignY=58&descSize=15" width="100%"/>

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-00C7B7?style=for-the-badge)](https://shap.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-006600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-A78BFA?style=for-the-badge)](LICENSE)

<br/>

> **Predict. Explain. Retain.**  
> A production-ready, end-to-end customer churn intelligence system powered by Machine Learning, SHAP Explainability, and an interactive Streamlit dashboard.

<br/>

**[🚀 Live Demo](#)** &nbsp;·&nbsp; **[📊 Explore Dataset](#-dataset)** &nbsp;·&nbsp; **[🐛 Report Bug](https://github.com/iiiii0vicky0-0singh0iiiii/customer-churn-analytics/issues)** &nbsp;·&nbsp; **[💡 Request Feature](https://github.com/iiiii0vicky0-0singh0iiiii/customer-churn-analytics/issues)**

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [ML Models](#-ml-models--auto-selection)
- [Explainable AI](#-explainable-ai-shap)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Run](#️-installation--run-locally)
- [How It Works](#-how-it-works)
- [Results & Performance](#-results--performance)
- [Features](#-features)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Customer churn is one of the most critical business KPIs — losing a customer costs **5–25×** more than retaining one. This project delivers a **full-stack ML analytics pipeline** that:

- Ingests and preprocesses real customer data
- Trains and compares **multiple ML models** side-by-side
- **Auto-selects the best model** based on performance metrics
- Explains predictions using **SHAP (SHapley Additive exPlanations)**
- Visualizes everything through a clean, interactive **Streamlit dashboard**

```
Business Problem  →  Data Pipeline  →  Multi-Model Training
       ↓                                        ↓
  SHAP Insights  ←  Auto Model Selection  ←  Evaluation
       ↓
  Streamlit Dashboard  →  Actionable Retention Strategies
```

---

## ✨ Key Highlights

| 🏆 Capability | 💡 Description |
|---|---|
| 🤖 **Multi-Model Comparison** | Train & compare Random Forest, XGBoost, Logistic Regression, SVM, and more |
| ⚡ **Auto Model Selection** | Automatically picks the best-performing model based on AUC-ROC / F1 |
| 🧠 **Explainable AI (SHAP)** | Global & local explanations — understand *why* a customer churns |
| 📊 **Interactive Dashboard** | Full Streamlit UI with filters, charts, and prediction interface |
| 🔁 **End-to-End Pipeline** | From raw CSV to deployed predictions in one unified system |
| 📈 **Rich Visualizations** | Confusion matrix, ROC curves, feature importance, SHAP waterfall plots |
| 🎯 **Single-Customer Prediction** | Input customer data → get real-time churn probability + explanation |

---

## 🎬 Demo

<div align="center">

| Dashboard View | SHAP Explanation | Model Comparison |
|---|---|---|
| 📊 Interactive filters & KPIs | 🔍 Feature-level prediction explanation | 📉 Side-by-side model metrics |

> 🔗 **[Launch Live App →](#)** *(hosted on Streamlit Cloud)*

</div>

---

## 📊 Dataset

> Customer demographic and service usage data with a binary churn target.

| Feature Category | Columns Included |
|---|---|
| 👤 **Demographics** | Gender, Senior Citizen, Partner, Dependents |
| 📋 **Account Info** | Tenure, Contract Type, Payment Method |
| 💰 **Charges** | Monthly Charges, Total Charges |
| 🌐 **Internet Services** | Internet Service, Online Security, Streaming TV/Movies |
| 📞 **Phone Services** | Phone Service, Multiple Lines |
| 🎯 **Target Variable** | **Churn** (Yes / No) |

**Dataset Stats:**

```
Total Records   :  ~7,000+ customers
Features        :  20 input features
Target Classes  :  Binary — Churn: Yes / No
Class Imbalance :  Handled via SMOTE / class weighting
```

---

## 🤖 ML Models & Auto Selection

The system trains and evaluates **multiple classifiers** and selects the best automatically:

```python
models = {
    "Logistic Regression" : LogisticRegression(),
    "Random Forest"        : RandomForestClassifier(),
    "XGBoost"              : XGBClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Gradient Boosting"    : GradientBoostingClassifier(),
    "K-Nearest Neighbors"  : KNeighborsClassifier(),
}
```

**Auto Selection Logic:**

```
1. Train all models on train split
2. Evaluate each → AUC-ROC, F1-Score, Precision, Recall, Accuracy
3. Rank by weighted score (AUC-ROC priority)
4. Auto-select & save best model as best_model.pkl
5. Use best model for dashboard predictions
```

---

## 🧠 Explainable AI (SHAP)

> *"A model you can't explain is a model you can't trust."*

SHAP provides **transparent, human-readable explanations** at both global and local levels:

| Explanation Type | What It Shows |
|---|---|
| 🌍 **Global Feature Importance** | Which features most influence churn across all customers |
| 🔍 **Local Waterfall Plot** | Why *this specific customer* is predicted to churn |
| 🐝 **Beeswarm Plot** | Distribution of SHAP values across the dataset |
| 📊 **Dependence Plot** | How a single feature affects churn probability |

```python
import shap

explainer    = shap.TreeExplainer(best_model)
shap_values  = explainer.shap_values(X_test)

# Global summary
shap.summary_plot(shap_values, X_test)

# Local explanation for one customer
shap.waterfall_plot(shap.Explanation(values=shap_values[0], ...))
```

---

## 🧠 Tech Stack

<div align="center">

**Core ML & Data**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-00C7B7?style=for-the-badge)

**Visualization & Dashboard**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**Tools & Deployment**

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

</div>

---

## 📂 Project Structure

```
customer-churn-analytics/
│
├── 📁 data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│   └── processed/
│       └── churn_processed.csv                     # Cleaned & encoded data
│
├── 📁 models/
│   ├── best_model.pkl                              # Auto-selected best model
│   ├── scaler.pkl                                  # Feature scaler
│   └── label_encoders.pkl                          # Saved encoders
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb                                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb                      # Feature engineering
│   └── 03_model_training.ipynb                     # Model comparison & SHAP
│
├── 📁 src/
│   ├── preprocess.py                               # Data cleaning pipeline
│   ├── train.py                                    # Model training & auto selection
│   ├── evaluate.py                                 # Metrics & comparison
│   └── explain.py                                  # SHAP explanation logic
│
├── app.py                                          # Streamlit dashboard entry point
├── requirements.txt                                # Python dependencies
├── config.yaml                                     # Project configuration
└── README.md                                       # Project documentation
```

---

## ⚙️ Installation & Run Locally

### Prerequisites

- Python **3.9+**
- pip package manager
- Git

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/iiiii0vicky0-0singh0iiiii/customer-churn-analytics.git
```

**2. Move into the project directory**
```bash
cd customer-churn-analytics
```

**3. Create and activate a virtual environment**
```bash
# Create environment
python -m venv venv

# Activate — Linux / macOS
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

**4. Install all dependencies**
```bash
pip install -r requirements.txt
```

**5. Train the models (optional — pretrained model included)**
```bash
python src/train.py
```

**6. Launch the Streamlit dashboard**
```bash
streamlit run app.py
```

> 🌐 Open your browser at `http://localhost:8501`

---

## 🔄 How It Works

```
┌─────────────────────────────────────────────────────┐
│                  DATA PIPELINE                       │
│  Raw CSV  →  Cleaning  →  Encoding  →  Scaling       │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│               MODEL TRAINING                         │
│  6 Models trained  →  Cross-validated  →  Evaluated  │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│             AUTO MODEL SELECTION                     │
│  Best AUC-ROC model  →  Saved as best_model.pkl      │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│               SHAP EXPLAINABILITY                    │
│  Global importance  +  Local waterfall per customer  │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│             STREAMLIT DASHBOARD                      │
│  Overview · Prediction · Model Compare · SHAP Viz    │
└─────────────────────────────────────────────────────┘
```

---

## 📈 Results & Performance

> Metrics shown are representative values — your results may vary based on train/test split.

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| 🥇 **XGBoost** | **82.1%** | **80.4%** | **78.9%** | **79.6%** | **0.874** |
| 🥈 Random Forest | 80.5% | 78.2% | 76.1% | 77.1% | 0.856 |
| 🥉 Gradient Boosting | 79.8% | 77.9% | 75.3% | 76.6% | 0.851 |
| Logistic Regression | 77.2% | 74.5% | 72.8% | 73.6% | 0.829 |
| SVM | 76.9% | 74.1% | 71.2% | 72.6% | 0.821 |
| KNN | 74.3% | 71.8% | 68.4% | 70.1% | 0.802 |

**Top Churn Predictors (SHAP):**
```
1. 📄 Contract Type       — Month-to-month contracts → highest churn risk
2. 💰 Monthly Charges     — Higher charges → higher churn probability
3. ⏳ Tenure              — Short tenure → higher churn risk
4. 🔒 Online Security     — No security service → more likely to churn
5. 🌐 Internet Service    — Fiber optic users churn more than DSL
```

---

## 🖥️ Features

### 📊 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Overview** | KPI cards — total customers, churn rate, avg tenure, avg charges |
| 🔍 **EDA** | Interactive charts — churn by contract, service type, charges distribution |
| 🤖 **Model Comparison** | Side-by-side metrics table + ROC curve plot for all 6 models |
| 🎯 **Predict** | Input customer details → get churn probability + confidence gauge |
| 🧠 **SHAP Explainer** | Global beeswarm + local waterfall plot for any prediction |
| 📥 **Export** | Download predictions and reports as CSV |

---

## 🤝 Contributing

Contributions make this project better — all PRs are welcome!

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/NewFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m "Add: NewFeature description"
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/NewFeature
   ```
5. **Open** a Pull Request with a clear description

### 💡 Ideas for Contribution
- [ ] Add deep learning model (PyTorch / TensorFlow)
- [ ] Add real-time data ingestion via API
- [ ] Docker containerization
- [ ] Add LIME explanations alongside SHAP
- [ ] Unit tests for preprocessing pipeline

---

## 📄 License

Distributed under the **MIT License** — see [`LICENSE`](LICENSE) for full details.

---

## 📬 Contact

<div align="center">

[![Email](https://img.shields.io/badge/Gmail-indianarmysniper@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:indianarmysniper@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-iiiii0vicky0--0singh0iiiii-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/iiiii0vicky0-0singh0iiiii)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=120&section=footer&animation=fadeIn" width="100%"/>

*Built with ❤️ by **Vicky Kumar Singh** — if this helped you, please ⭐ star the repo!*

</div>
