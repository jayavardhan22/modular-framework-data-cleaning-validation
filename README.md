# 🚗 A Modular Framework for Data Cleaning and Validation
### MSc Dissertation Project – Dublin Business School (2024–2025)
**Author:** Jayavardhan Premnath  
**Dataset:** [Motor Vehicle Collisions – Crashes](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes)

---

## 🎯 Core Objective
The aim of this project is to develop a **modular, automated framework** that can:
- Clean, validate, and standardize real-world crash data.
- Detect and handle anomalies, missing values, and inconsistencies.
- Optionally apply **machine learning** methods to validate or impute errors.

This project combines **data preprocessing**, **ML-based data quality checks**, and **cross-domain validation** to ensure reliable and ethically handled datasets.

---

## ⚙️ Framework Modules

| Module | Tool / Library | Description |
|:--------|:----------------|:-------------|
| **1. Rule Validation** | Great Expectations | Applies rule-based checks (nulls, ranges, duplicates). |
| **2. Schema Enforcement** | Pandera | Ensures field consistency, type validation, and schema integrity. |
| **3. Anomaly Detection** | Scikit-learn (Isolation Forest, DBSCAN) | Detects outliers or unusual patterns in crash data. |
| **4. Integration & Reporting** | Pandas / Python | Merges cleaned data and generates validation reports. |
| **5. Visualization (optional)** | Streamlit | Displays data quality dashboards interactively. |

---

## 🧩 Cleaning and Validation Workflow
1. **Rule-based Validation** – Identify missing values, incorrect ranges, duplicates.  
2. **Schema Validation** – Verify data types and constraints with Pandera.  
3. **Anomaly Detection** – Use ML models (Isolation Forest, DBSCAN).  
4. **Integration** – Combine cleaned and validated data into a processed dataset.  
5. **Visualization** – Generate interactive dashboards and reports.

---

## 🧰 Tools & Libraries
- **Python 3.x**
- **Pandas, NumPy**
- **Great Expectations**
- **Pandera**
- **Scikit-learn**
- **Matplotlib / Seaborn**
- **Streamlit (optional dashboard)**

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/modular-framework-data-cleaning-validation.git
   cd modular-framework-data-cleaning-validation
