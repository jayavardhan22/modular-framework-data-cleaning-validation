🚦 Traffic Collision Data Quality Validation

Hybrid Rule-Based & Machine Learning Framework

🔹 MSc Data Analytics – Applied Research Project  
🔹 Production-style data quality validation pipeline

⸻

🔍 Problem

Real-world traffic collision datasets often contain:
	•	Missing or invalid coordinates
	•	Duplicate records
	•	Logical inconsistencies in casualty counts
	•	Irregular spatial and numerical patterns

These issues reduce trust in analytics, dashboards, and ML models.

⸻

💡 Solution

A modular, explainable data quality validation framework that combines:
	•	🧱 Schema validation (structure & data types)
	•	✅ Rule-based checks (deterministic, auditable)
	•	🤖 Unsupervised anomaly detection (pattern-based)

Designed to scale, reduce manual preprocessing, and preserve data integrity.

🧠 Architecture

Raw Data
   ↓
Schema Validation
   ↓
Rule-Based Validation
   ↓
Anomaly Detection (IF, LOF, DBSCAN)
   ↓
Explainable Validation Reports + Dashboard

⸻

🤖 Models Used
	•	Isolation Forest – primary anomaly detector
	•	Local Outlier Factor (LOF) – local density anomalies
	•	DBSCAN – spatial density anomalies
	•	Random Forest – benchmark only

⸻

📊 Key Results
	•	Isolation Forest
	•	Precision: 88.5%
	•	Recall: 83.7%
	•	F1-Score: 86.0%
	•	Error Detection Rate: 82.4%

✔ High anomaly coverage  
✔ Strong interpretability  
✔ Reduced manual data cleaning  

⸻

🖥️ Streamlit Dashboard
	•	End-to-end execution of validation pipeline
	•	Visualisation of anomalies & rule violations
	•	Metrics for model performance and detection rate

⸻

🛠️ Tech Stack
	•	Python
	•	Pandas / NumPy
	•	Scikit-learn
	•	Streamlit
	•	Matplotlib / Seaborn

⸻

📁 Repository Structure

├── streamlit_dashboard.py
├── validation_pipeline/
├── data/
├── outputs/
├── notebooks/
└── README.md

🎯 Engineering Highlights
	•	Modular, reusable pipeline design
	•	No black-box corrections (flag, don’t fix)
	•	Designed for real-world public-sector data
	•	ML + rules combined for better coverage

⸻

🎓 Author

Jayavardhan Premnath  
MSc Data Analytics | Data Engineering & ML  
Dublin Business School

⸻

⭐ Built with a production-first mindset for data quality engineering.

⸻
