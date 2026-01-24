🚦 Traffic Collision Data Quality Validation

A Machine Learning & Rule-Based Framework

📘 MSc Data Analytics – Applied Research Project
🎓 Dublin Business School

⸻

📌 Project Summary

Traffic collision data plays a crucial role in road safety analysis, urban planning, and policy decision-making. However, real-world collision datasets often suffer from data quality issues such as missing values, duplicates, logical inconsistencies, and anomalous records.

This project presents a hybrid, modular, and explainable data quality validation framework that combines:
	•	✅ Rule-based validation
	•	🧱 Schema validation
	•	🤖 Machine learning–based anomaly detection

The framework improves data reliability, reduces manual cleaning effort, and ensures transparent and auditable validation for traffic collision datasets.

⸻

🎯 Research Aim

To design and evaluate an automated yet explainable validation framework that:
	•	Detects structural and logical data issues
	•	Identifies anomalous collision records
	•	Preserves original data (no forced corrections)
	•	Enhances trust in downstream analytics

⸻

🧠 Validation Framework Overview

The framework follows a step-by-step modular pipeline:

1️⃣ Schema Validation
	•	Verifies required fields exist
	•	Checks data types and structure
	•	Flags schema violations without modifying records

2️⃣ Rule-Based Validation

Detects explicit data quality issues such as:
	•	Missing or invalid latitude/longitude
	•	Duplicate collision IDs
	•	Negative or illogical casualty counts
	•	Mismatches between total and category-wise injuries

🟢 Fully transparent and easy to audit

3️⃣ Machine Learning–Based Anomaly Detection

Unsupervised models identify irregular patterns not captured by rules:
	•	Isolation Forest – global anomaly detection
	•	Local Outlier Factor (LOF) – local density anomalies
	•	DBSCAN – spatial density-based anomalies
	•	Random Forest – used only as a benchmark

4️⃣ Explainable Outputs
	•	Clear separation between:
	•	Rule violations
	•	Structural issues
	•	Statistical anomalies
	•	No black-box decisions
	•	No automatic deletion or correction
