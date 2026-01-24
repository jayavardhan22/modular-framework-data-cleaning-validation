A Machine Learning and Rule-Based Approach for Data Quality Validation in Traffic Collision Data
📌 Project Overview
This repository contains the source code and documentation for the MSc Dissertation project: "A Machine Learning and Rule-Based Approach for Data Quality Validation in Traffic Collision Data".


Traffic collision datasets are safety-critical but often plagued by missing values, inconsistencies, and logical errors. Traditional manual cleaning is time-consuming and prone to error. This project proposes and evaluates a hybrid data quality framework that combines rule-based validation (Pandera) with Machine Learning techniques (Isolation Forest, DBSCAN) to automate the detection of anomalies in large-scale datasets.




🎯 Research Objectives
The primary objective is to design a functional artifact (Design Science Research) that:

Combines domain-specific logic rules with unsupervised machine learning.

Provides an automated, scalable solution for data validation.

Visualizes data quality issues through an interactive dashboard.

🛠️ Architecture & Methodology
The project follows an End-to-End Hybrid Framework Flow:


Data Loading & Initial Cleaning: Standardization of column names and types.



Schema Validation: Utilizing Pandera to enforce data types, nullability, and value ranges.


Rule-Based Validation:


Logical Rules: Checking consistency (e.g., pedestrians injured cannot exceed total injured).


Geospatial Rules: Constraining Latitude/Longitude to NYC bounding boxes.

Unsupervised Anomaly Detection:


Isolation Forest: The primary model for identifying global outliers.


DBSCAN: Used for detecting spatial clusters and noise.


Supervised Learning Benchmark: A Random Forest classifier is used to evaluate the performance of the anomaly detection.


Visualization: A Streamlit dashboard for real-time analytics.

📂 Dataset

Source: NYC Motor Vehicle Collisions Dataset (NYC Open Data Portal).


Volume: ~1 million+ records.


Features: 29 attributes including temporal, geospatial, and casualty counts.
