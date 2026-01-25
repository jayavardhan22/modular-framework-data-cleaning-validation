"""
Streamlit Dashboard for Motor Vehicle Collisions Data Quality Framework
Production-ready version without emojis
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import tempfile
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # <-- NEW
from typing import Tuple, Dict, Any, List
import warnings
import gc
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Pandera imports with error handling
try:
    import pandera.pandas as pa
    from pandera.pandas import Column, Check, DataFrameSchema
    PANDERA_AVAILABLE = True
except ImportError:
    try:
        import pandera as pa
        from pandera import Column, Check, DataFrameSchema
        PANDERA_AVAILABLE = True
    except ImportError as e:
        PANDERA_AVAILABLE = False
        PANDERA_ERROR = str(e)
        class pa:
            class DateTime: pass
            class String: pass
            class errors:
                class SchemaWarning: pass
                class SchemaErrors(Exception): pass
        Column = None
        Check = None
        DataFrameSchema = None

# Suppress warnings
if PANDERA_AVAILABLE:
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandera")
    warnings.filterwarnings("ignore", message=".*DataFrame concatenation.*")
    warnings.filterwarnings("ignore", message=".*Importing pandas-specific classes.*")
    try:
        warnings.filterwarnings("ignore", category=pa.errors.SchemaWarning)
    except AttributeError:
        pass

# Page configuration
st.set_page_config(
    page_title="Data Quality Framework Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem 1rem; }
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; }
    [data-testid="stMetricLabel"] { font-size: 1rem; color: #666; }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 { color: white; margin: 0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Constants
GEOSPATIAL_COLS = ['LATITUDE', 'LONGITUDE']
INJURY_COLS = [
    'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
    'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
    'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
    'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED'
]

EXPECTED_COLUMN_NAMES = [
    'CRASH DATE', 'CRASH TIME', 'BOROUGH', 'ZIP CODE', 'LATITUDE', 'LONGITUDE',
    'ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME',
    'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
    'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
    'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
    'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED',
    'CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2',
    'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4',
    'CONTRIBUTING FACTOR VEHICLE 5',
    'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3',
    'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5',
    'COLLISION_ID'
]

# Define Pandera Schema
if PANDERA_AVAILABLE:
    try:
        from pandera import dtypes
        core_schema = pa.DataFrameSchema({
            'COLLISION_ID': Column(dtypes.Int64, nullable=False, checks=Check.greater_than(0)),
            'CRASH DATE': Column(dtypes.DateTime, nullable=False),
            'CRASH TIME': Column(dtypes.String, nullable=False, checks=Check.str_matches(r'^\d{2}:\d{2}$')),
            'NUMBER OF PERSONS INJURED': Column(dtypes.Int32, checks=[Check.greater_than_or_equal_to(0)]),
            'LATITUDE': Column(dtypes.Float64, nullable=True),
            'LONGITUDE': Column(dtypes.Float64, nullable=True),
        }, strict=False)
    except (ImportError, AttributeError):
        try:
            core_schema = pa.DataFrameSchema({
                'COLLISION_ID': Column(np.int64, nullable=False, checks=Check.greater_than(0)),
                'CRASH DATE': Column(pa.DateTime, nullable=False),
                'CRASH TIME': Column(pa.String, nullable=False, checks=Check.str_matches(r'^\d{2}:\d{2}$')),
                'NUMBER OF PERSONS INJURED': Column(np.int32, checks=[Check.greater_than_or_equal_to(0)]),
                'LATITUDE': Column(np.float64, nullable=True),
                'LONGITUDE': Column(np.float64, nullable=True),
            }, strict=False)
        except (AttributeError, TypeError):
            core_schema = pa.DataFrameSchema({
                'COLLISION_ID': Column(int, nullable=False, checks=Check.greater_than(0)),
                'CRASH DATE': Column('datetime64[ns]', nullable=False),
                'CRASH TIME': Column(str, nullable=False, checks=Check.str_matches(r'^\d{2}:\d{2}$')),
                'NUMBER OF PERSONS INJURED': Column(int, checks=[Check.greater_than_or_equal_to(0)]),
                'LATITUDE': Column(float, nullable=True),
                'LONGITUDE': Column(float, nullable=True),
            }, strict=False)
else:
    core_schema = None

# Initialize session state
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_processing' not in st.session_state:
    st.session_state.data_processing = False

def load_and_clean_data(file_name: str) -> pd.DataFrame:
    """Loads and performs initial type conversion and standardization."""
    try:
        df = pd.read_csv(
            file_name,
            low_memory=False,
            on_bad_lines='skip',
            encoding='utf-8',
            header=None
        )
    except FileNotFoundError:
        st.error(f"Error: The file '{file_name}' was not found.")
        return None

    if len(df.columns) != len(EXPECTED_COLUMN_NAMES):
        if len(df.columns) < len(EXPECTED_COLUMN_NAMES):
            df.columns = EXPECTED_COLUMN_NAMES[:len(df.columns)]
        else:
            df = df.iloc[:, :len(EXPECTED_COLUMN_NAMES)]
            df.columns = EXPECTED_COLUMN_NAMES
    else:
        df.columns = EXPECTED_COLUMN_NAMES

    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'], format='%m/%d/%Y', errors='coerce')
    df['CRASH TIME'] = df['CRASH TIME'].astype(str).str.strip().str.slice(0, 5)
    df['CRASH DATETIME'] = pd.to_datetime(
        df['CRASH DATE'].dt.strftime('%Y-%m-%d') + ' ' + df['CRASH TIME'],
        errors='coerce'
    )

    for col in INJURY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int32)

    # NEW: LOC metric for dataset file
    try:
        if 'loc' not in st.session_state:
            with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
                st.session_state.loc = sum(1 for _ in f)
    except Exception:
        pass

    return df.copy().reset_index(drop=True)

def schema_and_rule_validation(df: pd.DataFrame, schema) -> Tuple[pd.DataFrame, Dict[str, Any], int, float]:
    """Applies Pandera schema enforcement and rule-based checks."""
    validation_report = {}
    total_violations_before_correction = 0
    start_time = time.time()

    if schema is not None and PANDERA_AVAILABLE:
        try:
            df = schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            for column, errors in err.failure_cases.groupby('column'):
                validation_report[f"Schema_Violation_Pandera_{column}"] = f"{len(errors)} errors found (Check/Type/Missing)."
                total_violations_before_correction += len(errors)

    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

    lat_violations = df[(df['LATITUDE'] < 40.4) | (df['LATITUDE'] > 41.0)].dropna(subset=['LATITUDE']).shape[0]
    lon_violations = df[(df['LONGITUDE'] < -74.3) | (df['LONGITUDE'] > -73.7)].dropna(subset=['LONGITUDE']).shape[0]

    if lat_violations + lon_violations > 0:
        validation_report["Rule_Violation_Geospatial"] = f"{lat_violations + lon_violations} records outside NYC bounds."
        total_violations_before_correction += (lat_violations + lon_violations)

    total_injured = df['NUMBER OF PERSONS INJURED']
    parts_injured = df['NUMBER OF PEDESTRIANS INJURED'] + df['NUMBER OF CYCLIST INJURED'] + df['NUMBER OF MOTORIST INJURED']
    inconsistency_count = (total_injured < parts_injured).sum()

    if inconsistency_count > 0:
        validation_report["Rule_Violation_Logic_Injury_Count"] = f"{inconsistency_count} records where total injured is less than sum of parts."
        total_violations_before_correction += inconsistency_count

    time_module2_execution = time.time() - start_time
    return df, validation_report, total_violations_before_correction, time_module2_execution

def anomaly_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Applies Isolation Forest and DBSCAN for comparative anomaly detection."""
    performance_metrics = {}

    df['TOTAL_CASUALTIES'] = df[INJURY_COLS].sum(axis=1).astype(np.int32)
    model_features = GEOSPATIAL_COLS + ['TOTAL_CASUALTIES']
    anomaly_data = df[model_features].copy()

    anomaly_data['LATITUDE'] = anomaly_data['LATITUDE'].fillna(anomaly_data['LATITUDE'].median())
    anomaly_data['LONGITUDE'] = anomaly_data['LONGITUDE'].fillna(anomaly_data['LONGITUDE'].median())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(anomaly_data)
    del anomaly_data
    gc.collect()

    dataset_size = len(df)
    if dataset_size > 1000000:
        max_samples = min(100000, dataset_size)
    elif dataset_size > 500000:
        max_samples = min(50000, dataset_size)
    else:
        max_samples = 'auto'

    start_time_if = time.time()
    n_jobs_if = min(4, max(1, int(os.cpu_count() * 0.5))) if hasattr(os, 'cpu_count') else 2

    iso_forest = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_jobs=n_jobs_if,
        max_samples=max_samples,
        verbose=0
    )

    df['ANOMALY_FLAG_IF'] = iso_forest.fit_predict(scaled_data)
    anomalies_if_count = (df['ANOMALY_FLAG_IF'] == -1).sum()
    time_if = time.time() - start_time_if

    performance_metrics['IF_Time_s'] = time_if
    performance_metrics['IF_Anomaly_Count'] = anomalies_if_count
    del iso_forest
    gc.collect()

    if dataset_size > 1000000:
        DBSCAN_SAMPLE_SIZE = 10000
    elif dataset_size > 500000:
        DBSCAN_SAMPLE_SIZE = 20000
    elif dataset_size > 100000:
        DBSCAN_SAMPLE_SIZE = 30000
    else:
        DBSCAN_SAMPLE_SIZE = min(50000, dataset_size)

    DBSCAN_SAMPLE_SIZE = min(DBSCAN_SAMPLE_SIZE, len(scaled_data))
    dbs_sample = scaled_data[:DBSCAN_SAMPLE_SIZE]

    start_time_dbs = time.time()
    dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=2)
    dbscan_labels = dbscan.fit_predict(dbs_sample)

    anomalies_dbs_count = (dbscan_labels == -1).sum()
    time_dbs = time.time() - start_time_dbs

    df['ANOMALY_FLAG_DBSCAN'] = 0
    df.loc[:len(dbscan_labels)-1, 'ANOMALY_FLAG_DBSCAN'] = dbscan_labels

    performance_metrics['DBSCAN_Time_s'] = time_dbs
    performance_metrics['DBSCAN_Anomaly_Count_Sample'] = anomalies_dbs_count

    del dbscan, dbs_sample, scaled_data, dbscan_labels
    gc.collect()

    return df, performance_metrics

def run_baseline_comparison(df: pd.DataFrame) -> Dict[str, float]:
    """Compares the framework's rule-based validation against standard Pandas methods."""
    baseline_metrics = {}
    start_time_base = time.time()

    baseline_df = df.copy()
    for col in INJURY_COLS:
        if col in baseline_df.columns:
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce').fillna(0)

    violation_col = 'NUMBER OF PERSONS KILLED'
    if violation_col in baseline_df.columns:
        baseline_violations_count = (baseline_df[violation_col] > 10).sum()
    else:
        baseline_violations_count = 0

    time_base = time.time() - start_time_base
    baseline_metrics['Baseline_Time_s'] = time_base
    baseline_metrics['Baseline_Violations'] = baseline_violations_count

    del baseline_df
    gc.collect()
    return baseline_metrics

def compute_metrics_and_summarize(
    df: pd.DataFrame,
    validation_report: Dict[str, Any],
    performance_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    total_violations_before_correction: int,
    time_module2_execution: float
) -> pd.DataFrame:
    """Computes all required metrics (P, R, F1, EDR) and generates a summary table."""
    is_rule_violation = np.zeros(len(df), dtype=int)

    total_injured = df['NUMBER OF PERSONS INJURED']
    parts_injured = df['NUMBER OF PEDESTRIANS INJURED'] + df['NUMBER OF CYCLIST INJURED'] + df['NUMBER OF MOTORIST INJURED']
    is_rule_violation[(total_injured < parts_injured).values] = 1

    is_geospatial_violation = (
        (df['LATITUDE'] < 40.4) | (df['LATITUDE'] > 41.0) |
        (df['LONGITUDE'] < -74.3) | (df['LONGITUDE'] > -73.7)
    ).fillna(False).values
    is_rule_violation[is_geospatial_violation] = 1

    y_true = is_rule_violation
    results = []

    anomaly_models = {
        'Isolation Forest': (df['ANOMALY_FLAG_IF'].map({-1: 1, 1: 0}).values, performance_metrics['IF_Time_s']),
        'DBSCAN (Sample)': (((df['ANOMALY_FLAG_DBSCAN'] == -1).astype(int)).values[:len(y_true)], performance_metrics['DBSCAN_Time_s']),
        'Random Forest (Supervised)': (None, None),  # placeholder, will be updated
    }

    # NEW: Train Random Forest on rule-based labels as supervised baseline
    try:
        feature_cols = GEOSPATIAL_COLS + ['TOTAL_CASUALTIES'] if 'TOTAL_CASUALTIES' in df.columns else GEOSPATIAL_COLS
        X_rf = df[feature_cols].copy()
        X_rf = X_rf.fillna(X_rf.median(numeric_only=True))
        y_rf = y_true

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf
        )

        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_start = time.time()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_rf)
        rf_time = time.time() - rf_start

        anomaly_models['Random Forest (Supervised)'] = (y_pred_rf, rf_time)
    except Exception:
        pass

    for name, (y_pred, exec_time) in anomaly_models.items():
        if y_pred is None or exec_time is None:
            continue
        if 'DBSCAN' in name:
            y_true_temp = y_true[:len(y_pred)]
        else:
            y_true_temp = y_true

        prec = precision_score(y_true_temp, y_pred, zero_division=0)
        rec = recall_score(y_true_temp, y_pred, zero_division=0)
        f1 = f1_score(y_true_temp, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true_temp, y_pred, labels=[0, 1]).ravel()
        edr = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append([
            name, f"{exec_time:.4f}s", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{edr:.4f}"
        ])

    results.append([
        "Rule-Based Validation (Pandera + Logic)",
        f"{time_module2_execution:.4f}s",
        "N/A", "1.0000", "N/A", "1.0000"
    ])

    results.append([
        "Baseline (Pandas Proxy)",
        f"{baseline_metrics['Baseline_Time_s']:.4f}s",
        "N/A", "N/A", "N/A", "N/A"
    ])

    cols = ["Module/Model", "Execution Time (Speed)", "Precision", "Recall", "F1-Score", "EDR"]
    results_df = pd.DataFrame(results, columns=cols).sort_values(by="F1-Score", ascending=False, na_position='last').reset_index(drop=True)
    return results_df

# Main Dashboard
def main():
    if not PANDERA_AVAILABLE:
        st.error(f"Import Error: Pandera is not available. Error: {PANDERA_ERROR}. Please install: pip install pandera>=0.26.0")
        st.stop()
    
    st.markdown("""
    <div class="main-header">
        <h1>Motor Vehicle Collisions Data Quality Framework</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Comprehensive Data Quality Analysis & Anomaly Detection Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Find dataset
    dataset_path = None
    if os.path.exists("dataset.csv"):
        dataset_path = "dataset.csv"
    elif os.path.exists("./dataset.csv"):
        dataset_path = "./dataset.csv"
    
    # Step 1: Load data if not loaded
    if dataset_path and st.session_state.raw_df is None and not st.session_state.data_processing:
        st.session_state.data_processing = True
        with st.spinner("Loading dataset..."):
            raw_df = load_and_clean_data(dataset_path)
            if raw_df is not None:
                st.session_state.raw_df = raw_df
                st.session_state.data_processing = False
                st.success(f"Data loaded successfully: {len(raw_df):,} records")
                st.rerun()
            else:
                st.session_state.data_processing = False
                st.error(f"Failed to load data from {dataset_path}")
    
    # Step 2: Show Process Data button if data is loaded but not processed
    if (st.session_state.raw_df is not None and 
        not st.session_state.data_loaded and 
        not st.session_state.data_processing):
        
        st.info("Data loaded successfully. Click the button below to run the full analysis framework.")
        
        # Use a form to ensure button state is properly handled
        with st.form("process_data_form", clear_on_submit=False):
            submitted = st.form_submit_button("Process Data", type="primary", use_container_width=True)
            if submitted:
                st.session_state.data_processing = True
                st.session_state.data_loaded = False
                st.rerun()
    
    # Step 3: Process data if flag is set
    if (st.session_state.raw_df is not None and 
        st.session_state.data_processing and 
        not st.session_state.data_loaded):
        
        st.info("Processing started. Please wait...")
        
        try:
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            raw_df = st.session_state.raw_df
            
            # Module 2: Schema and Rule Validation
            status_text.text("Running schema validation...")
            progress_bar.progress(15)
            
            if core_schema is None:
                st.error("Pandera schema is not available. Please install pandera: pip install pandera>=0.26.0")
                st.session_state.data_processing = False
                progress_bar.empty()
                status_text.empty()
                st.stop()
            
            rule_validated_df, validation_summary, total_violations, time_module2 = schema_and_rule_validation(
                raw_df.copy(), core_schema
            )
            
            # Module 3: Anomaly Detection
            status_text.text("Running anomaly detection (this may take a few minutes)...")
            progress_bar.progress(30)
            anomaly_flagged_df, performance_metrics = anomaly_detection(rule_validated_df.copy())
            
            # Module 4: Baseline Comparison
            status_text.text("Running baseline comparison...")
            progress_bar.progress(70)
            baseline_metrics = run_baseline_comparison(raw_df.copy())
            
            # Module 5: Metrics and Summary
            status_text.text("Computing performance metrics...")
            progress_bar.progress(85)
            final_results_df = compute_metrics_and_summarize(
                anomaly_flagged_df,
                validation_summary,
                performance_metrics,
                baseline_metrics,
                total_violations,
                time_module2
            )
            
            progress_bar.progress(95)
            status_text.text("Saving results...")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Store results
            st.session_state.processed_df = anomaly_flagged_df
            st.session_state.results = {
                'validation_summary': validation_summary,
                'performance_metrics': performance_metrics,
                'baseline_metrics': baseline_metrics,
                'total_violations': total_violations,
                'results_df': final_results_df,
                'execution_time': execution_time
            }
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            
            st.session_state.data_loaded = True
            st.session_state.data_processing = False
            progress_bar.empty()
            status_text.empty()
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.session_state.data_processing = False
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Dashboard</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Data Quality Framework</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### Data Status")
    
    if st.session_state.data_loaded and st.session_state.raw_df is not None:
        st.sidebar.success("Data Loaded")
        st.sidebar.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
            <p style="margin: 0;"><strong>Records:</strong> {len(st.session_state.raw_df):,}</p>
            <p style="margin: 0.5rem 0 0 0;"><strong>Processed:</strong> Ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Reload Data", help="Reload dataset.csv and reprocess", use_container_width=True):
            st.session_state.raw_df = None
            st.session_state.processed_df = None
            st.session_state.results = None
            st.session_state.data_loaded = False
            st.session_state.data_processing = False
            st.rerun()
        
        if st.session_state.results:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Quick Stats")
            results = st.session_state.results
            st.sidebar.metric("Violations", f"{results['total_violations']:,}")
            st.sidebar.metric("Anomalies", f"{results['performance_metrics']['IF_Anomaly_Count']:,}")
            
            
    elif st.session_state.data_processing:
        st.sidebar.info("Processing...")
        st.sidebar.markdown("Please wait while we analyze your data.")
    elif dataset_path:
        st.sidebar.warning("Ready to Load")
        st.sidebar.info(f"Found: `{dataset_path}`")
    else:
        st.sidebar.error("File Not Found")
        st.sidebar.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
            <p style="margin: 0;">Place <code>dataset.csv</code> in the same directory as this dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("""
    - Dashboard Overview - Key metrics
    - Performance Metrics Overview
    - Validation Report
    - Interactive Visualizations
    - Data Exploration & Export
    """)
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("About"):
        st.markdown("""
        **Data Quality Framework**
        
        Comprehensive analysis tool for Motor Vehicle Collisions data.
        
        **Features:**
        - Schema validation
        - Anomaly detection
        - Performance metrics
        - Interactive visualizations
        """)
    
    # Display results
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        processed_df = st.session_state.processed_df
        
        st.markdown("### Dashboard Overview")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_records = len(df)
        loc_value = st.session_state.get('loc', None)  # NEW
        
        with col1:
            st.metric("Total Records", f"{total_records:,}")
            if loc_value is not None:  # NEW
                st.caption(f"LOC (dataset.csv): {loc_value:,}")  # NEW
        
        if st.session_state.results:
            total_violations = st.session_state.results['total_violations']
            violations_pct = (total_violations / total_records * 100) if total_records > 0 else 0
            
            with col2:
                st.metric("Total Violations", f"{total_violations:,}", delta=f"{violations_pct:.2f}%", delta_color="inverse")
                st.progress(min(violations_pct / 10, 1.0))
            
            anomalies = st.session_state.results['performance_metrics']['IF_Anomaly_Count']
            anomalies_pct = (anomalies / total_records * 100) if total_records > 0 else 0
            
            with col3:
                st.metric("Anomalies (IF)", f"{anomalies:,}", delta=f"{anomalies_pct:.2f}%", delta_color="inverse")
                st.progress(min(anomalies_pct / 5, 1.0))
            
            exec_time = st.session_state.results['execution_time']
            with col4:
                st.metric("Processing Time", f"{exec_time:.1f}s")
            
            

        st.markdown("---")

        if st.session_state.results:
            st.markdown("### Performance Metrics Overview")
            results_df = st.session_state.results['results_df']
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            try:
                best_f1 = results_df[results_df['F1-Score'].astype(str) != 'N/A'].copy()
                if not best_f1.empty:
                    def safe_float_convert(x):
                        try:
                            return float(x) if str(x) != 'N/A' and str(x) != 'nan' else 0.0
                        except (ValueError, TypeError):
                            return 0.0
                    
                    best_f1['F1_Value'] = best_f1['F1-Score'].apply(safe_float_convert)
                    if best_f1['F1_Value'].notna().any():
                        best_idx = best_f1['F1_Value'].idxmax()
                        best_model = best_f1.loc[best_idx, 'Module/Model']
                        best_f1_score = best_f1['F1_Value'].max()
                        
                        with metrics_col1:
                            st.metric("Best Model", str(best_model).split('(')[0].strip())
                        with metrics_col2:
                            st.metric("Best F1-Score", f"{best_f1_score:.4f}")
            except Exception:
                pass
            
            try:
                fastest = results_df[results_df['Execution Time (Speed)'].astype(str) != 'N/A'].copy()
                if not fastest.empty:
                    def safe_time_convert(x):
                        try:
                            time_str = str(x).replace('s', '').strip()
                            return float(time_str) if time_str and time_str != 'N/A' and time_str != 'nan' else float('inf')
                        except (ValueError, TypeError, AttributeError):
                            return float('inf')
                    
                    fastest['Time'] = fastest['Execution Time (Speed)'].apply(safe_time_convert)
                    if fastest['Time'].notna().any() and (fastest['Time'] != float('inf')).any():
                        fastest_idx = fastest['Time'].idxmin()
                        fastest_model = fastest.loc[fastest_idx, 'Module/Model']
                        fastest_time = fastest['Time'].min()
                        
                        with metrics_col3:
                            st.metric("Fastest Model", str(fastest_model).split('(')[0].strip())
                        with metrics_col4:
                            st.metric("Fastest Time", f"{fastest_time:.2f}s")
            except Exception:
                pass
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.expander("View Detailed Performance Metrics Table", expanded=False):
                styled_df = results_df.copy()
                numeric_cols = ['Precision', 'Recall', 'F1-Score', 'EDR']
                
                valid_numeric_cols = []
                for col in numeric_cols:
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].replace('N/A', np.nan)
                        styled_df[col] = pd.to_numeric(styled_df[col], errors='coerce')
                        if styled_df[col].notna().any():
                            valid_numeric_cols.append(col)
                
                if valid_numeric_cols:
                    try:
                        styled_result = styled_df.style.background_gradient(
                            subset=valid_numeric_cols,
                            cmap='RdYlGn',
                            vmin=0,
                            vmax=1,
                            axis=0
                        )
                        for col in valid_numeric_cols:
                            styled_result = styled_result.format({col: '{:.4f}'}, na_rep='N/A')
                        st.dataframe(styled_result, use_container_width=True, hide_index=True)
                    except Exception:
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            st.markdown("### Validation Report")
            validation_summary = st.session_state.results['validation_summary']
            
            if validation_summary:
                schema_violations = {k: v for k, v in validation_summary.items() if 'Schema' in k}
                rule_violations = {k: v for k, v in validation_summary.items() if 'Rule' in k}
                
                col_val1, col_val2 = st.columns(2)
                
                with col_val1:
                    st.markdown("#### Schema Violations")
                    if schema_violations:
                        for key, value in schema_violations.items():
                            st.error(f"**{key.replace('Schema_Violation_Pandera_', '')}:** {value}")
                    else:
                        st.success("No schema violations")
                
                with col_val2:
                    st.markdown("#### Rule Violations")
                    if rule_violations:
                        for key, value in rule_violations.items():
                            st.warning(f"**{key.replace('Rule_Violation_', '')}:** {value}")
                    else:
                        st.success("No rule violations")
                
                violation_counts = []
                violation_labels = []
                
                for key, value in validation_summary.items():
                    try:
                        numbers = re.findall(r'[\d,]+', str(value))
                        if numbers:
                            count_str = numbers[0].replace(',', '')
                            count = int(count_str)
                            violation_counts.append(count)
                            label = key.replace('Schema_Violation_Pandera_', '').replace('Rule_Violation_', '')
                            violation_labels.append(label)
                    except (ValueError, IndexError):
                        continue
                
                if violation_counts and violation_labels:
                    violation_df = pd.DataFrame({
                        'Violation Type': violation_labels,
                        'Count': violation_counts
                    })
                    violation_df = violation_df.sort_values('Count', ascending=False)
                    
                    use_log_scale = max(violation_counts) / min(violation_counts) > 100 if violation_counts else False
                    
                    fig_violations = px.bar(
                        violation_df,
                        x='Violation Type',
                        y='Count',
                        title='Violation Types Distribution',
                        labels={'Count': 'Count', 'Violation Type': 'Violation Type'},
                        color='Count',
                        color_continuous_scale='Reds',
                        text='Count'
                    )
                    fig_violations.update_traces(texttemplate='%{text:,}', textposition='outside')
                    
                    if use_log_scale:
                        fig_violations.update_layout(yaxis_type="log")
                    
                    fig_violations.update_layout(
                        height=400,
                        template='plotly_white',
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_violations, use_container_width=True)
            else:
                st.success("No validation violations detected! Your data quality is excellent!")

            st.markdown("---")

            st.markdown("### Interactive Visualizations")
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Geospatial Analysis", 
                "Anomaly Distribution", 
                "Performance Metrics",
                "Temporal Analysis",
                "Data Quality Summary"
            ])
            
            with tab1:
                st.subheader("Geospatial Anomaly Detection")
                if processed_df is not None:
                    df_plot = processed_df.copy()
                    
                    sample_size = st.slider("Sample Size", 1000, min(50000, len(df_plot)), 10000, 1000)
                    if len(df_plot) > sample_size:
                        df_plot = df_plot.sample(n=sample_size, random_state=42)
                    
                    col_filter1, col_filter2 = st.columns(2)
                    with col_filter1:
                        show_normal = st.checkbox("Show Normal Records", value=True)
                    with col_filter2:
                        show_anomalies = st.checkbox("Show Anomalies", value=True)
                    
                    if not show_normal:
                        df_plot = df_plot[df_plot['ANOMALY_FLAG_IF'] == -1]
                    if not show_anomalies:
                        df_plot = df_plot[df_plot['ANOMALY_FLAG_IF'] == 1]
                    
                    fig = px.scatter(
                        df_plot,
                        x='LONGITUDE',
                        y='LATITUDE',
                        color=df_plot['ANOMALY_FLAG_IF'].map({-1: 'Anomaly', 1: 'Normal'}),
                        color_discrete_map={'Anomaly': '#FF4444', 'Normal': '#4444FF'},
                        hover_data=['COLLISION_ID', 'CRASH DATE', 'NUMBER OF PERSONS INJURED'],
                        title='Geospatial Anomaly Detection (Isolation Forest)',
                        labels={'ANOMALY_FLAG_IF': 'Status'},
                        opacity=0.6,
                        size_max=10
                    )
                    fig.update_layout(height=600, title_font_size=16, showlegend=True, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.info(f"**Total Points:** {len(df_plot):,}")
                    with col_stat2:
                        anomalies_count = (df_plot['ANOMALY_FLAG_IF'] == -1).sum()
                        st.warning(f"**Anomalies:** {anomalies_count:,}")
                    with col_stat3:
                        normal_count = (df_plot['ANOMALY_FLAG_IF'] == 1).sum()
                        st.success(f"**Normal:** {normal_count:,}")

            with tab2:
                st.subheader("Anomaly Distribution Analysis")
                if processed_df is not None:
                    df_plot = processed_df.copy()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if_counts = df_plot['ANOMALY_FLAG_IF'].value_counts()
                        labels = ['Normal', 'Anomaly']
                        values = [if_counts.get(1, 0), if_counts.get(-1, 0)]
                        colors = ['#2E7D32', '#C62828']
                        
                        fig1 = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.4,
                            marker_colors=colors,
                            textinfo='label+percent+value',
                            textfont_size=12
                        )])
                        fig1.update_layout(title='Isolation Forest Distribution', height=400, showlegend=True, template='plotly_white')
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig1_bar = px.bar(
                            x=labels,
                            y=values,
                            color=labels,
                            color_discrete_map={'Normal': '#2E7D32', 'Anomaly': '#C62828'},
                            title='Isolation Forest Counts',
                            labels={'x': 'Status', 'y': 'Count'},
                            text=values
                        )
                        fig1_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
                        fig1_bar.update_layout(height=300, template='plotly_white')
                        st.plotly_chart(fig1_bar, use_container_width=True)
                    
                    with col2:
                        dbs_counts = df_plot['ANOMALY_FLAG_DBSCAN'].value_counts()
                        dbs_labels = ['Normal', 'Anomaly']
                        dbs_values = [dbs_counts.get(0, 0) + sum([v for k, v in dbs_counts.items() if k > 0]), 
                                     dbs_counts.get(-1, 0)]
                        dbs_colors = ['#1976D2', '#D32F2F']
                        
                        fig2 = go.Figure(data=[go.Pie(
                            labels=dbs_labels,
                            values=dbs_values,
                            hole=0.4,
                            marker_colors=dbs_colors,
                            textinfo='label+percent+value',
                            textfont_size=12
                        )])
                        fig2.update_layout(title='DBSCAN Distribution (Sampled)', height=400, showlegend=True, template='plotly_white')
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        fig2_bar = px.bar(
                            x=dbs_labels,
                            y=dbs_values,
                            color=dbs_labels,
                            color_discrete_map={'Normal': '#1976D2', 'Anomaly': '#D32F2F'},
                            title='DBSCAN Counts',
                            labels={'x': 'Status', 'y': 'Count'},
                            text=dbs_values
                        )
                        fig2_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
                        fig2_bar.update_layout(height=300, template='plotly_white')
                        st.plotly_chart(fig2_bar, use_container_width=True)
                    
                    st.subheader("Model Comparison")
                    comparison_data = pd.DataFrame({
                        'Model': ['Isolation Forest', 'DBSCAN'],
                        'Anomalies': [if_counts.get(-1, 0), dbs_values[1]],
                        'Normal': [if_counts.get(1, 0), dbs_values[0]]
                    })
                    
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Bar(name='Anomalies', x=comparison_data['Model'], y=comparison_data['Anomalies'], marker_color='#C62828'))
                    fig_comp.add_trace(go.Bar(name='Normal', x=comparison_data['Model'], y=comparison_data['Normal'], marker_color='#2E7D32'))
                    fig_comp.update_layout(barmode='group', title='Anomaly Detection Model Comparison', height=400, template='plotly_white', yaxis_title='Count')
                    st.plotly_chart(fig_comp, use_container_width=True)

            with tab3:
                st.subheader("Performance Metrics Analysis")
                if st.session_state.results:
                    results_df = st.session_state.results['results_df']
                    
                    st.markdown("#### Model Performance Comparison")
                    
                    exec_times = []
                    models = []
                    for _, row in results_df.iterrows():
                        try:
                            time_str = str(row['Execution Time (Speed)'])
                            if time_str != 'N/A' and time_str != 'nan':
                                time_val = float(time_str.replace('s', '').strip())
                                exec_times.append(time_val)
                                models.append(row['Module/Model'])
                        except (ValueError, AttributeError, TypeError):
                            continue
                    
                    if exec_times:
                        fig_time = px.bar(
                            x=models,
                            y=exec_times,
                            color=exec_times,
                            color_continuous_scale='Viridis',
                            title='Execution Time Comparison',
                            labels={'x': 'Model', 'y': 'Time (seconds)'},
                            text=[f'{t:.2f}s' for t in exec_times]
                        )
                        fig_time.update_traces(textposition='outside')
                        fig_time.update_layout(height=400, template='plotly_white')
                        st.plotly_chart(fig_time, use_container_width=True)
                    
                    metrics_data = []
                    for _, row in results_df.iterrows():
                        try:
                            precision = row.get('Precision', 'N/A')
                            recall = row.get('Recall', 'N/A')
                            
                            if str(precision) != 'N/A' and str(recall) != 'N/A':
                                f1_val = row.get('F1-Score', 'N/A')
                                edr = row.get('EDR', 'N/A')
                                
                                metrics_data.append({
                                    'Model': row.get('Module/Model', 'Unknown'),
                                    'Precision': float(precision) if str(precision) != 'N/A' else 0.0,
                                    'Recall': float(recall) if str(recall) != 'N/A' else 0.0,
                                    'F1-Score': float(f1_val) if str(f1_val) != 'N/A' else 0.0,
                                    'EDR': float(edr) if str(edr) != 'N/A' else 0.0
                                })
                        except (ValueError, TypeError, AttributeError):
                            continue
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        fig_metrics = go.Figure()
                        
                        for model in metrics_df['Model']:
                            model_data = metrics_df[metrics_df['Model'] == model].iloc[0]
                            fig_metrics.add_trace(go.Scatter(
                                x=['Precision', 'Recall', 'F1-Score', 'EDR'],
                                y=[model_data['Precision'], model_data['Recall'], 
                                   model_data['F1-Score'], model_data['EDR']],
                                mode='lines+markers',
                                name=model,
                                fill='toself'
                            ))
                        
                        fig_metrics.update_layout(
                            title='Performance Metrics Comparison',
                            height=500,
                            template='plotly_white',
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig_metrics, use_container_width=True)

            with tab4:
                st.subheader("Temporal Analysis")
                if processed_df is not None and 'CRASH DATE' in processed_df.columns:
                    df_plot = processed_df.copy()
                    df_plot['CRASH DATE'] = pd.to_datetime(df_plot['CRASH DATE'], errors='coerce')
                    df_plot = df_plot.dropna(subset=['CRASH DATE'])
                    
                    df_plot['Year'] = df_plot['CRASH DATE'].dt.year
                    df_plot['Month'] = df_plot['CRASH DATE'].dt.month
                    df_plot['YearMonth'] = df_plot['CRASH DATE'].dt.to_period('M').astype(str)
                    
                    time_series = df_plot.groupby('YearMonth').size().reset_index(name='Count')
                    time_series['YearMonth'] = pd.to_datetime(time_series['YearMonth'])
                    
                    fig_time = px.line(
                        time_series,
                        x='YearMonth',
                        y='Count',
                        title='Collisions Over Time',
                        labels={'YearMonth': 'Date', 'Count': 'Number of Collisions'},
                        markers=True
                    )
                    fig_time.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    df_plot['IsAnomaly'] = (df_plot['ANOMALY_FLAG_IF'] == -1).astype(int)
                    anomaly_time = df_plot.groupby('YearMonth').agg({
                        'IsAnomaly': 'sum',
                        'COLLISION_ID': 'count'
                    }).reset_index()
                    anomaly_time['YearMonth'] = pd.to_datetime(anomaly_time['YearMonth'])
                    anomaly_time['AnomalyRate'] = (anomaly_time['IsAnomaly'] / anomaly_time['COLLISION_ID'] * 100)
                    
                    fig_anomaly = px.line(
                        anomaly_time,
                        x='YearMonth',
                        y='AnomalyRate',
                        title='Anomaly Rate Over Time (%)',
                        labels={'YearMonth': 'Date', 'AnomalyRate': 'Anomaly Rate (%)'},
                        markers=True
                    )
                    fig_anomaly.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    monthly = df_plot.groupby('Month').size().reset_index(name='Count')
                    fig_monthly = px.bar(
                        monthly,
                        x='Month',
                        y='Count',
                        title='Collisions by Month',
                        labels={'Month': 'Month', 'Count': 'Number of Collisions'},
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    fig_monthly.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig_monthly, use_container_width=True)

            with tab5:
                st.subheader("Data Quality Summary")
                if processed_df is not None:
                    df_plot = processed_df.copy()
                    
                    col_sum1, col_sum2 = st.columns(2)
                    
                    with col_sum1:
                        st.markdown("#### Dataset Statistics")
                        summary_data = {
                            'Metric': [
                                'Total Records',
                                'Records with Valid Coordinates',
                                'Records with Missing Coordinates',
                                'Total Injuries',
                                'Total Fatalities',
                                'Isolation Forest Anomalies',
                                'DBSCAN Anomalies (Sampled)'
                            ],
                            'Value': [
                                len(df_plot),
                                df_plot[['LATITUDE', 'LONGITUDE']].notna().all(axis=1).sum(),
                                df_plot[['LATITUDE', 'LONGITUDE']].isna().any(axis=1).sum(),
                                df_plot['NUMBER OF PERSONS INJURED'].sum(),
                                df_plot['NUMBER OF PERSONS KILLED'].sum(),
                                (df_plot['ANOMALY_FLAG_IF'] == -1).sum(),
                                (df_plot['ANOMALY_FLAG_DBSCAN'] == -1).sum()
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    with col_sum2:
                        st.markdown("#### Quality Indicators")
                        total = len(df_plot)
                        valid_coords = df_plot[['LATITUDE', 'LONGITUDE']].notna().all(axis=1).sum()
                        anomalies_if = (df_plot['ANOMALY_FLAG_IF'] == -1).sum()
                        
                        quality_metrics = {
                            'Metric': [
                                'Data Completeness',
                                'Coordinate Validity',
                                'Anomaly Rate (IF)',
                                'Data Quality Score'
                            ],
                            'Value': [
                                f"{(valid_coords/total*100):.2f}%",
                                f"{(valid_coords/total*100):.2f}%",
                                f"{(anomalies_if/total*100):.2f}%",
                                f"{max(0, 100 - (anomalies_if/total*100)):.2f}%"
                            ]
                        }
                        quality_df = pd.DataFrame(quality_metrics)
                        st.dataframe(quality_df, use_container_width=True, hide_index=True)
                        
                        fig_quality = go.Figure()
                        fig_quality.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=valid_coords/total*100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Data Completeness"},
                            gauge={'axis': {'range': [None, 100]},
                                   'bar': {'color': "darkblue"},
                                   'steps': [
                                       {'range': [0, 50], 'color': "lightgray"},
                                       {'range': [50, 100], 'color': "gray"}],
                                   'threshold': {'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75, 'value': 90}})
                        )
                        fig_quality.update_layout(height=300)
                        st.plotly_chart(fig_quality, use_container_width=True)

            st.markdown("---")
            st.markdown("### Data Exploration & Export")
            
            preview_tab1, preview_tab2, preview_tab3 = st.tabs([
                "Raw Data Preview",
                "Anomaly Records",
                "Export Data"
            ])
            
            with preview_tab1:
                st.markdown("#### Raw Dataset Preview")
                num_rows = st.slider("Number of rows to display", 10, 1000, 100, 10)
                
                search_col = st.selectbox("Search in column", ['All'] + list(df.columns))
                search_term = st.text_input("Search term", "")
                
                display_df = df.copy()
                if search_term and search_col != 'All':
                    if search_col in display_df.columns:
                        display_df = display_df[display_df[search_col].astype(str).str.contains(search_term, case=False, na=False)]
                elif search_term:
                    mask = pd.Series([False] * len(display_df))
                    for col in display_df.columns:
                        mask |= display_df[col].astype(str).str.contains(search_term, case=False, na=False)
                    display_df = display_df[mask]
                
                st.dataframe(display_df.head(num_rows), use_container_width=True, height=400)
                st.caption(f"Showing {min(num_rows, len(display_df)):,} of {len(display_df):,} records")
            
            with preview_tab2:
                st.markdown("#### Anomaly Records Analysis")
                if processed_df is not None:
                    anomaly_df = processed_df[processed_df['ANOMALY_FLAG_IF'] == -1].copy()
                    
                    if len(anomaly_df) > 0:
                        st.info(f"**Total Anomalies Found:** {len(anomaly_df):,}")
                        
                        col_anom1, col_anom2, col_anom3 = st.columns(3)
                        with col_anom1:
                            st.metric("Avg Injuries", f"{anomaly_df['NUMBER OF PERSONS INJURED'].mean():.2f}")
                        with col_anom2:
                            st.metric("Max Injuries", f"{anomaly_df['NUMBER OF PERSONS INJURED'].max()}")
                        with col_anom3:
                            st.metric("Records with Missing Coords", 
                                     f"{anomaly_df[['LATITUDE', 'LONGITUDE']].isna().any(axis=1).sum():,}")
                        
                        num_anom_rows = st.slider("Rows to display", 10, 500, 100, 10, key="anom_rows")
                        st.dataframe(anomaly_df.head(num_anom_rows), use_container_width=True, height=400)
                        st.caption(f"Showing {min(num_anom_rows, len(anomaly_df)):,} of {len(anomaly_df):,} anomaly records")
                    else:
                        st.success("No anomalies detected!")
            
            with preview_tab3:
                st.markdown("#### Export Processed Data")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    st.markdown("**Export Options:**")
                    export_raw = st.checkbox("Include Raw Data", value=False)
                    export_anomalies = st.checkbox("Export Anomalies Only", value=False)
                    export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
                
                with export_col2:
                    st.markdown("**Export Statistics:**")
                    if processed_df is not None:
                        st.info(f"Total Records: {len(processed_df):,}")
                        st.info(f"Anomalies: {(processed_df['ANOMALY_FLAG_IF'] == -1).sum():,}")
                
                if st.button("Generate Export", type="primary"):
                    export_df = processed_df.copy() if processed_df is not None else df.copy()
                    
                    if export_anomalies:
                        export_df = export_df[export_df['ANOMALY_FLAG_IF'] == -1]
                    
                    if export_format == "CSV":
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"collisions_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        json_str = export_df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"collisions_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    elif export_format == "Excel":
                        try:
                            from io import BytesIO
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                export_df.to_excel(writer, index=False, sheet_name='Data')
                            output.seek(0)
                            st.download_button(
                                label="Download Excel",
                                data=output,
                                file_name=f"collisions_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except ImportError:
                            st.error("Excel export requires openpyxl. Install with: pip install openpyxl")
    else:
        st.info("Please wait for data to load or place dataset.csv in the current directory.")
        
        st.markdown("""
        ### About This Dashboard
        
        This dashboard provides a comprehensive data quality framework for Motor Vehicle Collisions datasets.
        
        **Features:**
        - Schema validation using Pandera
        - Rule-based validation (geospatial bounds, logic checks)
        - Anomaly detection (Isolation Forest & DBSCAN)
        - Performance metrics (Precision, Recall, F1-Score, EDR)
        - Interactive visualizations
        
        **How to Use:**
        - The dashboard automatically loads `dataset.csv` from the current directory
        - Data is processed automatically on first load
        - Use the "Reload Data" button in the sidebar if you need to refresh
        - Explore the results in the dashboard sections below
        """)


if __name__ == "__main__":
    main()
