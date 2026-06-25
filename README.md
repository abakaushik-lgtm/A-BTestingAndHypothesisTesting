# A/B Testing & Statistical Experimentation Platform

An advanced, corporate-grade statistical experimentation platform designed to clean user interaction logs, analyze conversion rates, run hypothesis testing (Z-test, Chi-Square, Welch's T-Test), estimate financial lifts, profile users via K-Means clustering, and automatically compile executive-level PDF reports.

This platform simulates the experimentation systems utilized by companies like Google, Amazon, and Netflix to evaluate design options, campaign variations, and layout changes.

---

## 🌟 Key Features

### 1. Preprocessing & Integrity Pipeline (`data_cleaner.py`)
Automatically scans for data quality anomalies to protect experimental validity:
* **Duplicate Rows / Clicks**: Eliminates duplicate event logs.
* **Split Assignment Violations (User Leakage)**: Purges records of users assigned to both Variant A and B.
* **Corrupt Identifiers / Empty Fields**: Flags and removes null `User ID`, `Session ID`, or `Variant`.
* **Out-of-Bound Statistics**: Excludes records with negative revenue, negative session duration, or page views.
* **Standardization**: Dynamically standardizes variant string formats (e.g. `control`, `Variant_B` -> standardized `A` and `B`).

### 2. Statistical Experimentation Engine (`stats_engine.py`)
Computes mathematical indicators to determine significance:
* **Two-Proportion Z-Test**: Calculates conversions, relative and absolute conversion lifts, Z-Score, two-tailed P-Value, and 95% Confidence Intervals for conversion rates.
* **Chi-Square Test of Independence**: Validates the frequency table of conversions vs. non-conversions.
* **Welch's Two-Sample Independent T-Test**: Analyzes continuous metrics (Average Order Value, Revenue per User, Time on Site, Page Views) without assuming equal variances.
* **Segmented Analysis**: Runs Z-tests on customer groups (Device Type, Country, Customer Type) to locate localized impacts.

### 3. Machine Learning Enhancements (`ml_module.py`)
Utilizes ML for advanced behavioral profiling:
* **K-Means User Segmentation**: Clusters users into groups (e.g., *High-Value Buyers*, *Engaged Browsers*, *Casual Visitors*) based on Page Views, Time on Site, and Revenue.
* **Conversion Drivers (Random Forest)**: Trains a classification model to output the top feature importances driving conversions.

### 4. Revenue Impact Projections (`recommendation_engine.py`)
* Translates percentage conversion lifts into monthly and annual revenue gains based on average order sizes and traffic parameters.
* Identifies trade-offs (e.g., alert when conversion rate increases but Average Order Value decreases).

### 5. Automated PDF Report Generator (`report_generator.py`)
* Compiles an executive summary report utilizing ReportLab.
* Contains metadata, data cleaning details, statistical significance tables, financial projections, growth action plans, and authorization sign-offs.

---

## 🚀 Installation & Running

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Install all required libraries using:
```bash
pip install streamlit pandas numpy scipy statsmodels plotly scikit-learn reportlab openpyxl
```

### 3. Run the App
Start the Streamlit dashboard by executing:
```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser to interact with the platform.

---

## 📁 Repository Structure

* `app.py`: Streamlit frontend dashboard coordinating views, inputs, and charts.
* `data_generator.py`: Generates customizable synthetic experimental datasets.
* `data_cleaner.py`: Business logic for data diagnostics and preprocessing.
* `stats_engine.py`: Mathematical module running Z-tests, Chi-square tests, T-tests, and segmentation splits.
* `recommendation_engine.py`: Prepares qualitative corporate decisions and projects financial returns.
* `ml_module.py`: Performs customer K-Means clustering and extracts Random Forest importances.
* `report_generator.py`: PDF compiler layout.
* `test_app.py`: Python unittest suite validating core modules.

---

## 🧪 Running Unit Tests

Run the automated test suite to verify code math and cleaning pipelines:
```bash
python -m unittest test_app.py
```
