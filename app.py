import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from io import BytesIO

# Import custom modules
from data_generator import generate_experiment_data
from data_cleaner import inspect_data_quality, clean_experiment_data
from stats_engine import run_z_test, run_chi_square_test, run_t_test, analyze_segments
from recommendation_engine import calculate_revenue_impact, generate_recommendations
from ml_module import perform_user_segmentation, get_conversion_drivers
from report_generator import generate_pdf_report

# Page config
st.set_page_config(
    page_title="A/B Testing & Statistical Experimentation Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Executive Look
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main dashboard container adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    
    /* Custom headers */
    h1 {
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    h2 {
        color: #0D9488;
        font-weight: 600;
    }
    
    /* Styled Metric Card Container */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.85rem;
        color: #111827;
        font-weight: 700;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .delta-positive {
        color: #10B981;
    }
    .delta-negative {
        color: #EF4444;
    }
    .delta-neutral {
        color: #6B7280;
    }
    
    /* Significance Badges */
    .sig-badge-positive {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        border: 1.5px solid #10B981;
        display: inline-block;
        font-size: 1rem;
    }
    .sig-badge-negative {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        border: 1.5px solid #EF4444;
        display: inline-block;
        font-size: 1rem;
    }
    .sig-badge-neutral {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        border: 1.5px solid #F59E0B;
        display: inline-block;
        font-size: 1rem;
    }
    
    /* Clean sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Clean tab layout */
    button[data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75rem 1rem;
        color: #475569;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1E3A8A !important;
        border-bottom-color: #1E3A8A !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "cleaning_summary" not in st.session_state:
    st.session_state["cleaning_summary"] = None
if "data_cleaned" not in st.session_state:
    st.session_state["data_cleaned"] = False
if "expected_traffic" not in st.session_state:
    st.session_state["expected_traffic"] = 1000000
if "aov_input" not in st.session_state:
    st.session_state["aov_input"] = 75.0

# Define Title / Header Area
st.title("🧪 A/B Testing & Statistical Experimentation Platform")
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.image("https://img.icons8.com/color/96/flask.png", width=64)
st.sidebar.header("Control Panel")

# Section 1: Data Acquisition
data_source = st.sidebar.radio("Data Source", ["Use Synthetic Data Generator", "Upload Dataset File"])

if data_source == "Use Synthetic Data Generator":
    st.sidebar.markdown("### Generator Settings")
    n_users = st.sidebar.slider("Number of Users", 5000, 200000, 50000, step=5000)
    cr_a = st.sidebar.slider("Conversion Rate A (Control)", 0.01, 0.20, 0.08, step=0.005, format="%.3f")
    
    # Calculate slider range for B based on A
    lift_input = st.sidebar.slider("Relative Lift B vs A (%)", -30.0, 50.0, 15.0, step=1.0)
    cr_b = cr_a * (1.0 + lift_input / 100.0)
    
    aov_a = st.sidebar.number_input("Average Revenue per Order A ($)", value=70.0, min_value=1.0)
    aov_b = aov_a * (1.0 + (lift_input / 2.0) / 100.0)  # Slight correlation
    
    inject_issues = st.sidebar.checkbox("Inject Data Quality Anomalies", value=True, help="Inject duplicates, user leakage, invalid revenue, and dates for testing cleaner module.")
    
    if st.sidebar.button("Generate & Load Dataset", type="primary"):
        with st.spinner("Generating synthetic experimentation logs..."):
            df = generate_experiment_data(
                num_users=n_users,
                conversion_rate_a=cr_a,
                conversion_rate_b=cr_b,
                revenue_mean_a=aov_a,
                revenue_mean_b=aov_b,
                inject_anomalies=inject_issues,
                random_seed=42
            )
            st.session_state["raw_df"] = df
            st.session_state["cleaned_df"] = None
            st.session_state["cleaning_summary"] = None
            st.session_state["data_cleaned"] = False
            st.sidebar.success("Dataset generated successfully!")

else: # Upload Custom File
    st.sidebar.markdown("### Upload Files")
    uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, or JSON File", type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        try:
            if file_ext == ".csv":
                df = pd.read_csv(uploaded_file)
            elif file_ext == ".xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_ext == ".json":
                df = pd.read_json(uploaded_file)
            
            st.session_state["raw_df"] = df
            st.session_state["cleaned_df"] = None
            st.session_state["cleaning_summary"] = None
            st.session_state["data_cleaned"] = False
            st.sidebar.success("Dataset uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

# Section 2: Experiment Settings
st.sidebar.markdown("### Statistical Settings")
alpha_val = st.sidebar.selectbox("Significance Level (Alpha)", [0.01, 0.05, 0.10], index=1)
target_power = st.sidebar.slider("Statistical Power (Beta)", 0.70, 0.95, 0.80, step=0.05)

# Section 3: Revenue Impact Constants
st.sidebar.markdown("### Business Assumptions")
st.session_state["expected_traffic"] = st.sidebar.number_input("Expected Monthly Traffic", min_value=1000, value=1000000, step=50000)
st.session_state["aov_input"] = st.sidebar.number_input("Assumed Average Order Value ($)", min_value=1.0, value=75.0, step=5.0)

# Check if data exists
if st.session_state["raw_df"] is None:
    # Set default synthetic dataset at start so app is populated
    df = generate_experiment_data(
        num_users=30000,
        conversion_rate_a=0.08,
        conversion_rate_b=0.095,
        revenue_mean_a=70.0,
        revenue_mean_b=75.0,
        inject_anomalies=True,
        random_seed=42
    )
    st.session_state["raw_df"] = df

# Quick variables
raw_df = st.session_state["raw_df"]
cleaned_df = st.session_state["cleaned_df"] if st.session_state["data_cleaned"] else raw_df
is_cleaned = st.session_state["data_cleaned"]

# ================= MAIN AREA TABS =================
tab_overview, tab_cleaning, tab_analytics, tab_testing, tab_impact, tab_report = st.tabs([
    "📂 Dataset Overview", 
    "🧹 Data Cleaning", 
    "📊 Experiment Analytics", 
    "🧪 Hypothesis Testing", 
    "💰 Revenue Impact", 
    "📄 Executive Report & ML"
])

# ----------------- TAB 1: DATASET OVERVIEW -----------------
with tab_overview:
    st.header("📂 Raw Dataset Overview")
    st.markdown("Review the raw imported session-level log before processing validation and testing.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Raw Log Rows</div>
            <div class="metric-value">{len(raw_df):,}</div>
            <div class="metric-delta delta-neutral">Session clicks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        n_unique_users = raw_df["User ID"].nunique() if "User ID" in raw_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Unique Users</div>
            <div class="metric-value">{n_unique_users:,}</div>
            <div class="metric-delta delta-neutral">Experimental Subjects</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        n_variants = raw_df["Variant"].nunique() if "Variant" in raw_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Variants Identified</div>
            <div class="metric-value">{n_variants}</div>
            <div class="metric-delta delta-neutral">Groups: {', '.join(raw_df['Variant'].dropna().unique().astype(str)) if 'Variant' in raw_df.columns else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        total_rev = raw_df["Revenue"].sum() if "Revenue" in raw_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Raw Revenue Recorded</div>
            <div class="metric-value">${total_rev:,.2f}</div>
            <div class="metric-delta delta-neutral">Sum of purchases</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Raw Data Sample")
    st.dataframe(raw_df.head(100), use_container_width=True)
    
    col_desc, col_types = st.columns([2, 1])
    with col_desc:
        st.subheader("Descriptive Statistics")
        st.dataframe(raw_df.describe(include=[np.number]), use_container_width=True)
    with col_types:
        st.subheader("Column Schemas")
        schema_df = pd.DataFrame({
            "DataType": raw_df.dtypes.astype(str),
            "Null Values": raw_df.isna().sum(),
            "Null %": (raw_df.isna().sum() / len(raw_df) * 100).round(2)
        })
        st.dataframe(schema_df, use_container_width=True)

# ----------------- TAB 2: DATA CLEANING -----------------
with tab_cleaning:
    st.header("🧹 Preprocessing & Quality Validation")
    st.markdown("Validate tracking integrity, resolve split assignment leakage, and clean anomalous records.")
    
    # Analyze raw logs quality
    quality_report = inspect_data_quality(raw_df)
    
    col_report_left, col_report_right = st.columns(2)
    
    with col_report_left:
        st.subheader("Anomalies & Integrity Diagnostics")
        
        diag_data = [
            {"Issue Detected": "Duplicate click records (exact row duplicates)", "Found": quality_report["duplicate_records"], "Action": "Drop duplicate records"},
            {"Issue Detected": "Missing User IDs / Session IDs", "Found": quality_report["missing_user_ids"], "Action": "Drop invalid records"},
            {"Issue Detected": "Missing Variant group labels", "Found": quality_report["missing_variants"], "Action": "Drop invalid records"},
            {"Issue Detected": "Corrupted Variant labels (not 'A' or 'B')", "Found": quality_report["corrupted_variants"], "Action": "Standardize to A or B"},
            {"Issue Detected": "User Variant Leakage (assigned to BOTH variants)", "Found": quality_report["leaked_users"], "Action": "Purge all records of leaked users"},
            {"Issue Detected": "Invalid Revenue metrics (< 0)", "Found": quality_report["negative_revenue"], "Action": "Drop records"},
            {"Issue Detected": "Invalid timestamps (not parseable)", "Found": quality_report["invalid_timestamps"], "Action": "Drop records"},
        ]
        diag_df = pd.DataFrame(diag_data)
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        
        btn_clean = st.button("Run Preprocessing Pipeline", type="primary", use_container_width=True)
        if btn_clean:
            cleaned, summary = clean_experiment_data(raw_df)
            st.session_state["cleaned_df"] = cleaned
            st.session_state["cleaning_summary"] = summary
            st.session_state["data_cleaned"] = True
            st.rerun()
            
    with col_report_right:
        st.subheader("Cleaning Diagnostics Output")
        
        if is_cleaned:
            summary = st.session_state["cleaning_summary"]
            st.success("Cleaning Pipeline completed successfully! Cleaned dataset loaded.")
            
            # Show summary stats
            clean_metrics = [
                ("Duplicates Removed", f"{summary['duplicates_removed']:,}"),
                ("Invalid/Null Users Removed", f"{summary['missing_users_removed']:,}"),
                ("Corrupt Variants Standardized", f"{summary['corrupted_variants_cleaned']:,}"),
                ("Leaked Users Eliminated", f"{summary['leaked_users_count']:,} users ({summary['leaked_records_removed']:,} records)"),
                ("Negative Revenue Removed", f"{summary['negative_revenue_records_removed']:,}"),
                ("Final Cleaned Records", f"{summary['final_records_count']:,} (out of {summary['initial_records_count']:,})"),
                ("Data Size Reduction %", f"{(summary['total_records_removed'] / summary['initial_records_count'] * 100):.2f}%")
            ]
            
            summary_df = pd.DataFrame(clean_metrics, columns=["Process Step", "Records / Percentage"])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("Status: Waiting for processing pipeline execution. Click the button to the left to run cleaning routines.")

# ----------------- TAB 3: EXPERIMENT ANALYTICS -----------------
with tab_analytics:
    st.header("📊 Interactive Behavioral Analytics")
    
    if is_cleaned:
        st.info("Using Cleaned Experiment Dataset")
    else:
        st.warning("⚠️ Showing raw uncleaned dataset. Run the preprocessing tab for clean experimental integrity.")
        
    # Standardize columns checks
    has_variant = "Variant" in cleaned_df.columns
    has_purchase = "Purchase Completed" in cleaned_df.columns
    has_revenue = "Revenue" in cleaned_df.columns
    has_device = "Device" in cleaned_df.columns
    has_country = "Country" in cleaned_df.columns
    has_timestamp = "Timestamp" in cleaned_df.columns
    
    if not (has_variant and has_purchase):
        st.error("Dataset must contain 'Variant' and 'Purchase Completed' columns for analytics.")
    else:
        # Precompute user-level aggregations
        # Aggregation of purchase completed to user level: 1 if converted at least once
        user_level = cleaned_df.groupby("User ID").agg({
            "Variant": "first",
            "Device": "first",
            "Country": "first",
            "Customer Type": "first",
            "Page Views": "sum",
            "Time on Site": "sum",
            "Purchase Completed": "max",
            "Revenue": "sum"
        }).reset_index()
        
        users_a = user_level[user_level["Variant"] == "A"]
        users_b = user_level[user_level["Variant"] == "B"]
        
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Control (A) Users</div>
                <div class="metric-value">{len(users_a):,}</div>
                <div class="metric-delta delta-neutral">Share: {len(users_a)/len(user_level)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Treatment (B) Users</div>
                <div class="metric-value">{len(users_b):,}</div>
                <div class="metric-delta delta-neutral">Share: {len(users_b)/len(user_level)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            conv_a = users_a["Purchase Completed"].sum()
            conv_b = users_b["Purchase Completed"].sum()
            rate_a = conv_a / len(users_a) if len(users_a) > 0 else 0
            rate_b = conv_b / len(users_b) if len(users_b) > 0 else 0
            lift = (rate_b - rate_a) / rate_a if rate_a > 0 else 0
            delta_class = "delta-positive" if lift > 0 else ("delta-negative" if lift < 0 else "delta-neutral")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Conversion Rates</div>
                <div class="metric-value">A: {rate_a:.2%} | B: {rate_b:.2%}</div>
                <div class="metric-delta {delta_class}">Lift: {lift:+.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            rev_a = users_a["Revenue"].sum()
            rev_b = users_b["Revenue"].sum()
            aov_a = rev_a / conv_a if conv_a > 0 else 0
            aov_b = rev_b / conv_b if conv_b > 0 else 0
            aov_lift = (aov_b - aov_a) / aov_a if aov_a > 0 else 0
            delta_class_aov = "delta-positive" if aov_lift > 0 else ("delta-negative" if aov_lift < 0 else "delta-neutral")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Average Order Value</div>
                <div class="metric-value">A: ${aov_a:.2f} | B: ${aov_b:.2f}</div>
                <div class="metric-delta {delta_class_aov}">Lift: {aov_lift:+.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Visual charts
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.subheader("Conversion Funnel Comparison")
            # Construct Funnel Steps
            # Steps: Visitors -> Checkout Started -> Purchase Completed
            # Aggregate sessions-level counts
            sessions_a = cleaned_df[cleaned_df["Variant"] == "A"]
            sessions_b = cleaned_df[cleaned_df["Variant"] == "B"]
            
            funnel_data = pd.DataFrame([
                {"Variant": "A", "Step": "1. Session Visit", "Count": len(sessions_a)},
                {"Variant": "A", "Step": "2. Checkout Start", "Count": sessions_a["Checkout Started"].sum() if "Checkout Started" in sessions_a.columns else 0},
                {"Variant": "A", "Step": "3. Purchase Complete", "Count": sessions_a["Purchase Completed"].sum()},
                
                {"Variant": "B", "Step": "1. Session Visit", "Count": len(sessions_b)},
                {"Variant": "B", "Step": "2. Checkout Start", "Count": sessions_b["Checkout Started"].sum() if "Checkout Started" in sessions_b.columns else 0},
                {"Variant": "B", "Step": "3. Purchase Complete", "Count": sessions_b["Purchase Completed"].sum()}
            ])
            
            fig_funnel = px.funnel(
                funnel_data, 
                x="Count", 
                y="Step", 
                color="Variant", 
                title="Experiment Conversion Funnel Breakdown",
                color_discrete_map={"A": "#1E3A8A", "B": "#0D9488"}
            )
            fig_funnel.update_layout(height=400)
            st.plotly_chart(fig_funnel, use_container_width=True)
            
        with row1_col2:
            st.subheader("Conversion Trends Over Time")
            if has_timestamp:
                # Group by day and variant
                # Clone timestamp safely to date
                df_trend = cleaned_df.copy()
                df_trend["Date"] = pd.to_datetime(df_trend["Timestamp"], errors="coerce").dt.date
                daily_perf = df_trend.groupby(["Date", "Variant"]).agg({
                    "User ID": "nunique",
                    "Purchase Completed": "sum"
                }).reset_index()
                daily_perf["Conversion Rate"] = daily_perf["Purchase Completed"] / daily_perf["User ID"]
                
                fig_trend = px.line(
                    daily_perf, 
                    x="Date", 
                    y="Conversion Rate", 
                    color="Variant", 
                    line_shape="spline",
                    title="Daily Conversion Rate Performance Trends",
                    color_discrete_map={"A": "#1E3A8A", "B": "#0D9488"}
                )
                fig_trend.update_layout(yaxis_tickformat=".2%", height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.warning("Timestamp column not found in dataset; daily trends unavailable.")
                
        # Device and Country breakdown
        st.markdown("<br/>", unsafe_allow_html=True)
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.subheader("Segment Analysis: Conversion Rate by Device Type")
            if has_device:
                dev_stats = analyze_segments(cleaned_df, "Device", alpha=alpha_val)
                # Plotly grouped bar
                dev_plot_df = pd.melt(dev_stats, id_vars=["Segment Value"], value_vars=["Rate A", "Rate B"],
                                      var_name="Variant", value_name="Conversion Rate")
                dev_plot_df["Variant"] = dev_plot_df["Variant"].str.replace("Rate ", "")
                
                fig_device = px.bar(
                    dev_plot_df, 
                    x="Segment Value", 
                    y="Conversion Rate", 
                    color="Variant", 
                    barmode="group",
                    title="Conversion Rates across Devices",
                    color_discrete_map={"A": "#1E3A8A", "B": "#0D9488"}
                )
                fig_device.update_layout(yaxis_tickformat=".2%", height=350, xaxis_title="Device Type")
                st.plotly_chart(fig_device, use_container_width=True)
            else:
                st.info("Device column not found.")
                
        with row2_col2:
            st.subheader("Segment Analysis: Conversion Rate by Country")
            if has_country:
                country_stats = analyze_segments(cleaned_df, "Country", alpha=alpha_val)
                country_plot_df = pd.melt(country_stats, id_vars=["Segment Value"], value_vars=["Rate A", "Rate B"],
                                          var_name="Variant", value_name="Conversion Rate")
                country_plot_df["Variant"] = country_plot_df["Variant"].str.replace("Rate ", "")
                
                fig_country = px.bar(
                    country_plot_df, 
                    x="Segment Value", 
                    y="Conversion Rate", 
                    color="Variant", 
                    barmode="group",
                    title="Conversion Rates across Regions",
                    color_discrete_map={"A": "#1E3A8A", "B": "#0D9488"}
                )
                fig_country.update_layout(yaxis_tickformat=".2%", height=350, xaxis_title="Country")
                st.plotly_chart(fig_country, use_container_width=True)
            else:
                st.info("Country column not found.")

# ----------------- TAB 4: HYPOTHESIS TESTING -----------------
with tab_testing:
    st.header("🧪 Advanced Statistical Hypothesis Engine")
    st.markdown("Evaluate conversion lifts, significance indicators, and sample distributions.")
    
    if not (has_variant and has_purchase):
        st.error("Required fields not present.")
    else:
        # Precompute user aggregates
        user_agg = cleaned_df.groupby("User ID").agg({
            "Variant": "first",
            "Purchase Completed": "max"
        }).reset_index()
        
        n_a = user_agg[user_agg["Variant"] == "A"].shape[0]
        conv_a = user_agg[(user_agg["Variant"] == "A") & (user_agg["Purchase Completed"] == 1)].shape[0]
        
        n_b = user_agg[user_agg["Variant"] == "B"].shape[0]
        conv_b = user_agg[(user_agg["Variant"] == "B") & (user_agg["Purchase Completed"] == 1)].shape[0]
        
        # Primary Hypothesis definition
        st.markdown(f"""
        <div style="background-color: #F1F5F9; padding: 1.25rem; border-radius: 8px; border-left: 5px solid #1E3A8A; margin-bottom: 1.5rem;">
            <b>Null Hypothesis (H₀):</b> No difference exists between Variant A (Control) and Variant B (Treatment) conversion rates (<i>p<sub>A</sub> = p<sub>B</sub></i>).<br/>
            <b>Alternative Hypothesis (H₁):</b> A difference exists between Variant A and Variant B conversion rates (<i>p<sub>A</sub> &ne; p<sub>B</sub></i>).
        </div>
        """, unsafe_allow_html=True)
        
        # Run Z-Test & Chi-Square
        z_res = run_z_test(conv_a, n_a, conv_b, n_b, alpha=alpha_val)
        chi_res = run_chi_square_test(conv_a, n_a, conv_b, n_b)
        
        # Row 1: Large decision callouts
        t_col1, t_col2 = st.columns([1, 2])
        with t_col1:
            st.subheader("Statistical Validation Status")
            if z_res["significant"]:
                if z_res["rel_lift"] > 0:
                    st.markdown("""<div class="sig-badge-positive">✓ STATISTICALLY SIGNIFICANT (WINNER: B)</div>""", unsafe_allow_html=True)
                    st.markdown("<p style='margin-top: 10px; color:#10B981;'>Variant B outperformed Control. Proceed with deployment.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="sig-badge-negative">✓ STATISTICALLY SIGNIFICANT (LOSER: B)</div>""", unsafe_allow_html=True)
                    st.markdown("<p style='margin-top: 10px; color:#EF4444;'>Variant B performed worse than Control. Retain Variant A.</p>", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="sig-badge-neutral">✗ NOT STATISTICALLY SIGNIFICANT</div>""", unsafe_allow_html=True)
                st.markdown("<p style='margin-top: 10px; color:#F59E0B;'>No reliable performance difference detected. Maintain control or run experiment longer.</p>", unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)
            # P-value vs Alpha gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = z_res["p_value"],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "P-Value Outcome", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1E3A8A"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, alpha_val], 'color': '#D1FAE5'},
                        {'range': [alpha_val, 1.0], 'color': '#F1F5F9'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': alpha_val
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with t_col2:
            st.subheader("Proportion Z-Test Diagnostics")
            
            z_details = pd.DataFrame([
                {"Parameter": "Variant A (Control) Conversion Rate", "Value": f"{z_res['rate_a']:.4%}"},
                {"Parameter": "Variant B (Treatment) Conversion Rate", "Value": f"{z_res['rate_b']:.4%}"},
                {"Parameter": "Absolute Conversion Rate Lift", "Value": f"{z_res['abs_lift']:+.4%}"},
                {"Parameter": "Relative Lifts (B vs A %)", "Value": f"{z_res['rel_lift']:+.2%}"},
                {"Parameter": "Two-Proportion Z Statistic", "Value": f"{z_res['z_stat']:.4f}"},
                {"Parameter": "Two-Tailed P-Value", "Value": f"{z_res['p_value']:.6f}"},
                {"Parameter": f"{int((1-alpha_val)*100)}% Confidence Interval of Difference", "Value": f"[{z_res['ci_lower']:.4%}, {z_res['ci_upper']:.4%}]"},
            ])
            st.dataframe(z_details, use_container_width=True, hide_index=True)
            
            # Confidence Interval Chart
            ci_fig = go.Figure()
            ci_fig.add_trace(go.Scatter(
                x=[z_res["abs_lift"]],
                y=["Conversion Rate Diff"],
                mode='markers+text',
                marker=dict(size=12, color='#0D9488'),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[z_res["ci_upper"] - z_res["abs_lift"]],
                    arrayminus=[z_res["abs_lift"] - z_res["ci_lower"]],
                    color='#1E3A8A',
                    thickness=3,
                    width=10
                ),
                text=[f"Lift: {z_res['abs_lift']:+.2%}"],
                textposition="top center"
            ))
            ci_fig.add_shape(
                type="line", x0=0, y0=-0.5, x1=0, y1=0.5,
                line=dict(color="red", width=1.5, dash="dash")
            )
            ci_fig.update_layout(
                title=f"95% Confidence Interval for Absolute Conversion Lift",
                xaxis=dict(title="Absolute Conversion Rate Lift", tickformat=".2%"),
                yaxis=dict(showticklabels=False),
                height=180,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(ci_fig, use_container_width=True)
            
        # Contingency & Chi-Square
        st.subheader("Categorical Chi-Square Verification")
        st.markdown("Chi-Square verification compares categorical frequencies (Purchased vs Non-Purchased) of Variant A and Variant B.")
        
        non_conv_a = n_a - conv_a
        non_conv_b = n_b - conv_b
        
        chi_data = [
            {"Purchase Status": "Completed (Yes)", "Variant A (Control)": f"{conv_a:,}", "Variant B (Treatment)": f"{conv_b:,}"},
            {"Purchase Status": "Abandoned (No)", "Variant A (Control)": f"{non_conv_a:,}", "Variant B (Treatment)": f"{non_conv_b:,}"},
        ]
        st.dataframe(pd.DataFrame(chi_data), use_container_width=True, hide_index=True)
        
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Chi-Square Statistic", f"{chi_res['chi2_stat']:.4f}")
        cc2.metric("Chi-Square P-Value", f"{chi_res['p_value']:.6f}")
        cc3.metric("Chi-Square Decision", "Significant (H₁)" if chi_res["significant"] else "Not Significant (H₀)")
        
        # Secondary continuous metrics (T-test)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.subheader("Continuous Metrics Evaluation (Welch's T-Test)")
        st.markdown("Evaluate continuous behavioral features. Select a metric below to perform Welch's T-test analysis:")
        
        available_t_metrics = []
        if has_revenue:
            available_t_metrics.append("Revenue")
        if "Time on Site" in cleaned_df.columns:
            available_t_metrics.append("Time on Site")
        if "Page Views" in cleaned_df.columns:
            available_t_metrics.append("Page Views")
            
        selected_t_metric = st.selectbox("Select Continuous Metric for T-Test", available_t_metrics)
        
        if selected_t_metric:
            # Group distributions
            if selected_t_metric == "Revenue":
                # For AOV we look only at purchasing users (revenue > 0)
                # For ARPU we look at all users (revenue >= 0)
                t_sub = st.radio("Measurement Scope", ["Average Order Value (AOV) - Purchasing Sessions Only", "Average Revenue Per User (ARPU) - All Users"], horizontal=True)
                if "AOV" in t_sub:
                    val_a = cleaned_df[(cleaned_df["Variant"] == "A") & (cleaned_df["Revenue"] > 0)]["Revenue"].values
                    val_b = cleaned_df[(cleaned_df["Variant"] == "B") & (cleaned_df["Revenue"] > 0)]["Revenue"].values
                else:
                    val_a = user_level[user_level["Variant"] == "A"]["Revenue"].values
                    val_b = user_level[user_level["Variant"] == "B"]["Revenue"].values
            elif selected_t_metric == "Time on Site":
                val_a = cleaned_df[cleaned_df["Variant"] == "A"]["Time on Site"].values
                val_b = cleaned_df[cleaned_df["Variant"] == "B"]["Time on Site"].values
            else: # Page Views
                val_a = cleaned_df[cleaned_df["Variant"] == "A"]["Page Views"].values
                val_b = cleaned_df[cleaned_df["Variant"] == "B"]["Page Views"].values
                
            t_res = run_t_test(val_a, val_b, alpha=alpha_val)
            
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown(f"**Selected Metric: {selected_t_metric}**")
                t_details = pd.DataFrame([
                    {"Parameter": "Variant A (Control) Mean", "Value": f"{t_res['mean_a']:.2f}"},
                    {"Parameter": "Variant B (Treatment) Mean", "Value": f"{t_res['mean_b']:.2f}"},
                    {"Parameter": "Absolute Mean Lift", "Value": f"{t_res['abs_lift']:+.2f}"},
                    {"Parameter": "Relative Lift (%)", "Value": f"{t_res['rel_lift']:+.2%}"},
                    {"Parameter": "T-Statistic", "Value": f"{t_res['t_stat']:.4f}"},
                    {"Parameter": "T-Test P-Value", "Value": f"{t_res['p_value']:.6f}"},
                    {"Parameter": "Statistically Significant", "Value": str(t_res['significant'])},
                ])
                st.dataframe(t_details, use_container_width=True, hide_index=True)
                
            with t_col2:
                # Plot distributions
                dist_df = pd.DataFrame(
                    [(x, "A") for x in val_a] + [(x, "B") for x in val_b],
                    columns=["Value", "Variant"]
                )
                # Sample the plot if too many values to render fast
                if len(dist_df) > 10000:
                    dist_df = dist_df.sample(10000, random_state=42)
                    
                fig_dist = px.box(
                    dist_df, 
                    x="Variant", 
                    y="Value", 
                    color="Variant",
                    title=f"{selected_t_metric} Distribution Box Plot",
                    color_discrete_map={"A": "#1E3A8A", "B": "#0D9488"}
                )
                fig_dist.update_layout(height=300)
                st.plotly_chart(fig_dist, use_container_width=True)

# ----------------- TAB 5: REVENUE IMPACT -----------------
with tab_impact:
    st.header("💰 Business Outcome & Financial Projections")
    st.markdown("Quantify commercial yields and project monetary returns of experiment deployment.")
    
    # Calculate conversion metrics to supply the calculator
    if not (has_variant and has_purchase):
        st.error("Insufficient variables.")
    else:
        user_agg = cleaned_df.groupby("User ID").agg({
            "Variant": "first",
            "Purchase Completed": "max",
            "Revenue": "sum"
        }).reset_index()
        
        n_a = user_agg[user_agg["Variant"] == "A"].shape[0]
        conv_a = user_agg[(user_agg["Variant"] == "A") & (user_agg["Purchase Completed"] == 1)].shape[0]
        n_b = user_agg[user_agg["Variant"] == "B"].shape[0]
        conv_b = user_agg[(user_agg["Variant"] == "B") & (user_agg["Purchase Completed"] == 1)].shape[0]
        
        rate_a = conv_a / n_a if n_a > 0 else 0
        rate_b = conv_b / n_b if n_b > 0 else 0
        
        # Calculate calculated average order value from dataset
        avg_order_val = cleaned_df[cleaned_df["Revenue"] > 0]["Revenue"].mean() if "Revenue" in cleaned_df.columns else st.session_state["aov_input"]
        if pd.isna(avg_order_val):
            avg_order_val = st.session_state["aov_input"]
            
        calc_traffic = st.session_state["expected_traffic"]
        calc_aov = st.session_state["aov_input"]
        
        col_calc_left, col_calc_right = st.columns([1, 2])
        
        with col_calc_left:
            st.subheader("Financial Calculator")
            monthly_traffic_input = st.number_input("Expected Monthly Traffic (visitors)", min_value=1000, value=int(calc_traffic), step=50000)
            aov_input_val = st.number_input("Assumed Average Order Value ($)", min_value=1.0, value=float(calc_aov), step=5.0)
            
            # Interactive Lift Overrides
            st.markdown("### Lifts Overrides (Observed values defaulted)")
            c_rate_a_override = st.number_input("Variant A Conversion Rate (%)", min_value=0.0, max_value=100.0, value=float(rate_a*100), step=0.5) / 100
            c_rate_b_override = st.number_input("Variant B Conversion Rate (%)", min_value=0.0, max_value=100.0, value=float(rate_b*100), step=0.5) / 100
            
            rev_impact = calculate_revenue_impact(
                monthly_traffic_input,
                c_rate_a_override,
                c_rate_b_override,
                aov_input_val
            )
            
        with col_calc_right:
            st.subheader("Projected Incremental Returns")
            
            rc1, rc2 = st.columns(2)
            with rc1:
                monthly_gain = rev_impact["monthly_revenue_gain"]
                monthly_symbol = "+" if monthly_gain >= 0 else ""
                st.metric("Projected Monthly Revenue Gain", f"{monthly_symbol}${monthly_gain:,.2f}", 
                          delta=f"{rev_impact['monthly_conversions_gain']:+,.0f} conversions")
            with rc2:
                annual_gain = rev_impact["annual_revenue_gain"]
                annual_symbol = "+" if annual_gain >= 0 else ""
                st.metric("Projected Annual Revenue Gain", f"{annual_symbol}${annual_gain:,.2f}",
                          delta=f"{rev_impact['annual_conversions_gain']:+,.0f} conversions")
                
            st.markdown("---")
            st.subheader("Annual Growth Projections")
            
            # Create a monthly accumulation dataframe
            months = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6", "Month 7", "Month 8", "Month 9", "Month 10", "Month 11", "Month 12"]
            acc_rev = [monthly_gain * (i+1) for i in range(12)]
            acc_conv = [rev_impact["monthly_conversions_gain"] * (i+1) for i in range(12)]
            
            proj_df = pd.DataFrame({
                "Timeline": months,
                "Accumulated Revenue": acc_rev,
                "Accumulated Conversions": acc_conv
            })
            
            fig_proj = px.bar(
                proj_df, 
                x="Timeline", 
                y="Accumulated Revenue",
                title="Accumulated Revenue Lift Projection (12 Months)",
                labels={"Accumulated Revenue": "Revenue Lift ($)"},
                color_discrete_sequence=["#0D9488"]
            )
            fig_proj.update_layout(height=300)
            st.plotly_chart(fig_proj, use_container_width=True)

# ----------------- TAB 6: EXECUTIVE REPORT & ML -----------------
with tab_report:
    st.header("📄 Executive Documentation & Advanced ML Insights")
    st.markdown("Compile printable PDF reports, explore automated decisions, and utilize clustering segmentation.")
    
    if not (has_variant and has_purchase):
        st.error("Dataset lacks required columns.")
    else:
        # Precompute variables for ReportLab
        user_agg = cleaned_df.groupby("User ID").agg({
            "Variant": "first",
            "Purchase Completed": "max",
            "Revenue": "sum",
            "Time on Site": "mean",
            "Page Views": "mean"
        }).reset_index()
        
        n_a = user_agg[user_agg["Variant"] == "A"].shape[0]
        conv_a = user_agg[(user_agg["Variant"] == "A") & (user_agg["Purchase Completed"] == 1)].shape[0]
        n_b = user_agg[user_agg["Variant"] == "B"].shape[0]
        conv_b = user_agg[(user_agg["Variant"] == "B") & (user_agg["Purchase Completed"] == 1)].shape[0]
        
        z_res = run_z_test(conv_a, n_a, conv_b, n_b, alpha=alpha_val)
        
        # AOV continuous T-test
        aov_a_vals = cleaned_df[(cleaned_df["Variant"] == "A") & (cleaned_df["Revenue"] > 0)]["Revenue"].values
        aov_b_vals = cleaned_df[(cleaned_df["Variant"] == "B") & (cleaned_df["Revenue"] > 0)]["Revenue"].values
        t_res_aov = run_t_test(aov_a_vals, aov_b_vals, alpha=alpha_val)
        
        # Time on Site T-test
        dur_a_vals = cleaned_df[cleaned_df["Variant"] == "A"]["Time on Site"].values
        dur_b_vals = cleaned_df[cleaned_df["Variant"] == "B"]["Time on Site"].values
        t_res_dur = run_t_test(dur_a_vals, dur_b_vals, alpha=alpha_val)
        
        # Recommendation
        rev_impact = calculate_revenue_impact(
            st.session_state["expected_traffic"],
            z_res["rate_a"],
            z_res["rate_b"],
            st.session_state["aov_input"]
        )
        
        recs = generate_recommendations(
            z_res=z_res,
            t_result_aov=t_res_aov,
            t_result_duration=t_res_dur,
            revenue_impact=rev_impact,
            alpha=alpha_val
        )
        
        # Section A: Recommendation Engine
        st.subheader("Automated Recommendation Engine")
        
        rec_bg = "#E6F4EA" if recs["decision"] == "DEPLOY_B" else ("#FCE8E6" if recs["decision"] == "RETAIN_A" else "#FEF7E0")
        rec_border = "#10B981" if recs["decision"] == "DEPLOY_B" else ("#EF4444" if recs["decision"] == "RETAIN_A" else "#F59E0B")
        
        st.markdown(f"""
        <div style="background-color: {rec_bg}; border: 1.5px solid {rec_border}; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h3 style="margin-top:0; color:#1E293B;">DECISION: {recs['decision_text']}</h3>
            <p><b>Business Rationale:</b></p>
            <ul>
                {"".join([f"<li>{r}</li>" for r in recs['rationale']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if recs["risks"]:
            st.warning("⚠️ **Secondary Metric Risks & Warnings:**")
            for r in recs["risks"]:
                st.write(f"- {r}")
                
        st.info("💡 **Growth Action Plan & Next Steps:**")
        for s in recs["next_steps"]:
            st.write(f"- {s}")
            
        # PDF compilation download
        st.subheader("Export Executive Summary Report")
        st.markdown("Download a fully formatted executive report (PDF) summarizing parameters, preprocessing diagnostic charts, statistical evaluations, and final signatures.")
        
        # Build Report PDF in memory
        try:
            pdf_buffer = BytesIO()
            dataset_summary = {
                "users_a": n_a,
                "conversions_a": conv_a,
                "users_b": n_b,
                "conversions_b": conv_b
            }
            
            # Extract cleaning summary from state
            if is_cleaned:
                clean_summ = st.session_state["cleaning_summary"]
            else:
                # Default cleaning summary fallback if cleaner was skipped
                clean_summ = {
                    "initial_records_count": len(raw_df),
                    "final_records_count": len(raw_df),
                    "duplicates_removed": 0,
                    "missing_users_removed": 0,
                    "corrupted_variants_cleaned": 0,
                    "leaked_users_count": 0,
                    "leaked_records_removed": 0,
                    "negative_revenue_records_removed": 0,
                    "total_records_removed": 0,
                    "invalid_timestamps_removed": 0,
                    "invalid_page_views_removed": 0,
                    "invalid_time_on_site_removed": 0
                }
                
            generate_pdf_report(
                pdf_buffer,
                dataset_summary=dataset_summary,
                cleaning_summary=clean_summ,
                z_res=z_res,
                t_res_aov=t_res_aov,
                t_res_dur=t_res_dur,
                recs=recs,
                revenue_impact=rev_impact,
                expected_traffic=st.session_state["expected_traffic"],
                aov_input=st.session_state["aov_input"]
            )
            pdf_data = pdf_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Executive PDF Report",
                data=pdf_data,
                file_name=f"AB_Testing_Executive_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error compiling PDF: {e}")
            
        # Section B: Machine Learning Insights
        st.markdown("<br/><hr/><br/>", unsafe_allow_html=True)
        st.subheader("🤖 Machine Learning Enhancements")
        
        ml_tab1, ml_tab2 = st.tabs(["K-Means User Segmentation", "Conversion Driver Importances (Random Forest)"])
        
        with ml_tab1:
            st.markdown("Perform K-Means clustering on aggregated page views, session duration, and revenue to discover core buyer groups.")
            
            if len(cleaned_df) < 500:
                st.warning("Insufficient dataset size to perform reliable clustering. Load a larger dataset.")
            else:
                try:
                    user_seg_df, cl_summary = perform_user_segmentation(cleaned_df, n_clusters=3)
                    
                    sc1, sc2 = st.columns([2, 1])
                    with sc1:
                        # Cluster Scatter Plot (Page Views vs Revenue)
                        fig_clus = px.scatter(
                            user_seg_df.sample(min(len(user_seg_df), 5000), random_state=42),
                            x="Page Views",
                            y="Revenue",
                            color="Segment Name",
                            hover_data=["User ID", "Customer Type"],
                            title="User Segments Behavioral Profiling",
                            color_discrete_sequence=["#1E3A8A", "#0D9488", "#F59E0B"]
                        )
                        fig_clus.update_layout(height=350)
                        st.plotly_chart(fig_clus, use_container_width=True)
                        
                    with sc2:
                        st.markdown("**Segment Profiles Summary**")
                        # Format dataframe
                        cl_summary_fmt = cl_summary.copy()
                        cl_summary_fmt["Page Views"] = cl_summary_fmt["Page Views"].round(1)
                        cl_summary_fmt["Time on Site"] = cl_summary_fmt["Time on Site"].round(1)
                        cl_summary_fmt["Conversion Rate"] = (cl_summary_fmt["Conversion Rate"] * 100).round(2).astype(str) + "%"
                        cl_summary_fmt["Revenue"] = "$" + cl_summary_fmt["Revenue"].round(2).astype(str)
                        st.dataframe(cl_summary_fmt, use_container_width=True, hide_index=True)
                except Exception as ex:
                    st.error(f"Clustering error: {ex}")
                    
        with ml_tab2:
            st.markdown("Train a Random Forest classifier to extract variables driving user conversions (Purchases).")
            
            if len(cleaned_df) < 500:
                st.warning("Insufficient dataset size to run classification models.")
            else:
                try:
                    importances = get_conversion_drivers(cleaned_df)
                    imp_df = pd.DataFrame(importances)
                    
                    col_imp_l, col_imp_r = st.columns([1, 1])
                    
                    with col_imp_l:
                        # Bar chart
                        fig_imp = px.bar(
                            imp_df.head(10),
                            x="importance",
                            y="feature",
                            orientation='h',
                            title="Top 10 Feature Importances Driving Conversion",
                            labels={"importance": "Importance Weight", "feature": "User Variable"},
                            color_discrete_sequence=["#1E3A8A"]
                        )
                        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
                        st.plotly_chart(fig_imp, use_container_width=True)
                        
                    with col_imp_r:
                        st.markdown("**Key Takeaways from ML Predictor:**")
                        top_features = imp_df.head(3)["feature"].tolist()
                        st.markdown(f"1. **{top_features[0]}** is the leading behavioral predictor of conversion.")
                        st.markdown(f"2. **{top_features[1]}** is the second most critical metric.")
                        st.markdown(f"3. **{top_features[2]}** ranks third in predicting conversions.")
                        st.markdown("Users showing high engagement on these key drivers should be targeted with targeted campaigns or optimized navigation routes.")
                except Exception as ex:
                    st.error(f"Classification modeling error: {ex}")
