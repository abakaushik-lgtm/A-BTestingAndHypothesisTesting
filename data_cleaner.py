import pandas as pd
import numpy as np

def inspect_data_quality(df):
    """
    Analyzes the dataframe and returns a report of data quality issues.
    
    Parameters:
    - df: pd.DataFrame
    
    Returns:
    - dict containing counts of anomalies
    """
    report = {}
    
    # 1. Total records
    report["total_records"] = len(df)
    
    # 2. Missing User IDs
    report["missing_user_ids"] = df["User ID"].isna().sum()
    
    # 3. Missing Variant
    report["missing_variants"] = df["Variant"].isna().sum()
    
    # 4. Duplicate rows
    report["duplicate_records"] = df.duplicated().sum()
    
    # 5. Duplicate sessions (same session ID appearing more than once)
    report["duplicate_sessions"] = df["Session ID"].duplicated().sum() if "Session ID" in df.columns else 0
    
    # 6. Negative Revenue
    if "Revenue" in df.columns:
        report["negative_revenue"] = (df["Revenue"] < 0).sum()
    else:
        report["negative_revenue"] = 0
        
    # 7. Invalid Timestamps
    if "Timestamp" in df.columns:
        # Check how many cannot be parsed
        invalid_ts_count = 0
        for val in df["Timestamp"]:
            if pd.isna(val):
                invalid_ts_count += 1
            elif isinstance(val, str):
                try:
                    pd.to_datetime(val)
                except (ValueError, TypeError):
                    invalid_ts_count += 1
        report["invalid_timestamps"] = invalid_ts_count
    else:
        report["invalid_timestamps"] = 0
        
    # 8. Corrupted Variants (not 'A' and not 'B')
    if "Variant" in df.columns:
        valid_mask = df["Variant"].isin(["A", "B"])
        null_mask = df["Variant"].isna()
        report["corrupted_variants"] = (~valid_mask & ~null_mask).sum()
    else:
        report["corrupted_variants"] = 0
        
    # 9. User variant leakage (users appearing in both A and B groups)
    if "User ID" in df.columns and "Variant" in df.columns:
        # Get mapping of User ID to set of variants they have seen
        # Clean user_ids and variants first (ignore NaNs)
        valid_users = df.dropna(subset=["User ID", "Variant"])
        user_variants = valid_users.groupby("User ID")["Variant"].nunique()
        leaked_users = user_variants[user_variants > 1].index.tolist()
        report["leaked_users"] = len(leaked_users)
        report["leaked_records"] = df[df["User ID"].isin(leaked_users)].shape[0]
    else:
        report["leaked_users"] = 0
        report["leaked_records"] = 0
        
    return report

def clean_experiment_data(df):
    """
    Cleans the input dataframe according to rigorous experiment data standards.
    
    Parameters:
    - df: pd.DataFrame
    
    Returns:
    - pd.DataFrame (cleaned)
    - dict (cleaning summary)
    """
    initial_count = len(df)
    summary = {}
    
    cleaned_df = df.copy()
    
    # 1. Drop exact duplicate rows
    dups_before = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    summary["duplicates_removed"] = dups_before - len(cleaned_df)
    
    # 2. Parse timestamps and remove invalid ones
    if "Timestamp" in cleaned_df.columns:
        ts_before = len(cleaned_df)
        # Force convert to datetime, invalid entries become NaT
        cleaned_df["Timestamp"] = pd.to_datetime(cleaned_df["Timestamp"], errors="coerce")
        cleaned_df = cleaned_df.dropna(subset=["Timestamp"])
        summary["invalid_timestamps_removed"] = ts_before - len(cleaned_df)
    else:
        summary["invalid_timestamps_removed"] = 0
        
    # 3. Clean missing User IDs and Session IDs
    users_before = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=["User ID"])
    if "Session ID" in cleaned_df.columns:
        cleaned_df = cleaned_df.dropna(subset=["Session ID"])
    summary["missing_users_removed"] = users_before - len(cleaned_df)
    
    # 4. Standardize Variant column and clean nulls
    if "Variant" in cleaned_df.columns:
        # Standardize strings (e.g. 'control' -> 'A', 'Variant_B' -> 'B')
        def standardize_variant(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip().upper()
            if "CONTROL" in val_str or val_str == "A":
                return "A"
            elif "TREATMENT" in val_str or "VARIANT" in val_str or val_str == "B":
                return "B"
            else:
                return val_str  # Keep it as-is for now, will drop if invalid
                
        cleaned_df["Variant"] = cleaned_df["Variant"].apply(standardize_variant)
        
        # Check for non-standardized variants and remove them
        vars_before = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df["Variant"].isin(["A", "B"])]
        summary["corrupted_variants_cleaned"] = vars_before - len(cleaned_df)
    else:
        summary["corrupted_variants_cleaned"] = 0
        
    # 5. Resolve user variant leakage (split-assignment violations)
    if "User ID" in cleaned_df.columns and "Variant" in cleaned_df.columns:
        # Identify users who appear in both A and B
        user_variants = cleaned_df.groupby("User ID")["Variant"].nunique()
        leaked_user_ids = user_variants[user_variants > 1].index.tolist()
        
        leak_before = len(cleaned_df)
        cleaned_df = cleaned_df[~cleaned_df["User ID"].isin(leaked_user_ids)]
        summary["leaked_users_count"] = len(leaked_user_ids)
        summary["leaked_records_removed"] = leak_before - len(cleaned_df)
    else:
        summary["leaked_users_count"] = 0
        summary["leaked_records_removed"] = 0
        
    # 6. Handle Negative Revenue
    if "Revenue" in cleaned_df.columns:
        rev_before = len(cleaned_df)
        # Let's drop records with negative revenue, as it suggests log corruption
        cleaned_df = cleaned_df[cleaned_df["Revenue"] >= 0]
        summary["negative_revenue_records_removed"] = rev_before - len(cleaned_df)
    else:
        summary["negative_revenue_records_removed"] = 0
        
    # 7. Check for page views and time on site anomalies
    if "Page Views" in cleaned_df.columns:
        pv_before = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df["Page Views"] >= 0]
        summary["invalid_page_views_removed"] = pv_before - len(cleaned_df)
    else:
        summary["invalid_page_views_removed"] = 0
        
    if "Time on Site" in cleaned_df.columns:
        tos_before = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df["Time on Site"] >= 0]
        summary["invalid_time_on_site_removed"] = tos_before - len(cleaned_df)
    else:
        summary["invalid_time_on_site_removed"] = 0
        
    # 8. Post-cleaning duplicates check (duplicates might be created during variant standardization)
    post_dups_before = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    summary["duplicates_removed"] += (post_dups_before - len(cleaned_df))

    summary["final_records_count"] = len(cleaned_df)
    summary["initial_records_count"] = initial_count
    summary["total_records_removed"] = initial_count - len(cleaned_df)
    
    return cleaned_df, summary
