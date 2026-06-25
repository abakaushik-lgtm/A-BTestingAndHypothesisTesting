import numpy as np
import pandas as pd
import scipy.stats as stats

def run_z_test(conv_a, n_a, conv_b, n_b, alpha=0.05):
    """
    Computes Two-Proportion Z-Test metrics.
    
    Parameters:
    - conv_a: Number of conversions in Control (A)
    - n_a: Total sample size in Control (A)
    - conv_b: Number of conversions in Treatment (B)
    - n_b: Total sample size in Treatment (B)
    - alpha: Significance level (default 0.05)
    
    Returns:
    - dict containing: conversion rates, lift, z-score, p-value, confidence intervals, and decision.
    """
    if n_a == 0 or n_b == 0:
        return {
            "rate_a": 0.0, "rate_b": 0.0, "abs_lift": 0.0, "rel_lift": 0.0,
            "z_stat": 0.0, "p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "significant": False
        }
        
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    
    abs_lift = p_b - p_a
    rel_lift = (p_b - p_a) / p_a if p_a > 0 else 0.0
    
    # Pooled conversion rate
    p_pooled = (conv_a + conv_b) / (n_a + n_b)
    
    # Standard error for hypothesis test (pooled)
    se_pooled = np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n_a + 1.0 / n_b))
    
    # Z-statistic
    if se_pooled > 0:
        z_stat = abs_lift / se_pooled
    else:
        z_stat = 0.0
        
    # P-value (two-tailed)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    
    # Standard error for confidence interval (unpooled)
    se_unpooled = np.sqrt((p_a * (1.0 - p_a) / n_a) + (p_b * (1.0 - p_b) / n_b))
    
    # Critical value
    z_critical = stats.norm.ppf(1.0 - alpha / 2.0)
    
    # Confidence interval bounds
    margin_of_error = z_critical * se_unpooled
    ci_lower = abs_lift - margin_of_error
    ci_upper = abs_lift + margin_of_error
    
    significant = p_value < alpha
    
    return {
        "rate_a": float(p_a),
        "rate_b": float(p_b),
        "abs_lift": float(abs_lift),
        "rel_lift": float(rel_lift),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": bool(significant)
    }

def run_chi_square_test(conv_a, n_a, conv_b, n_b):
    """
    Computes Chi-Square Test of Independence for conversions.
    
    Parameters:
    - conv_a, n_a, conv_b, n_b: counts
    
    Returns:
    - dict with chi2 stat, p-value, and significance status.
    """
    non_conv_a = n_a - conv_a
    non_conv_b = n_b - conv_b
    
    contingency_table = [
        [conv_a, non_conv_a],
        [conv_b, non_conv_b]
    ]
    
    # If any cell is 0, test can fail or be invalid
    if min(conv_a, non_conv_a, conv_b, non_conv_b) <= 0:
        return {
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "significant": False
        }
        
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table, correction=True)
    
    return {
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }

def run_t_test(values_a, values_b, alpha=0.05):
    """
    Computes Welch's Two-Sample Independent T-Test for continuous metrics.
    
    Parameters:
    - values_a: array-like (group A continuous values)
    - values_b: array-like (group B continuous values)
    - alpha: significance level
    
    Returns:
    - dict with mean_a, mean_b, rel_lift, t_stat, p_value, and confidence interval of difference.
    """
    n_a = len(values_a)
    n_b = len(values_b)
    
    if n_a < 2 or n_b < 2:
        return {
            "mean_a": 0.0, "mean_b": 0.0, "abs_lift": 0.0, "rel_lift": 0.0,
            "t_stat": 0.0, "p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "significant": False
        }
        
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    
    var_a = np.var(values_a, ddof=1)
    var_b = np.var(values_b, ddof=1)
    
    abs_lift = mean_b - mean_a
    rel_lift = (mean_b - mean_a) / mean_a if mean_a > 0 else 0.0
    
    # Perform Welch's T-Test (equal_var=False)
    t_stat, p_value = stats.ttest_ind(values_b, values_a, equal_var=False)
    
    # Welch-Satterthwaite degrees of freedom
    se_diff = np.sqrt(var_a / n_a + var_b / n_b)
    
    # Degrees of freedom calculation for Welch's T-test
    if se_diff > 0:
        df_numerator = (var_a / n_a + var_b / n_b) ** 2
        df_denominator = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
        df = df_numerator / df_denominator if df_denominator > 0 else (n_a + n_b - 2)
    else:
        df = n_a + n_b - 2
        
    # Critical value of t
    t_critical = stats.t.ppf(1.0 - alpha / 2.0, df)
    
    margin_of_error = t_critical * se_diff
    ci_lower = abs_lift - margin_of_error
    ci_upper = abs_lift + margin_of_error
    
    significant = p_value < alpha
    
    return {
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "abs_lift": float(abs_lift),
        "rel_lift": float(rel_lift),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": bool(significant)
    }

def analyze_segments(df, segment_col, alpha=0.05):
    """
    Performs Z-test conversion rate analysis segmented by a specific column.
    
    Parameters:
    - df: Cleaned DataFrame
    - segment_col: Column name to segment by (e.g. 'Device', 'Country')
    - alpha: Significance level
    
    Returns:
    - pd.DataFrame containing metrics for each segment level.
    """
    results = []
    
    # Unique values in segment column
    segments = df[segment_col].dropna().unique()
    
    for seg in segments:
        seg_df = df[df[segment_col] == seg]
        
        # User level aggregate
        user_agg = seg_df.groupby("User ID").agg({
            "Variant": "first",
            "Purchase Completed": "max"  # 1 if user made at least 1 purchase
        }).reset_index()
        
        n_a = user_agg[user_agg["Variant"] == "A"].shape[0]
        conv_a = user_agg[(user_agg["Variant"] == "A") & (user_agg["Purchase Completed"] == 1)].shape[0]
        
        n_b = user_agg[user_agg["Variant"] == "B"].shape[0]
        conv_b = user_agg[(user_agg["Variant"] == "B") & (user_agg["Purchase Completed"] == 1)].shape[0]
        
        z_res = run_z_test(conv_a, n_a, conv_b, n_b, alpha)
        
        results.append({
            "Segment Value": seg,
            "Users A": n_a,
            "Conversions A": conv_a,
            "Rate A": z_res["rate_a"],
            "Users B": n_b,
            "Conversions B": conv_b,
            "Rate B": z_res["rate_b"],
            "Relative Lift": z_res["rel_lift"],
            "P-Value": z_res["p_value"],
            "Significant": z_res["significant"]
        })
        
    return pd.DataFrame(results)
