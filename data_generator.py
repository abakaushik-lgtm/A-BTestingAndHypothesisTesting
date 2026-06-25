import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_experiment_data(
    num_users=50000,
    conversion_rate_a=0.08,
    conversion_rate_b=0.095,
    revenue_mean_a=70.0,
    revenue_mean_b=75.0,
    aov_std=25.0,
    inject_anomalies=False,
    random_seed=42
):
    """
    Generates synthetic user activity logs for an A/B test experiment.
    
    Parameters:
    - num_users: Number of unique users to generate.
    - conversion_rate_a: Baseline conversion rate for Variant A (control).
    - conversion_rate_b: Baseline conversion rate for Variant B (treatment).
    - revenue_mean_a: Mean revenue for purchasing users in Variant A.
    - revenue_mean_b: Mean revenue for purchasing users in Variant B.
    - aov_std: Standard deviation of purchase revenue.
    - inject_anomalies: Whether to inject data quality issues (duplicates, nulls, invalid values).
    - random_seed: Random seed for reproducibility.
    
    Returns:
    - pd.DataFrame containing user sessions data.
    """
    np.random.seed(random_seed)
    
    # 1. Generate core user identifiers
    user_ids = [f"USR-{i:06d}" for i in range(100000, 100000 + num_users)]
    
    # Assign variants (roughly 50/50 split)
    variants = np.random.choice(["A", "B"], size=num_users, p=[0.5, 0.5])
    
    # User characteristics
    countries = np.random.choice(
        ["US", "UK", "DE", "FR", "JP", "IN", "CA", "AU"], 
        size=num_users, 
        p=[0.40, 0.15, 0.12, 0.08, 0.10, 0.08, 0.04, 0.03]
    )
    devices = np.random.choice(
        ["Desktop", "Mobile", "Tablet"], 
        size=num_users, 
        p=[0.45, 0.45, 0.10]
    )
    customer_types = np.random.choice(
        ["New", "Returning", "Premium"], 
        size=num_users, 
        p=[0.50, 0.40, 0.10]
    )
    
    # Build list of sessions
    data = []
    
    start_date = datetime(2026, 6, 1)
    
    for i in range(num_users):
        u_id = user_ids[i]
        var = variants[i]
        country = countries[i]
        device = devices[i]
        cust_type = customer_types[i]
        
        # Decide number of sessions for this user (1 to 4)
        num_sessions = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
        
        # Determine conversion probability based on variant, customer type, device
        # Premium and returning customers convert higher
        cust_modifier = 1.0
        if cust_type == "Premium":
            cust_modifier = 1.8
        elif cust_type == "Returning":
            cust_modifier = 1.3
            
        # Mobile converts slightly lower generally
        device_modifier = 1.0
        if device == "Mobile":
            device_modifier = 0.85
        elif device == "Tablet":
            device_modifier = 0.90
            
        base_p = conversion_rate_a if var == "A" else conversion_rate_b
        conversion_p = min(0.95, base_p * cust_modifier * device_modifier)
        
        for s in range(num_sessions):
            session_id = f"SES-{u_id[4:]}-{s}"
            
            # Days offset from start_date
            days_offset = np.random.randint(0, 25)
            seconds_offset = np.random.randint(0, 86400)
            timestamp = start_date + timedelta(days=days_offset, seconds=seconds_offset)
            
            # Engagement metrics
            # Converted users have higher engagement
            is_converted = np.random.random() < conversion_p
            
            if is_converted:
                page_views = int(np.random.negative_binomial(10, 0.5)) + 5  # Mean around 15 page views
                time_on_site = float(np.random.normal(300, 90))  # Mean 300 seconds
                checkout_started = 1
                purchase_completed = 1
                
                # Revenue calculation
                mean_rev = revenue_mean_a if var == "A" else revenue_mean_b
                # Premium users spend more
                if cust_type == "Premium":
                    mean_rev *= 1.5
                revenue = float(max(5.00, np.random.normal(mean_rev, aov_std)))
            else:
                # Did not convert
                checkout_started = 1 if np.random.random() < 0.25 else 0
                purchase_completed = 0
                revenue = 0.0
                
                if checkout_started:
                    page_views = int(np.random.negative_binomial(8, 0.6)) + 3 # Mean ~8
                    time_on_site = float(np.random.normal(150, 60))
                else:
                    page_views = int(np.random.negative_binomial(4, 0.7)) + 1 # Mean ~2-3
                    time_on_site = float(np.random.normal(60, 40))
            
            time_on_site = float(max(5.0, time_on_site))
            page_views = max(1, page_views)
            
            data.append({
                "User ID": u_id,
                "Timestamp": timestamp,
                "Variant": var,
                "Session ID": session_id,
                "Device": device,
                "Country": country,
                "Customer Type": cust_type,
                "Page Views": page_views,
                "Time on Site": round(time_on_site, 1),
                "Checkout Started": checkout_started,
                "Purchase Completed": purchase_completed,
                "Revenue": round(revenue, 2)
            })
            
    df = pd.DataFrame(data)
    
    # 2. Inject anomalies if requested
    if inject_anomalies:
        # Save a fraction of sizes
        n_records = len(df)
        
        # A. Missing values: User ID (0.5%), Country (0.5%), Device (0.5%), Variant (0.5%)
        null_indices_user = np.random.choice(n_records, size=int(n_records * 0.005), replace=False)
        df.loc[null_indices_user, "User ID"] = np.nan
        
        null_indices_country = np.random.choice(n_records, size=int(n_records * 0.005), replace=False)
        df.loc[null_indices_country, "Country"] = np.nan
        
        null_indices_device = np.random.choice(n_records, size=int(n_records * 0.005), replace=False)
        df.loc[null_indices_device, "Device"] = np.nan
        
        null_indices_var = np.random.choice(n_records, size=int(n_records * 0.005), replace=False)
        df.loc[null_indices_var, "Variant"] = np.nan
        
        # B. Duplicate records (exact row duplicates) - 1.5%
        dup_indices = np.random.choice(n_records, size=int(n_records * 0.015), replace=False)
        df_dups = df.iloc[dup_indices].copy()
        # Alter timestamp slightly or keep exact duplicate
        df = pd.concat([df, df_dups], ignore_index=True)
        n_records = len(df)
        
        # C. Negative Revenue (erroneous data entries) - 0.5%
        # Change positive revenue to negative
        pos_rev_indices = df[df["Revenue"] > 0].index.values
        if len(pos_rev_indices) > 0:
            neg_indices = np.random.choice(pos_rev_indices, size=min(len(pos_rev_indices), int(n_records * 0.005)), replace=False)
            df.loc[neg_indices, "Revenue"] = -df.loc[neg_indices, "Revenue"]
            
        # D. Invalid Timestamps (as strings or completely out of range) - 0.2%
        df["Timestamp"] = df["Timestamp"].astype(object)
        date_indices = np.random.choice(n_records, size=int(n_records * 0.002), replace=False)
        df.loc[date_indices, "Timestamp"] = "Invalid_Date_String"
        
        # E. Corrupted variant labels (e.g. 'control', 'treatment', 'variant_A')
        var_indices_a = df[df["Variant"] == "A"].index.values
        var_indices_b = df[df["Variant"] == "B"].index.values
        if len(var_indices_a) > 0:
            corrupt_a = np.random.choice(var_indices_a, size=min(len(var_indices_a), int(n_records * 0.005)), replace=False)
            df.loc[corrupt_a, "Variant"] = "control"
        if len(var_indices_b) > 0:
            corrupt_b = np.random.choice(var_indices_b, size=min(len(var_indices_b), int(n_records * 0.005)), replace=False)
            df.loc[corrupt_b, "Variant"] = "Variant_B"
            
        # F. User Variant Leakage (User belongs to BOTH Variant A and Variant B sessions)
        # Find some users with multiple sessions and swap variant in one of the sessions
        # Let's find unique user IDs that appear multiple times
        user_counts = df["User ID"].value_counts()
        multi_session_users = user_counts[user_counts > 1].index.values
        if len(multi_session_users) > 50:
            leak_users = np.random.choice(multi_session_users, size=50, replace=False)
            for u in leak_users:
                # Find all records for this user
                idx = df[df["User ID"] == u].index.values
                # Change the variant of the second session to the opposite of the first
                first_var = df.loc[idx[0], "Variant"]
                opposite_var = "B" if first_var in ["A", "control"] else "A"
                df.loc[idx[1], "Variant"] = opposite_var

    # Shuffle the final dataframe to look natural
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df
