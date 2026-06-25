import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def perform_user_segmentation(df, n_clusters=3, random_state=42):
    """
    Groups users into behavioral segments using K-Means clustering.
    
    Parameters:
    - df: Cleaned DataFrame (session logs)
    - n_clusters: Number of clusters (default 3)
    
    Returns:
    - user_df: DataFrame with one row per user and their cluster assignment.
    - cluster_summary: Summary statistics for each cluster.
    """
    # 1. Aggregate data by User ID
    user_df = df.groupby("User ID").agg({
        "Variant": "first",
        "Device": "first",
        "Country": "first",
        "Customer Type": "first",
        "Page Views": "sum",
        "Time on Site": "sum",
        "Purchase Completed": "max",
        "Revenue": "sum"
    }).reset_index()
    
    # 2. Features for clustering
    clustering_features = ["Page Views", "Time on Site", "Revenue"]
    X = user_df[clustering_features].copy()
    
    # 3. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    user_df["Cluster"] = kmeans.fit_predict(X_scaled)
    
    # 5. Determine cluster profiles to assign human-readable names
    # Calculate mean stats per cluster to label them dynamically
    cluster_means = user_df.groupby("Cluster")[clustering_features].mean()
    
    # Sort cluster IDs by mean revenue to label them
    sorted_by_revenue = cluster_means.sort_values(by="Revenue").index.tolist()
    
    # Lowest revenue = Casual Visitors
    # Highest revenue = High-Value Buyers
    # Middle = Engaged Browsers
    cluster_mapping = {}
    if n_clusters == 3:
        cluster_mapping[sorted_by_revenue[0]] = "Casual Visitors"
        cluster_mapping[sorted_by_revenue[1]] = "Engaged Browsers"
        cluster_mapping[sorted_by_revenue[2]] = "High-Value Buyers"
    else:
        for idx, cluster_id in enumerate(sorted_by_revenue):
            cluster_mapping[cluster_id] = f"Segment {idx+1}"
            
    user_df["Segment Name"] = user_df["Cluster"].map(cluster_mapping)
    
    # Calculate cluster summaries
    cluster_summary = user_df.groupby("Segment Name").agg({
        "User ID": "count",
        "Page Views": "mean",
        "Time on Site": "mean",
        "Purchase Completed": "mean",
        "Revenue": "mean"
    }).rename(columns={"User ID": "User Count", "Purchase Completed": "Conversion Rate"}).reset_index()
    
    return user_df, cluster_summary

def get_conversion_drivers(df, random_state=42):
    """
    Trains a Random Forest model on user features to calculate feature importances
    for predicting conversions.
    
    Parameters:
    - df: Cleaned DataFrame (session logs)
    
    Returns:
    - importances: list of dicts containing feature names and relative weights.
    """
    # 1. Aggregate to user level
    user_df = df.groupby("User ID").agg({
        "Variant": "first",
        "Device": "first",
        "Country": "first",
        "Customer Type": "first",
        "Page Views": "mean",      # Mean page views per session
        "Time on Site": "mean",    # Mean time on site per session
        "Purchase Completed": "max" # Converted (1) or not (0)
    }).reset_index()
    
    # 2. Prep categorical variables
    categorical_cols = ["Variant", "Device", "Country", "Customer Type"]
    features_df = user_df[["Page Views", "Time on Site"]].copy()
    
    # Dynamic dummy variables creation
    for col in categorical_cols:
        dummies = pd.get_dummies(user_df[col], prefix=col, drop_first=False)
        features_df = pd.concat([features_df, dummies], axis=1)
        
    X = features_df
    y = user_df["Purchase Completed"]
    
    # 3. Fit Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=random_state)
    rf.fit(X, y)
    
    # 4. Extract importances
    importances_arr = rf.feature_importances_
    features_list = X.columns.tolist()
    
    # Format and sort
    importances = []
    for feat, imp in zip(features_list, importances_arr):
        # Format names for readability (e.g. Variant_B -> Variant: B)
        clean_name = feat
        if "_" in feat:
            parts = feat.split("_", 1)
            clean_name = f"{parts[0]}: {parts[1]}"
        importances.append({
            "feature": clean_name,
            "raw_feature": feat,
            "importance": float(imp)
        })
        
    importances = sorted(importances, key=lambda x: x["importance"], reverse=True)
    return importances
