# to me, some important features to engineer are: cpu_power_mw;  cpu_pcluster_active_pct; cpu_ecluster_freq_mhz



import pandas as pd
import numpy as np
import logging
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame, interval_seconds: int = 5) -> pd.DataFrame:
    """
    Phase 4: Feature Engineering
    Optimized for Pandas memory efficiency to prevent highly fragmented DataFrames.
    """
    if df.empty:
        logger.warning("engineer_features received an empty DataFrame.")
        return df.copy()

    df = df.copy()
    df = df.sort_values(["session_id", "timestamp_utc"]).reset_index(drop=True)

    logger.info("Engineering features grouped by session_id.")

    eps = 1e-6
    steps_30s = int(30 / interval_seconds)   
    steps_1m  = int(60 / interval_seconds)   
    steps_5m  = int(300 / interval_seconds)  
    steps_15m = int(900 / interval_seconds)  

    lag_steps = [1, 2, 3, 4, 5, 10]

    lag_vars = [
        "cpu_die_temp_c",
        "cpu_power_mw",
        "combined_power_mw", 
        "cpu_pcluster_active_pct",
        "cpu_pcluster_freq_mhz", 
    ]

    rolling_vars = [
        "cpu_die_temp_c",
        "combined_power_mw",
        "cpu_power_mw",
        "cpu_pcluster_active_pct",
        "cpu_pcluster_freq_mhz",
    ]

    windows = [
        ("30s", steps_30s),
        ("1m", steps_1m),
        ("5m", steps_5m),
    ]

    # Initialize a dictionary to hold all new feature columns
    new_features = {}

    # 1. Lag features
    for col in lag_vars:
        if col in df.columns:
            g = df.groupby("session_id")[col]
            for lag in lag_steps:
                new_features[f"{col}_lag_{lag}"] = g.shift(lag)

    # 2. Rolling statistics
    for col in rolling_vars:
        if col in df.columns:
            g = df.groupby("session_id")[col]

            for label, window in windows:
                roll = g.rolling(window=window, min_periods=max(3, window // 2))

                # Extract series directly to the dictionary
                new_features[f"{col}_roll_mean_{label}"] = roll.mean().reset_index(level=0, drop=True)
                new_features[f"{col}_roll_std_{label}"]  = roll.std().reset_index(level=0, drop=True)
                new_features[f"{col}_roll_max_{label}"]  = roll.max().reset_index(level=0, drop=True)
                new_features[f"{col}_roll_min_{label}"]  = roll.min().reset_index(level=0, drop=True)
                
                # Compute range from the dict entries
                new_features[f"{col}_roll_range_{label}"] = (
                    new_features[f"{col}_roll_max_{label}"] - new_features[f"{col}_roll_min_{label}"]
                )

    # Selective 15m heat-soak features
    for col in ["cpu_power_mw", "cpu_die_temp_c", "cpu_pcluster_freq_mhz"]:
        if col in df.columns:
            new_features[f"{col}_roll_mean_15m"] = (
                df.groupby("session_id")[col]
                  .rolling(window=steps_15m, min_periods=10)
                  .mean()
                  .reset_index(level=0, drop=True)
            )

    # 3. Rate of change / acceleration
    diff_vars = ["cpu_die_temp_c", "cpu_power_mw", "cpu_pcluster_freq_mhz"]

    for col in diff_vars:
        if col in df.columns:
            g = df.groupby("session_id")[col]
            new_features[f"{col}_diff_1"] = g.diff(1)
            new_features[f"{col}_diff_6"] = g.diff(steps_30s)
            
            # Group the newly created diff_1 by session to get acceleration
            new_features[f"{col}_accel"] = (
                new_features[f"{col}_diff_1"].groupby(df["session_id"]).diff(1)
            )

    # 4. Interaction / domain features
    if {"cpu_die_temp_c", "cpu_pcluster_freq_mhz"}.issubset(df.columns):
        new_features["thermal_distress_proxy"] = df["cpu_die_temp_c"] / (df["cpu_pcluster_freq_mhz"] + eps)

    if {"cpu_power_mw", "cpu_pcluster_active_pct"}.issubset(df.columns):
        new_features["pcluster_power_pressure"] = df["cpu_power_mw"] * df["cpu_pcluster_active_pct"]

    if {"ram_used_gb", "ram_total_gb"}.issubset(df.columns):
        new_features["ram_pressure_ratio"] = df["ram_used_gb"] / (df["ram_total_gb"] + eps)

    # --- THE FIX: Concatenate all new features at once ---
    new_features_df = pd.DataFrame(new_features)
    df = pd.concat([df, new_features_df], axis=1)

    # Force a defragmentation copy just to be perfectly safe
    df = df.copy()

    logger.info("Feature engineering complete.")
    return df

def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop boundary rows created by lag/rolling operations.

    We anchor validity on:
    - the largest short-memory rolling feature used in core modelling
    - the largest lag used
    - the target label Y
    """
    if df.empty:
        return df.copy()

    # Now perfectly aligns with the generated features
    required_cols = [col for col in [
        "combined_power_mw_roll_mean_5m",
        "combined_power_mw_lag_10", 
        "Y"
    ] if col in df.columns]

    initial_rows = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped = initial_rows - len(df)

    logger.info("Dropped %d boundary rows with invalid feature history.", dropped)

    return df




def fit_tree_importance(X_train, y_train, feature_names):
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    gb_imp = pd.Series(gb.feature_importances_, index=feature_names).sort_values(ascending=False)

    return rf, gb, rf_imp, gb_imp


def get_permutation_importance(model, X_val, y_val, feature_names, scoring="f1"):
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=20,
        random_state=42,
        scoring=scoring,
        n_jobs=-1
    )
    imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    std = pd.Series(result.importances_std, index=feature_names).loc[imp.index]
    return imp, std

def feature_family(name):
    if "_lag_" in name:
        return "lag"
    if "_roll_" in name:
        return "rolling"
    if name.endswith("_diff_1") or name.endswith("_diff_6"):
        return "diff"
    if name.endswith("_accel"):
        return "accel"
    if name in ["pcluster_power_pressure", "thermal_distress_proxy"]:
        return "interaction"
    if name in [
        "cpu_die_temp_c", "cpu_power_mw", "combined_power_mw",
        "cpu_pcluster_active_pct", "cpu_pcluster_freq_mhz"
    ]:
        return "raw_core"
    return "other"

def aggregate_family_importance(importance_series):
    df = importance_series.reset_index()
    df.columns = ["feature", "importance"]
    df["family"] = df["feature"].apply(feature_family)
    return df.groupby("family")["importance"].sum().sort_values(ascending=False)