import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def handle_missing_values(
    df: pd.DataFrame, 
    missing_threshold: float = 0.1,
    max_consecutive_ffill: int = 3
) -> pd.DataFrame:
    """
    Handles missing values using time-series logical structures:
    - Bounded forward-fill (ffill) to preserve recent physics.
    - Fallback to session median, then global median.
    - Fields with missing data >= missing_threshold are explicitly flagged.
    """
    if df.empty:
        return df.copy()
        
    df = df.copy()
    
    # CRITICAL: Time series operations require guaranteed chronological ordering
    if "session_id" in df.columns and "timestamp_utc" in df.columns:
        df = df.sort_values(["session_id", "timestamp_utc"]).reset_index(drop=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        missing_pct = df[col].isna().mean()
        
        if missing_pct == 0:
            continue

        # --- Flagging Severe Missing Data ---
        if missing_pct >= missing_threshold:
            logger.warning(f"Column '{col}' has {missing_pct:.2%} missing! Creating structural flag.")
            # Create a flag column so the model learns this was missing space
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
            
        logger.info(f"Imputing gaps for '{col}' (Missing: {missing_pct:.2%}).")
        
        # --- Imputation Pipeline (Runs for ALL columns to ensure 0 NaNs) ---
        
        # Step A: Strict Forward Fill ONLY (No bfill, prevents future leakage)
        df[col] = df.groupby("session_id")[col].ffill(limit=max_consecutive_ffill)
        
        # Step B: Fallback for remaining large gaps
        if df[col].isna().sum() > 0:
            # Calculate the median within each session context
            session_medians = df.groupby("session_id")[col].transform("median")
            # If a session is entirely NaN, fallback to global median
            global_median = df[col].median()
            
            # Chain the fills to guarantee a completely clean numeric column
            df[col] = df[col].fillna(session_medians).fillna(global_median)
            
    return df


def verify_target_distribution(df: pd.DataFrame):
    """
    Prints the class balance of the target variables.
    """
    print("\n--- Target Variable Distributions ---")
    
    if "Y" in df.columns:
        dist = df["Y"].value_counts(normalize=True) * 100
        counts = df["Y"].value_counts()
        print(f"\nTarget: Y (binary label)")
        for val in dist.index:
            print(f"  Class {val}: {counts[val]} rows ({dist[val]:.2f}%)")
            
    if "thermal_pressure_code" in df.columns:
        dist = df["thermal_pressure_code"].value_counts(normalize=True) * 100
        counts = df["thermal_pressure_code"].value_counts()
        print(f"\nTarget: thermal_pressure_code")
        for val in dist.index:
            print(f"  Code {val}: {counts[val]} rows ({dist[val]:.2f}%)")