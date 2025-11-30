# feature_engineering.py
"""Feature engineering utilities for the hotel booking cancellation dataset.
Adds engineered columns that improve model performance.
"""
import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the DataFrame.
    Expected input: DataFrame after basic cleaning (see preprocess.clean_data).
    Returns a new DataFrame with additional columns.
    """
    df = df.copy()
    # Total stay nights
    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
        df["total_stay_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    # Total guests (adults + children + babies)
    if {"adults", "no_of_children", "babies"}.issubset(df.columns):
        df["total_guests"] = df["adults"] + df["no_of_children"] + df["babies"]
    # Lead time category
    if "lead_time" in df.columns:
        bins = [-1, 30, 180, df["lead_time"].max()]
        labels = ["short", "medium", "long"]
        df["lead_time_category"] = pd.cut(df["lead_time"], bins=bins, labels=labels)
    # ADR per person (avoid division by zero)
    if "adr" in df.columns:
        df["adr_per_person"] = df.apply(
            lambda row: row["adr"] / row["total_guests"] if row["total_guests"] > 0 else 0,
            axis=1,
        )
    # Weekend booking flag based on arrival_day_of_week (0=Mon ... 6=Sun)
    if "arrival_day_of_week" in df.columns:
        df["is_weekend_booking"] = df["arrival_day_of_week"].isin([5, 6]).astype(int)
    # Month of arrival (already extracted as arrival_month_num in preprocessing)
    # Ensure column exists; if not, create from arrival_month_num if present
    if "arrival_month_num" not in df.columns and "arrival_month" in df.columns:
        df["arrival_month_num"] = df["arrival_month"]
    return df
