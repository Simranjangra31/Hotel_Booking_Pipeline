# preprocess.py
"""Data cleaning and preprocessing utilities for the hotel booking dataset.
This module handles missing values, duplicate removal, type conversion, and target encoding.
It also provides a scikit-learn compatible ColumnTransformer for categorical encoding.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning on the raw DataFrame.
    Steps:
    1. Drop exact duplicate rows.
    2. Convert date columns (arrival_year, arrival_month, arrival_date) into a datetime column.
    3. Encode the target column `booking_status` to binary (0 = Not_Canceled, 1 = Canceled).
    4. Ensure numeric columns are of proper dtype.
    """
    # 1. Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Create a proper datetime column
    # The dataset stores year, month, day as separate columns
    if {"arrival_year", "arrival_month", "arrival_date"}.issubset(df.columns):
        df["arrival_dt"] = pd.to_datetime(
            df[["arrival_year", "arrival_month", "arrival_date"]]
            .astype(str)
            .apply(lambda x: f"{x['arrival_year']}-{int(x['arrival_month']):02d}-{int(x['arrival_date']):02d}", axis=1),
            errors="coerce",
        )
        # Extract useful components
        df["arrival_month_num"] = df["arrival_dt"].dt.month
        df["arrival_day_of_week"] = df["arrival_dt"].dt.dayofweek  # Monday=0
        df = df.drop(columns=["arrival_year", "arrival_month", "arrival_date", "arrival_dt"])

    # 3. Encode target
    if "booking_status" in df.columns:
        df["target"] = df["booking_status"].map({"Not_Canceled": 0, "Canceled": 1})
        df = df.drop(columns=["booking_status"])

    # 4. Ensure numeric columns are numeric (some may be read as object)
    # Removed aggressive conversion to avoid turning categorical columns into NaNs
    # numeric_cols = df.select_dtypes(include=["object"]).columns
    # for col in numeric_cols:
    #     df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that handles missing values and one‑hot encodes categoricals.
    The function inspects the DataFrame to decide which columns are categorical vs numeric.
    """
    # Identify categorical columns (object dtype after cleaning)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Identify numeric columns (excluding the target column)
    numeric_cols = [c for c in df.columns if c not in categorical_cols + ["target"]]

    # Pipelines
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor
