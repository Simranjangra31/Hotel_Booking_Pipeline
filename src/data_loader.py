# data_loader.py
"""Data loading module for hotel booking cancellation prediction.
Loads the dataset from CSV and returns a pandas DataFrame.
"""
import pandas as pd
import os

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the hotel reservations CSV file.
    Args:
        csv_path: Path to the CSV file.
    Returns:
        pandas DataFrame with the data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df
