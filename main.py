# main.py
"""Entry point for the hotel booking cancellation prediction pipeline.
It orchestrates data loading, preprocessing, feature engineering, model training,
hyper‑parameter tuning, and optional quick evaluation.
"""
import argparse
import logging
import sys

from src.training import train_models
from src.evaluation import evaluate_model


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(csv_path: str, test: bool = False):
    setup_logging()
    logging.info("Starting the hotel cancellation pipeline")

    # Train models (includes SMOTE and hyper‑parameter tuning)
    train_models(csv_path, model_output_path="models/model_pipeline.joblib")
    logging.info("Model training completed")

    if test:
        logging.info("Running quick evaluation on the full dataset")
        evaluate_model(csv_path, model_path="models/model_pipeline.joblib")
        logging.info("Evaluation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hotel cancellation pipeline")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/Hotel Reservations.csv",
        help="Path to the hotel reservations CSV file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick evaluation after training",
    )
    args = parser.parse_args()
    main(args.csv, test=args.test)
