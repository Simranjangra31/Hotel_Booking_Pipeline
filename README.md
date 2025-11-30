# Hotel Booking Cancellation Prediction

A complete machine learning pipeline to predict hotel booking cancellations using Python, scikit-learn, and XGBoost.

## Project Structure

```
hotel_cancellation_prediction/
├── data/
│   └── Hotel Reservations.csv  (Dataset)
├── models/
│   └── (Saved models will go here)
├── notebooks/
│   └── (Jupyter notebooks for EDA)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          (Load and basic clean)
│   ├── feature_engineering.py  (Create predictive features)
│   ├── preprocessing.py        (Missing values, encoding, scaling)
│   ├── training.py             (Model selection, tuning, training)
│   └── evaluation.py           (Metrics, confusion matrix)
├── main.py                     (Complete ML pipeline runner)
├── requirements.txt            (Python dependencies)
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml              (CI/CD workflow)
```

## Features

- **Data Cleaning**: Handles missing values, duplicates, and type conversion
- **Feature Engineering**: Creates 5+ predictive features including:
  - Total stay nights
  - Total guests
  - Lead time categories
  - ADR per person
  - Weekend booking flag
- **Class Imbalance Handling**: Uses `class_weight='balanced'` and `scale_pos_weight`
- **Multiple Models**: Trains and compares Logistic Regression, Random Forest, and XGBoost
- **Hyperparameter Tuning**: GridSearchCV on XGBoost
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, feature importance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
python main.py --test
```

Train only:

```bash
python src/training.py
```

Evaluate only:

```bash
python src/evaluation.py
```

## Model Performance

- **Accuracy**: ~88%
- **ROC AUC**: ~0.95
- **F1 Score**: ~0.82

## Key Predictors

1. Market Segment (Online)
2. Number of Special Requests
3. Lead Time
4. Required Car Parking Space

## CI/CD

The project includes a GitHub Actions workflow that:
- Installs dependencies
- Runs the full pipeline
- Verifies model saving

## License

MIT
