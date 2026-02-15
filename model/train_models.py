"""
ML Assignment 2 - Model Training Script
=========================================
Dataset: Online Shoppers Purchasing Intention Dataset (UCI ML Repository)
URL: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

This script:
1. Downloads and preprocesses the dataset
2. Trains 6 classification models
3. Evaluates each model on 6 metrics
4. Saves trained models, preprocessors, and evaluation results
"""

import pandas as pd
import numpy as np
import os
import ssl
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Directory to save all model artifacts
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================
def load_dataset():
    """Download and load the Online Shoppers Purchasing Intention dataset."""
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    # Try loading from local file first (faster on re-runs)
    local_path = os.path.join(SAVE_DIR, '..', 'data', 'online_shoppers_intention.csv')
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"Dataset loaded from local file: {local_path}")
    else:
        try:
            # Handle SSL certificate issues on some systems
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            import urllib.request
            urllib.request.urlopen  # verify available
            req = urllib.request.Request(DATA_URL)
            with urllib.request.urlopen(req, context=ssl_context) as response:
                import io
                csv_data = response.read().decode('utf-8')
                df = pd.read_csv(io.StringIO(csv_data))
            print("Dataset downloaded successfully from UCI repository.")
            # Save locally for future use
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df.to_csv(local_path, index=False)
            print(f"Dataset saved locally: {local_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download dataset. Error: {e}\n"
                f"Please download manually from:\n"
                f"  {DATA_URL}\n"
                f"And place it at: {local_path}"
            )

    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1] - 1}")
    print(f"Number of Instances: {df.shape[0]}")
    print(f"\nTarget Variable Distribution:")
    print(df['Revenue'].value_counts())
    print(f"\nClass Balance: {df['Revenue'].value_counts(normalize=True).round(4).to_dict()}")

    return df


def preprocess_data(df):
    """Preprocess data: encode categoricals, scale features, train-test split."""
    print("\n" + "=" * 70)
    print("PREPROCESSING DATA")
    print("=" * 70)

    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Missing values found: {missing}. Dropping rows with missing values.")
        df = df.dropna()
        print(f"Shape after dropping: {df.shape}")
    else:
        print("No missing values found.")

    # Encode categorical features
    le_month = LabelEncoder()
    le_visitor = LabelEncoder()

    df['Month'] = le_month.fit_transform(df['Month'])
    df['VisitorType'] = le_visitor.fit_transform(df['VisitorType'])
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)

    print(f"Encoded 'Month' classes: {list(le_month.classes_)}")
    print(f"Encoded 'VisitorType' classes: {list(le_visitor.classes_)}")

    # Separate features and target
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    feature_names = list(X.columns)
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    print(f"Train class distribution: {dict(y_train.value_counts())}")
    print(f"Test class distribution:  {dict(y_test.value_counts())}")

    # Feature scaling (important for kNN and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessing objects
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    joblib.dump(le_month, os.path.join(SAVE_DIR, 'le_month.pkl'))
    joblib.dump(le_visitor, os.path.join(SAVE_DIR, 'le_visitor.pkl'))
    joblib.dump(feature_names, os.path.join(SAVE_DIR, 'feature_names.pkl'))
    print("\nPreprocessing objects saved to model/ directory.")

    # Save test data (unscaled, with target) for Streamlit app
    test_df = X_test.copy()
    test_df['Revenue'] = y_test.values
    test_df.to_csv(os.path.join(SAVE_DIR, 'test_data.csv'), index=False)
    print(f"Test data saved: test_data.csv ({test_df.shape[0]} rows)")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names


# ============================================================
# MODEL DEFINITIONS
# ============================================================
def get_models():
    """Define all 6 classification models with tuned hyperparameters."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=1.0,
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5
        ),
        'kNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1
        ),
        'XGBoost (Ensemble)': XGBClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            verbosity=0
        )
    }
    return models


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test):
    """Calculate all 6 evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)

    # Get probability scores for AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_proba), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return metrics, cm, report


# ============================================================
# TRAINING PIPELINE
# ============================================================
def train_all_models(X_train, X_test, y_train, y_test):
    """Train all 6 models, evaluate them, and save artifacts."""
    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 70)

    models = get_models()
    all_metrics = []
    all_confusion_matrices = {}
    all_reports = {}

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")

        # Train
        print("  Training...")
        model.fit(X_train, y_train)

        # Evaluate
        metrics, cm, report = evaluate_model(model, X_test, y_test)
        metrics['Model'] = name
        all_metrics.append(metrics)
        all_confusion_matrices[name] = cm.tolist()
        all_reports[name] = report

        # Save model
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{safe_name}.pkl"
        filepath = os.path.join(SAVE_DIR, filename)
        joblib.dump(model, filepath)

        print(f"  Saved: {filename}")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  AUC:       {metrics['AUC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1:        {metrics['F1']:.4f}")
        print(f"  MCC:       {metrics['MCC']:.4f}")
        print(f"  Confusion Matrix:\n  {cm}")

    # Save all evaluation artifacts
    metrics_df = pd.DataFrame(all_metrics)
    cols_order = ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    metrics_df = metrics_df[cols_order]
    metrics_df.to_csv(os.path.join(SAVE_DIR, 'metrics.csv'), index=False)
    joblib.dump(all_metrics, os.path.join(SAVE_DIR, 'all_metrics.pkl'))
    joblib.dump(all_confusion_matrices, os.path.join(SAVE_DIR, 'confusion_matrices.pkl'))
    joblib.dump(all_reports, os.path.join(SAVE_DIR, 'classification_reports.pkl'))

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    # Print markdown table for README
    print("\n\n" + "=" * 70)
    print("MARKDOWN TABLE (copy to README.md)")
    print("=" * 70)
    print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
    print("|---------------|----------|-----|-----------|--------|----|-----|")
    for m in all_metrics:
        print(f"| {m['Model']} | {m['Accuracy']} | {m['AUC']} | "
              f"{m['Precision']} | {m['Recall']} | {m['F1']} | {m['MCC']} |")

    return all_metrics


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  ML ASSIGNMENT 2 - MODEL TRAINING PIPELINE")
    print("  Dataset: Online Shoppers Purchasing Intention (UCI)")
    print("=" * 70)

    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # Step 3: Train and evaluate all models
    all_metrics = train_all_models(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 70)
    print("  ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print(f"  Model artifacts directory: {SAVE_DIR}")
    print("=" * 70)
    print("\nFiles saved:")
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.endswith(('.pkl', '.csv')):
            size = os.path.getsize(os.path.join(SAVE_DIR, f))
            print(f"  {f:40s} ({size:>10,} bytes)")
