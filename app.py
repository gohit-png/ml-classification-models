"""
ML Assignment 2 - Streamlit Web Application
=============================================
Interactive classification model demonstration app.

Dataset: Online Shoppers Purchasing Intention Dataset (UCI)
Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost

Features:
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ML Assignment 2 - Classification Models",
    page_icon="ðŸ“Š",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_DIR = 'model'
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'kNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest (Ensemble)': 'random_forest_ensemble.pkl',
    'XGBoost (Ensemble)': 'xgboost_ensemble.pkl'
}


# ============================================================
# RESOURCE LOADING (CACHED)
# ============================================================
@st.cache_resource
def load_all_models():
    """Load all trained classification models."""
    models = {}
    for name, filename in MODEL_FILES.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
    return models


@st.cache_resource
def load_preprocessing():
    """Load preprocessing objects (scaler, encoders, feature names)."""
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    le_month = joblib.load(os.path.join(MODEL_DIR, 'le_month.pkl'))
    le_visitor = joblib.load(os.path.join(MODEL_DIR, 'le_visitor.pkl'))
    return scaler, feature_names, le_month, le_visitor


@st.cache_data
def load_precomputed_metrics():
    """Load pre-computed evaluation metrics from training."""
    path = os.path.join(MODEL_DIR, 'all_metrics.pkl')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def load_precomputed_cm():
    """Load pre-computed confusion matrices."""
    path = os.path.join(MODEL_DIR, 'confusion_matrices.pkl')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def load_precomputed_reports():
    """Load pre-computed classification reports."""
    path = os.path.join(MODEL_DIR, 'classification_reports.pkl')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def load_test_data():
    """Load the saved test dataset."""
    path = os.path.join(MODEL_DIR, 'test_data.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def preprocess_uploaded_data(df, scaler, feature_names, le_month, le_visitor):
    """Preprocess uploaded CSV data to match training format."""
    df = df.copy()

    # Encode categorical columns if present as strings
    if 'Month' in df.columns and df['Month'].dtype == 'object':
        known_months = set(le_month.classes_)
        df['Month'] = df['Month'].apply(
            lambda x: le_month.transform([x])[0] if x in known_months else -1
        )

    if 'VisitorType' in df.columns and df['VisitorType'].dtype == 'object':
        known_visitors = set(le_visitor.classes_)
        df['VisitorType'] = df['VisitorType'].apply(
            lambda x: le_visitor.transform([x])[0] if x in known_visitors else -1
        )

    if 'Weekend' in df.columns:
        df['Weekend'] = df['Weekend'].astype(int)

    # Separate target if present
    y = None
    if 'Revenue' in df.columns:
        df['Revenue'] = df['Revenue'].astype(int)
        y = df['Revenue'].values
        df = df.drop('Revenue', axis=1)

    # Validate columns
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in uploaded data: {missing_cols}")
        return None, None

    # Select and scale features
    X = df[feature_names]
    X_scaled = scaler.transform(X)

    return X_scaled, y


def compute_metrics(y_true, y_pred, y_proba):
    """Calculate all 6 evaluation metrics."""
    return {
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'AUC': round(roc_auc_score(y_true, y_proba), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4)
    }


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Create a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['No Purchase (0)', 'Purchase (1)'],
        yticklabels=['No Purchase (0)', 'Purchase (1)']
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_list):
    """Create a grouped bar chart comparing all models."""
    df = pd.DataFrame(metrics_list)
    metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metric_cols))
    n_models = len(df)
    width = 0.13

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']

    for i, (_, row) in enumerate(df.iterrows()):
        offset = x + (i - n_models / 2 + 0.5) * width
        bars = ax.bar(offset, [row[m] for m in metric_cols], width,
                      label=row['Model'], color=colors[i], alpha=0.85)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # â”€â”€ Header â”€â”€
    st.title("ðŸ“Š ML Classification Models Dashboard")
    st.markdown("""
    **Course:** M.Tech (AIML/DSE) - Machine Learning | **Assignment 2**

    **Dataset:** [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) (UCI ML Repository)

    **Task:** Binary Classification â€” Predict whether an online shopping session results in a purchase (Revenue)

    **Models Implemented:** Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost
    """)
    st.markdown("---")

    # â”€â”€ Sidebar â”€â”€
    st.sidebar.header("âš™ï¸ Settings")

    # Model selection dropdown [1 mark]
    selected_model = st.sidebar.selectbox(
        "ðŸ”½ Select ML Model",
        list(MODEL_FILES.keys()),
        index=0,
        help="Choose a classification model to view its detailed evaluation"
    )

    st.sidebar.markdown("---")

    # Dataset upload option (CSV) [1 mark]
    st.sidebar.header("ðŸ“ Upload Test Data (CSV)")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV with the same features as the training data. "
             "Include 'Revenue' column to see evaluation metrics."
    )

    if uploaded_file:
        st.sidebar.success(f"âœ… File uploaded: {uploaded_file.name}")

    # Download sample test data
    test_data = load_test_data()
    if test_data is not None:
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="ðŸ“¥ Download Sample Test CSV",
            data=test_data.to_csv(index=False).encode('utf-8'),
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download a sample test file you can use for upload"
        )

    # â”€â”€ Load resources â”€â”€
    try:
        scaler, feature_names, le_month, le_visitor = load_preprocessing()
        models = load_all_models()
        precomputed_metrics = load_precomputed_metrics()
        precomputed_cm = load_precomputed_cm()
        precomputed_reports = load_precomputed_reports()
    except Exception as e:
        st.error(f"âŒ Error loading models: {e}")
        st.info("Please run `python model/train_models.py` first to train and save models.")
        return

    # â”€â”€ Tabs â”€â”€
    tab1, tab2, tab3 = st.tabs([
        "ðŸ“ˆ All Models Comparison",
        "ðŸ” Individual Model Analysis",
        "ðŸ“‹ Dataset Information"
    ])

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TAB 1: MODEL COMPARISON
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    with tab1:
        st.header("Model Comparison â€” All 6 Classifiers")

        if precomputed_metrics:
            # Metrics comparison table [1 mark - display of evaluation metrics]
            st.subheader("ðŸ“Š Evaluation Metrics Table")
            metrics_df = pd.DataFrame(precomputed_metrics)
            cols_order = ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            metrics_df = metrics_df[cols_order]

            # Style the dataframe - highlight best values
            st.dataframe(
                metrics_df.style.highlight_max(
                    subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                    color='#90EE90'
                ).format({
                    'Accuracy': '{:.4f}', 'AUC': '{:.4f}',
                    'Precision': '{:.4f}', 'Recall': '{:.4f}',
                    'F1': '{:.4f}', 'MCC': '{:.4f}'
                }),
                width='stretch',
                hide_index=True
            )

            # Best model highlight
            best_idx = metrics_df['F1'].idxmax()
            best_model = metrics_df.loc[best_idx, 'Model']
            best_f1 = metrics_df.loc[best_idx, 'F1']
            st.success(f"ðŸ† **Best Model (by F1 Score):** {best_model} with F1 = {best_f1:.4f}")

            # Visual comparison chart
            st.subheader("ðŸ“‰ Visual Comparison")
            fig = plot_metrics_comparison(precomputed_metrics)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Pre-computed metrics not found. Run training script first.")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TAB 2: INDIVIDUAL MODEL ANALYSIS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    with tab2:
        st.header(f"Detailed Analysis: {selected_model}")

        # Determine evaluation data source
        X_eval, y_eval = None, None
        data_source = None

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                X_eval, y_eval = preprocess_uploaded_data(
                    uploaded_df, scaler, feature_names, le_month, le_visitor
                )
                data_source = "uploaded"
                st.info(f"ðŸ“ Using uploaded data: {uploaded_df.shape[0]} rows, "
                        f"{uploaded_df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")

        # If no upload or upload failed, use precomputed/saved test data
        if X_eval is None and precomputed_metrics:
            data_source = "precomputed"

        if data_source == "precomputed":
            # â”€â”€ Show pre-computed results â”€â”€
            model_metrics = next(
                (m for m in precomputed_metrics if m['Model'] == selected_model), None
            )

            if model_metrics:
                # Display evaluation metrics [1 mark]
                st.subheader("ðŸ“Š Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
                col2.metric("AUC Score", f"{model_metrics['AUC']:.4f}")
                col3.metric("Precision", f"{model_metrics['Precision']:.4f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", f"{model_metrics['Recall']:.4f}")
                col5.metric("F1 Score", f"{model_metrics['F1']:.4f}")
                col6.metric("MCC Score", f"{model_metrics['MCC']:.4f}")

                # Confusion matrix [1 mark]
                if precomputed_cm and selected_model in precomputed_cm:
                    st.subheader("ðŸ”¢ Confusion Matrix")
                    cm = np.array(precomputed_cm[selected_model])
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        fig = plot_confusion_matrix(
                            cm, f"Confusion Matrix â€” {selected_model}"
                        )
                        st.pyplot(fig)
                        plt.close()
                    with col_b:
                        st.markdown("**Confusion Matrix Values:**")
                        cm_df = pd.DataFrame(
                            cm,
                            index=['Actual: No Purchase', 'Actual: Purchase'],
                            columns=['Predicted: No Purchase', 'Predicted: Purchase']
                        )
                        st.dataframe(cm_df, width='stretch')

                # Classification report [1 mark]
                if precomputed_reports and selected_model in precomputed_reports:
                    st.subheader("ðŸ“‹ Classification Report")
                    report = precomputed_reports[selected_model]
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df.style.format("{:.4f}"),
                        width='stretch'
                    )

        elif X_eval is not None and y_eval is not None:
            # â”€â”€ Compute fresh results from uploaded data â”€â”€
            model = models.get(selected_model)
            if model:
                y_pred = model.predict(X_eval)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_eval)[:, 1]
                else:
                    y_proba = y_pred.astype(float)

                metrics = compute_metrics(y_eval, y_pred, y_proba)
                cm = confusion_matrix(y_eval, y_pred)
                report = classification_report(y_eval, y_pred, output_dict=True)

                # Metrics display [1 mark]
                st.subheader("ðŸ“Š Evaluation Metrics (Uploaded Data)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
                col3.metric("Precision", f"{metrics['Precision']:.4f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", f"{metrics['Recall']:.4f}")
                col5.metric("F1 Score", f"{metrics['F1']:.4f}")
                col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

                # Confusion matrix [1 mark]
                st.subheader("ðŸ”¢ Confusion Matrix")
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    fig = plot_confusion_matrix(
                        cm, f"Confusion Matrix â€” {selected_model}"
                    )
                    st.pyplot(fig)
                    plt.close()
                with col_b:
                    cm_df = pd.DataFrame(
                        cm,
                        index=['Actual: No Purchase', 'Actual: Purchase'],
                        columns=['Predicted: No Purchase', 'Predicted: Purchase']
                    )
                    st.dataframe(cm_df, width='stretch')

                # Classification report [1 mark]
                st.subheader("ðŸ“‹ Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(
                    report_df.style.format("{:.4f}"),
                    width='stretch'
                )

                # Predictions preview
                st.subheader("ðŸ”® Predictions Preview (first 20 rows)")
                pred_df = pd.DataFrame({
                    'Actual': y_eval[:20],
                    'Predicted': y_pred[:20],
                    'Purchase Probability': np.round(y_proba[:20], 4)
                })
                st.dataframe(pred_df, width='stretch')

        elif X_eval is not None and y_eval is None:
            # â”€â”€ No target column - predictions only â”€â”€
            model = models.get(selected_model)
            if model:
                y_pred = model.predict(X_eval)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_eval)[:, 1]
                else:
                    y_proba = y_pred.astype(float)

                st.info("â„¹ï¸ No 'Revenue' column found in uploaded data. "
                        "Showing predictions only (no evaluation metrics).")

                st.subheader("ðŸ”® Predictions")
                col1, col2 = st.columns(2)
                col1.metric("Predicted: No Purchase", int((y_pred == 0).sum()))
                col2.metric("Predicted: Purchase", int((y_pred == 1).sum()))

                pred_df = pd.DataFrame({
                    'Predicted': y_pred,
                    'Purchase Probability': np.round(y_proba, 4)
                })
                st.dataframe(pred_df, width='stretch')
        else:
            st.info("Upload a CSV file to see model evaluation on custom data, "
                    "or pre-computed results will be displayed automatically.")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TAB 3: DATASET INFORMATION
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    with tab3:
        st.header("Dataset Information")

        st.markdown("""
        ### Online Shoppers Purchasing Intention Dataset

        **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

        **Citation:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Real-time prediction
        of online shoppers' purchasing intention using multilayer perceptron and LSTM
        recurrent neural networks.* Neural Comput & Applic 31, 6893â€“6908 (2019).

        ---

        ### Problem Statement

        The goal is to predict whether an online shopping session will result in a
        **purchase (Revenue = True)** or **not (Revenue = False)** based on various
        session-level features such as page visit counts, durations, bounce rates,
        and user information.

        This is a **binary classification** problem with real-world applications in
        e-commerce conversion optimization, targeted marketing, and user behavior analysis.

        ---

        ### Dataset Statistics

        | Property | Value |
        |----------|-------|
        | **Number of Instances** | 12,330 |
        | **Number of Features** | 17 |
        | **Target Variable** | Revenue (Boolean) |
        | **Class Distribution** | ~84.5% No Purchase, ~15.5% Purchase |
        | **Missing Values** | None (14 records with missing values dropped) |

        ---

        ### Feature Descriptions

        | # | Feature | Type | Description |
        |---|---------|------|-------------|
        | 1 | Administrative | Numerical | Number of administrative pages visited |
        | 2 | Administrative_Duration | Numerical | Total time spent on administrative pages (seconds) |
        | 3 | Informational | Numerical | Number of informational pages visited |
        | 4 | Informational_Duration | Numerical | Total time spent on informational pages (seconds) |
        | 5 | ProductRelated | Numerical | Number of product-related pages visited |
        | 6 | ProductRelated_Duration | Numerical | Total time spent on product-related pages (seconds) |
        | 7 | BounceRates | Numerical | Average bounce rate of pages visited by the user |
        | 8 | ExitRates | Numerical | Average exit rate of pages visited by the user |
        | 9 | PageValues | Numerical | Average page value of pages visited by the user |
        | 10 | SpecialDay | Numerical | Closeness of visit to a special day (e.g., Valentine's Day) |
        | 11 | Month | Categorical | Month of the year the session occurred |
        | 12 | OperatingSystems | Numerical | Operating system identifier |
        | 13 | Browser | Numerical | Browser identifier |
        | 14 | Region | Numerical | Geographic region identifier |
        | 15 | TrafficType | Numerical | Traffic source type identifier |
        | 16 | VisitorType | Categorical | New Visitor, Returning Visitor, or Other |
        | 17 | Weekend | Boolean | Whether the session occurred on a weekend |
        """)

        # Show sample data
        if test_data is not None:
            st.subheader("Sample Data (Test Set Preview)")
            st.dataframe(test_data.head(15), width='stretch')
            st.caption(f"Showing first 15 rows of {test_data.shape[0]} test samples")

    # â”€â”€ Footer â”€â”€
    st.markdown("---")
    st.markdown(
        "**ML Assignment 2** | M.Tech (AIML/DSE) â€” Machine Learning | "
        "Online Shoppers Purchasing Intention Dataset | "
        "Built with Streamlit"
    )


if __name__ == '__main__':
    main()
