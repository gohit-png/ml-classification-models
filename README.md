# ML Assignment 2 — Classification Models

## M.Tech (AIML/DSE) | Machine Learning

---

## a. Problem Statement

The goal of this project is to predict whether an online shopping session will result in a **purchase (Revenue = True)** or **not (Revenue = False)** based on various session-level features such as page visit counts, durations, bounce rates, and user demographic information.

This is a **binary classification** problem with real-world applications in e-commerce conversion optimization, targeted marketing, and user behavior analysis.

Six different machine learning classification models are trained, evaluated on six key metrics, and deployed via an interactive Streamlit web application.

---

## b. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

**Citation:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Comput & Applic 31, 6893–6908 (2019).

| Property | Value |
|----------|-------|
| **Number of Instances** | 12,330 |
| **Number of Features** | 17 |
| **Target Variable** | Revenue (Boolean: True/False) |
| **Task** | Binary Classification |
| **Class Distribution** | ~84.5% No Purchase, ~15.5% Purchase |
| **Missing Values** | 14 records (dropped during preprocessing) |

### Feature Descriptions

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | Administrative | Numerical | Number of administrative pages visited |
| 2 | Administrative_Duration | Numerical | Total time spent on administrative pages (seconds) |
| 3 | Informational | Numerical | Number of informational pages visited |
| 4 | Informational_Duration | Numerical | Total time spent on informational pages (seconds) |
| 5 | ProductRelated | Numerical | Number of product-related pages visited |
| 6 | ProductRelated_Duration | Numerical | Total time spent on product-related pages (seconds) |
| 7 | BounceRates | Numerical | Average bounce rate of pages visited |
| 8 | ExitRates | Numerical | Average exit rate of pages visited |
| 9 | PageValues | Numerical | Average page value of pages visited |
| 10 | SpecialDay | Numerical | Closeness of visit to a special day |
| 11 | Month | Categorical | Month of the session |
| 12 | OperatingSystems | Numerical | Operating system identifier |
| 13 | Browser | Numerical | Browser identifier |
| 14 | Region | Numerical | Geographic region identifier |
| 15 | TrafficType | Numerical | Traffic source type identifier |
| 16 | VisitorType | Categorical | New Visitor, Returning Visitor, or Other |
| 17 | Weekend | Boolean | Whether the session occurred on a weekend |

**Preprocessing Applied:**
- Label Encoding for categorical features (Month, VisitorType)
- Boolean to integer conversion (Weekend, Revenue)
- StandardScaler for feature normalization
- Stratified 80/20 train-test split (random_state=42)

---

## c. Models Used

### Classification Models Implemented

1. **Logistic Regression** — Linear model with L2 regularization (C=1.0, max_iter=1000)
2. **Decision Tree Classifier** — Non-linear tree model (max_depth=12, min_samples_split=10)
3. **K-Nearest Neighbor (kNN)** — Distance-based classifier (k=7, distance-weighted)
4. **Naive Bayes (Gaussian)** — Probabilistic classifier assuming Gaussian feature distributions
5. **Random Forest (Ensemble)** — Bagging ensemble of 150 decision trees (max_depth=15)
6. **XGBoost (Ensemble)** — Gradient boosting ensemble of 150 trees (learning_rate=0.1)

### Comparison Table — Evaluation Metrics

<!-- METRICS_TABLE_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8832 | 0.8653 | 0.7640 | 0.3560 | 0.4857 | 0.4696 |
| Decision Tree | 0.8804 | 0.8296 | 0.6390 | 0.5236 | 0.5755 | 0.5101 |
| kNN | 0.8775 | 0.8020 | 0.6942 | 0.3743 | 0.4864 | 0.4500 |
| Naive Bayes | 0.7794 | 0.8020 | 0.3802 | 0.6728 | 0.4858 | 0.3826 |
| Random Forest (Ensemble) | 0.8970 | 0.9173 | 0.7177 | 0.5524 | 0.6243 | 0.5723 |
| XGBoost (Ensemble) | 0.9011 | 0.9242 | 0.7226 | 0.5864 | 0.6474 | 0.5949 |
<!-- METRICS_TABLE_END -->

### Observations on Model Performance

<!-- OBSERVATIONS_TABLE_START -->
| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieves a high accuracy of 88.32% largely due to the class imbalance (84.5% negative class). The linear decision boundary captures some signal (AUC=0.8653) but struggles with recall (0.356), meaning it misses many actual purchasers. High precision (0.764) indicates that when it does predict a purchase, it is mostly correct. The low MCC (0.4696) reflects the imbalanced prediction pattern. Overall, it serves as a solid baseline but is not ideal for identifying buyers in this dataset. |
| Decision Tree | With controlled depth (max_depth=12), the Decision Tree achieves the best recall (0.5236) among non-ensemble models, correctly identifying over half the purchasers. However, its precision drops to 0.639, producing more false positives than Logistic Regression. The AUC of 0.8296 is moderate, suggesting limited ability to rank predictions. MCC of 0.5101 is higher than Logistic Regression, indicating a more balanced prediction. It captures non-linear patterns in browsing behavior but is prone to overfitting without depth constraints. |
| kNN | The distance-weighted kNN (k=7) achieves 87.75% accuracy with good precision (0.6942) but poor recall (0.3743), similar to Logistic Regression. Its AUC (0.802) is the lowest among all models, indicating weaker probabilistic ranking. Being a lazy learner, it relies entirely on the local neighborhood, which may not generalize well on this dataset where browsing patterns vary widely. Feature scaling (StandardScaler) helps, but the high dimensionality (17 features) reduces its effectiveness compared to tree-based models. |
| Naive Bayes | Gaussian NB has the lowest accuracy (77.94%) but notably the highest recall (0.6728) among all models — it identifies the most actual purchasers. However, this comes at the cost of very low precision (0.3802), producing many false positives (419 false alarms). The conditional independence assumption does not hold well for correlated web browsing features (e.g., BounceRates and ExitRates are correlated). Its MCC of 0.3826 is the lowest, confirming poor overall correlation between predicted and actual labels. Best suited when missing a purchaser is costlier than false alarms. |
| Random Forest (Ensemble) | The bagging ensemble of 150 trees achieves the second-best performance across most metrics: accuracy (89.70%), AUC (0.9173), F1 (0.6243), and MCC (0.5723). It significantly outperforms the single Decision Tree by reducing variance through averaging. The balanced precision (0.7177) and recall (0.5524) demonstrate better generalization. Feature importance analysis shows PageValues, BounceRates, and ExitRates as the top predictors. Its robustness to overfitting makes it a strong and reliable choice for this classification task. |
| XGBoost (Ensemble) | XGBoost achieves the best overall performance: highest accuracy (90.11%), AUC (0.9242), F1 (0.6474), and MCC (0.5949). The gradient boosting approach sequentially corrects errors, effectively handling the class imbalance and complex feature interactions in the dataset. It offers the best precision-recall trade-off (Precision=0.7226, Recall=0.5864) among all models. The strong AUC indicates excellent ranking capability. Its regularization mechanisms (controlled depth, learning rate) prevent overfitting. It is the recommended model for deployment in this purchase prediction task. |
<!-- OBSERVATIONS_TABLE_END -->

---

## Project Structure

```
project-folder/
│── app.py                  # Streamlit web application
│── requirements.txt        # Python dependencies
│── README.md               # This file
│── model/
│   ├── train_models.py     # Model training and evaluation script
│   ├── *.pkl               # Saved trained models and preprocessors
│   ├── metrics.csv         # Evaluation metrics summary
│   └── test_data.csv       # Saved test data for app
│── data/
│   └── online_shoppers_intention.csv  # Dataset (auto-downloaded)
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python model/train_models.py
```

This will:
- Download the dataset from UCI repository
- Preprocess and split the data
- Train all 6 classification models
- Save trained models, preprocessors, and metrics to `model/`

### 3. Run Streamlit App Locally

```bash
streamlit run app.py
```

### 4. Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App" → Select repository → Choose `app.py` → Deploy

---

## Streamlit App Features

- **Dataset upload option (CSV)** — Upload test data for evaluation
- **Model selection dropdown** — Choose any of the 6 trained models
- **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** — Visual heatmap with values
- **Classification report** — Detailed per-class precision, recall, F1

---

## Technologies Used

- Python 3.10+
- scikit-learn (ML models and metrics)
- XGBoost (gradient boosting)
- Streamlit (web application)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
