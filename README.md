# Detecting Anti-Jewish Hate Crimes Using Machine Learning

### Technical Data Analysis & Predictive Modeling on NYPD Hate Crimes

This project implements an end-to-end **data analysis and machine learning framework** for identifying Anti-Jewish hate crimes from structured NYPD incident data.
The work emphasizes **feature engineering, model benchmarking, interpretability, and robust evaluation**, rather than simple prediction.

---

## Project Objective

**Task:** Binary classification
**Target:**

* `1` → Anti-Jewish
* `0` → Other Bias

**Goal:**
Learn discriminative patterns from historical hate crime data and evaluate multiple modeling strategies to detect the minority class with high recall and stability.

---

## Dataset Overview

| Property    | Value                          |
| ----------- | ------------------------------ |
| Records     | 3,872                          |
| Features    | 14                             |
| Class Ratio | 27.3% Anti-Jewish, 72.7% Other |
| Time Span   | 2010–2023                      |
| Source      | NYPD Hate Crimes (Data.gov)    |

### Feature Categories

* **Temporal:** Complaint Year, Month
* **Geospatial:** Precinct Code, Borough, County
* **Legal:** Law Code Category, Offense Description
* **Categorical:** PD Code Description, Offense Category

---

## Analytical Pipeline

**Raw Data → Cleaning → Encoding → Scaling → Feature Engineering → Model Training → Evaluation → Interpretation**

### Preprocessing

* Missing values

  * Categorical → `"Unknown"`
  * Numerical → Median
* Label Encoding for all categorical variables
* Standard Scaling for numeric features
* Stratified 80/20 Train-Test split

---

## Models Evaluated

| Model               | Role                                  |
| ------------------- | ------------------------------------- |
| Logistic Regression | Linear baseline, interpretable        |
| Decision Tree       | Non-linear interactions               |
| Random Forest       | Ensemble learning, variance reduction |

---

## Model Performance

| Model               | Accuracy  | Precision | Recall    | F1-Score  | AUC-ROC   |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.694     | 0.687     | 0.640     | 0.663     | 0.751     |
| Decision Tree       | 0.941     | 0.892     | **0.995** | 0.940     | 0.973     |
| **Random Forest**   | **0.948** | **0.926** | 0.967     | **0.946** | **0.978** |

**Key Insight:**
Random Forest provides the best trade-off between recall and precision, making it the most reliable model for minority-class detection.

---

## Confusion Matrix (Random Forest)

|                        | Predicted Anti-Jewish | Predicted Other |
| ---------------------- | --------------------- | --------------- |
| **Actual Anti-Jewish** | 352 (TP)              | 12 (FN)         |
| **Actual Other Bias**  | 28 (FP)               | 383 (TN)        |

* True Positive Rate: **96.7%**
* False Negative Rate: **3.3%**
* Strong detection of minority class with low misclassification.

---

## Feature Importance (Random Forest)

| Feature                 | Importance |
| ----------------------- | ---------- |
| Offense Category        | 0.554      |
| PD Code Description     | 0.137      |
| Complaint Precinct Code | 0.085      |
| Offense Description     | 0.059      |
| Month                   | 0.059      |

These variables contribute most to model decision boundaries.

---

## Technical Enhancements

* Stratified cross-validation
* Class weighting to handle imbalance
* Randomized & Bayesian hyperparameter tuning
* Advanced feature engineering:

  * Temporal cyclic encoding
  * Precinct-level aggregation
  * Borough statistics
  * Interaction features

---

## Interpretability

* Feature importance via Random Forest
* Logistic regression coefficients
* SHAP global and local explanations
* Partial Dependence Plots for feature effects

---

## Limitations

* Class imbalance
* Label encoding ignores semantic relationships
* No text embeddings used
* Model trained only on NYC data

---

## Future Improvements

* TF-IDF or transformer embeddings for text fields
* Gradient Boosting (XGBoost, LightGBM)
* Neural networks
* Multi-class classification for all bias types
* Cross-city generalization
