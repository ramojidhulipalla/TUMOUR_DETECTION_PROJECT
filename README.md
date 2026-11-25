# Tumor Detection Project

Tute Dude Data Science Course - Assignment 7 (Module 14)

## Project Description

This project classifies tumors as **malignant (M)** or **benign (B)** from numerical tumor measurements using a **Random Forest Classifier**. The complete workflow - data loading, cleaning, exploratory data analysis, correlation analysis, standard scaling, train-test split, model training and evaluation - is in the Jupyter Notebook `Tumor_Detection.ipynb`.

## Problem Statement

Build a machine learning model that predicts whether a tumor is malignant (M) or benign (B) using the tumor characteristics (mean radius, texture, perimeter, etc.) stored in `Tumor_Detection.csv`, and visualise the patterns in the data.

## Dataset Description

The information below was verified by inspecting the provided `Tumor_Detection.csv`:

- **Rows:** 569 tumor samples
- **Columns in the raw file:** 33
- **Target column:** `diagnosis` with values `M` (malignant, 212 cases, 37.26%) and `B` (benign, 357 cases, 62.74%)
- **Identifier column:** `id` (removed - not a predictive feature)
- **Extra column:** `Unnamed: 32` (completely empty - removed)
- **Features:** 30 numeric measurements made up of ten measurements, each given as a `mean_`, an `_error` and a `worst_` version (for example `mean_radius`, `radius_error`, `worst_radius`)
- **Missing values** after removing `Unnamed: 32`: none
- **Duplicate rows:** none found

## Libraries Used

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (and the built-in `warnings` module)

## Methodology

1. Imported the libraries and loaded `Tumor_Detection.csv`.
2. Inspected the data with `head()`, `tail()`, `shape`, column names, `info()` and `describe()`.
3. Cleaned the data: removed `id` and `Unnamed: 32`, checked missing values with `isnull().sum()`, checked duplicates, kept the `diagnosis` target intact.
4. Performed EDA and visualisation.
5. Calculated the correlation matrix of the numeric features and plotted it as a heatmap.
6. Defined `X` (the 30 features) and `y` (`diagnosis` mapped as **M = 1, B = 0**, verified with a cross-tabulation).
7. Split the data with `train_test_split(test_size=0.2, random_state=42, stratify=y)` -> 455 training and 114 testing samples.
8. Applied `StandardScaler`: **fitted on the training data only**, then used to transform the test data (no data leakage).
9. Trained `RandomForestClassifier(random_state=42)` and predicted on the test set.
10. Evaluated the model and wrote the conclusion.

## EDA Performed

- Distribution and percentage of malignant vs benign cases
- Count plot of the `diagnosis` column
- Descriptive statistics of all numeric features
- Average of each `mean_` feature for benign vs malignant tumors
- Histograms and box plots of key features by diagnosis
- Correlation heatmap with correlation values, the strongest correlated feature pairs, and the features most correlated with the diagnosis

## Machine Learning Algorithm

`RandomForestClassifier` from `sklearn.ensemble` (`random_state=42`, default 100 trees). Random Forest does not strictly need scaling, but `StandardScaler` was included because the assignment requires it.

## Model Evaluation Approach

The model is evaluated only on the unseen 20% test set using `accuracy_score()`, the confusion matrix (table and heatmap) and `classification_report()`.

## Actual Results

These results were produced by executing the notebook:

| Metric | Value |
|---|---|
| Test accuracy | **97.37%** (111 of 114 correct) |
| True negatives (benign predicted benign) | 72 |
| False positives (benign predicted malignant) | 0 |
| False negatives (malignant predicted benign) | 3 |
| True positives (malignant predicted malignant) | 39 |
| Malignant precision / recall / F1 | 1.00 / 0.93 / 0.96 |
| Benign F1 | 0.98 |

Most important features according to the Random Forest: `worst_area`, `worst_concave_points`, `worst_radius`.

**Limitation:** This is an educational machine learning project using a single dataset and a single train-test split. It is **not a clinically validated diagnostic system** and must not be used for real medical decisions.

## Project Structure

```
Tumor_Detection-project/
├── Tumor_Detection.ipynb
├── Tumor_Detection.csv
└── README.md
```

## How to Run the Notebook

1. Install Python 3 and the required libraries:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Open a terminal in the `Tumor_Detection-project` folder (the CSV file must stay in the same folder as the notebook).
3. Start Jupyter:
   ```
   jupyter notebook
   ```
4. Open `Tumor_Detection.ipynb` and choose **Kernel -> Restart & Run All**.
