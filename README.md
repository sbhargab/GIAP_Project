# GIAP_Project
GIAP Capstone Project

# Heart Disease Prediction using Machine Learning

A simple machine learning project that predicts the presence of heart disease in patients using clinical features from the Kaggle Heart Disease Dataset.

## Overview

This project trains and evaluates three classification models:
- **Logistic Regression** – simple, interpretable baseline
- **Decision Tree** – visual and intuitive
- **Random Forest** – best performing (~98% accuracy)

## Dataset

Download `heart.csv` from Kaggle and place it in the project root:  
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset has 1,025 rows and 14 columns (13 features + 1 target).  
Target: `1` = Heart Disease, `0` = No Heart Disease.

## Project Structure

```
├── heart_disease_prediction.py
├── heart.csv 
├── requirements.txt 
└── README.md 
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/sbhargab/giap_project.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `heart.csv` from the Kaggle link above and place it in the project folder.

### 4. Run the script
```bash
python heart_disease_prediction.py
```

## Output

The script will print results to the console and save the following plots:

| File | Description |
|------|-------------|
| `plot_target_distribution.png` | Class balance bar chart |
| `plot_correlation_heatmap.png` | Feature correlation heatmap |
| `plot_age_vs_thalach.png` | Age vs Max Heart Rate scatter |
| `plot_cm_logistic_regression.png` | Confusion matrix |
| `plot_cm_decision_tree.png` | Confusion matrix |
| `plot_cm_random_forest.png` | Confusion matrix |
| `plot_roc_curves.png` | ROC curves for all models |
| `plot_feature_importance.png` | Random Forest feature importances |

## Results (Typical Run)

| Model | Accuracy | ROC-AUC | 5-Fold CV |
|-------|----------|---------|-----------|
| Logistic Regression | ~85% | ~93% | ~85% |
| Decision Tree | ~79% | ~79% | ~79% |
| **Random Forest** | **~98%** | **~99%** | **~98%** |

**Best Model: Random Forest**

## Features

| Feature | Description |
|---------|-------------|
| age | Patient age |
| sex | 1=Male, 0=Female |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar >120 mg/dl |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia (1-3) |
| **target** | **1=Disease, 0=No Disease** |

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn

