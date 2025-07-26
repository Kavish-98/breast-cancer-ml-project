# breast-cancer-ml-project
Machine Learning project for predicting breast cancer patient mortality and survival using classification models.

This repository contains three Jupyter Notebooks developed for a Machine Learning & Data Mining project. The project focuses on predicting mortality status and survival months of breast cancer patients using a real-world clinical dataset. It includes data understanding, preprocessing, model training, hyperparameter tuning, and evaluation using classification algorithms such as Logistic Regression, Decision Tree, and MLP.

# 🧠 Breast Cancer Mortality Prediction - Machine Learning Project

This project is  **Machine Learning & Data Mining**. It aims to predict breast cancer patient **mortality status** and **survival months** using clinical features from a real-world dataset.

The workflow is organized into three Jupyter Notebooks covering data understanding, preprocessing, model training, hyperparameter tuning, and evaluation.

---

## 📁 Notebooks Included

| Notebook | Description |
|----------|-------------|
| `Notebook1_DataUnderstanding.ipynb` | Exploratory Data Analysis (EDA), missing values, outlier detection, correlation analysis |
| `Notebook2_ModelTraining.ipynb` | Data preprocessing, feature selection, model building (Logistic Regression, Decision Tree, MLP) |
| `Notebook3_Evaluation_Tuning.ipynb` | Model evaluation (confusion matrix, accuracy, ROC-AUC), hyperparameter tuning, final results |

---

## 🧾 Dataset Overview

- **Total Records**: 4024 patients
- **Target Variables**:
  - `Mortality Status` (Alive or Dead)
  - `Survival Months` (Regression)
- **Features**: Clinical details including age, tumor stage, surgery type, and more

---

## ⚙️ Techniques Used

- Data Cleaning & Imputation
- IQR Method for Outlier Removal
- Train/Test Split with Stratification
- Logistic Regression, Decision Tree, MLP Classifier
- Model Evaluation: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Hyperparameter Tuning using `GridSearchCV`

---

## 🧪 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `imbalanced-learn` (if applied)
- `joblib` (for model saving, optional)

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-ml-project.git
   cd breast-cancer-ml-project
