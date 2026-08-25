# 🌳 Decision Tree & Random Forest Classification

A Machine Learning laboratory project that implements and compares **Decision Tree** and **Random Forest** classifiers for breast cancer diagnosis using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

> **Course:** ICS1512 – Machine Learning Algorithms Laboratory  
> **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

## 📌 Overview

This project explores tree-based supervised learning models for binary classification. A single **Decision Tree** and a **Random Forest** ensemble are trained, optimized using **5-fold GridSearchCV**, and evaluated to study overfitting, generalization, and feature importance.

---

## 🎯 Objectives

- Implement a Decision Tree classifier.
- Build a Random Forest ensemble model.
- Analyze the impact of tree depth and other hyperparameters.
- Perform hyperparameter tuning using 5-fold cross-validation.
- Compare single-tree and ensemble-tree performance.

---

## 📂 Dataset

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

| Property | Value |
|----------|------|
| Samples | 569 |
| Features | 30 numerical features |
| Classes | Benign (0), Malignant (1) |
| Missing Values | 0 |
| Train/Test Split | 80% / 20% (Stratified) |

The dataset contains measurements computed from digitized images of breast cell nuclei and is used to classify tumors as **Benign** or **Malignant**.

---

## 🛠️ Technologies Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 🤖 Machine Learning Models

### Decision Tree
- Entropy & Gini criteria
- Depth optimization
- Split & leaf regularization

### Random Forest
- Ensemble of Decision Trees
- Random feature sampling
- GridSearchCV optimization
- Feature importance analysis

---

## 📊 Exploratory Data Analysis

The notebook includes:

- Class distribution visualization
- Pearson correlation heatmap
- Feature importance analysis
- Confusion matrices
- ROC curve comparison

---

## 🏆 Final Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|:-------:|:---------:|:------:|:--------:|:------:|
| Decision Tree | 92.98% | **100.00%** | 80.95% | 89.47% | 0.9563 |
| **Random Forest** | **96.49%** | **100.00%** | **90.48%** | **95.00%** | **0.9950** |

---

## ⚙️ Best Hyperparameters

### Decision Tree

| Parameter | Value |
|----------|------|
| Criterion | Entropy |
| Max Depth | 4 |
| Min Samples Split | 2 |
| Min Samples Leaf | 2 |

### Random Forest

| Parameter | Value |
|----------|------|
| Trees | 50 |
| Max Depth | 10 |
| Max Features | sqrt |
| Bootstrap | False |

---

## 📈 5-Fold Cross Validation

| Model | Average Accuracy |
|------|:----------------:|
| Decision Tree | 94.07% |
| **Random Forest** | **97.14%** |

Random Forest demonstrated better stability and higher average accuracy across all validation folds.

---

## 🌟 Top Important Features

The Random Forest identified the most influential predictors as:

1. `concave_points_worst`
2. `area_worst`
3. `perimeter_worst`
4. `radius_worst`
5. `concavity_worst`

These geometric features contribute most strongly to distinguishing malignant from benign tumors.

---

## 📊 Visualizations Included

The notebook generates:

- Class Distribution
- Feature Correlation Heatmap
- Decision Tree Confusion Matrix
- Random Forest Confusion Matrix
- ROC Curve Comparison
- Performance Metric Comparison
- Top 10 Feature Importance Plot

---

## 🚀 Project Structure

```text
Decision-Tree-Random-Forest/
│
├── datasets/
│   └── wdbc.csv
│
├── notebooks/
│   └── Experiment_05.ipynb
│
├── outputs/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── feature_importance/
│
└── README.md
```

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/sangee-3010/Machine_Learning.git
cd Machine_Learning
```

### Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open **Experiment_05.ipynb** and run all cells.

---

## 📚 Learning Outcomes

- Implemented Decision Tree and Random Forest classifiers.
- Performed hyperparameter tuning using GridSearchCV.
- Compared overfitting and generalization in tree-based models.
- Analyzed feature importance for medical diagnosis.
- Evaluated models using cross-validation and ROC-AUC.

---

## 👩‍💻 Author

**Sangeetha S K**

- GitHub: https://github.com/sangee-3010

---

## 📖 References

- UCI Machine Learning Repository – Wisconsin Diagnostic Breast Cancer Dataset
- Scikit-learn Documentation
- NumPy Documentation
- Pandas Documentation
- Matplotlib Documentation