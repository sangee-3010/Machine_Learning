# 🤝 Bagging, Boosting & Stacked Ensemble Models

A Machine Learning laboratory project that implements and compares **Bagging**, **AdaBoost**, and **Stacked Ensemble** classifiers for breast cancer diagnosis using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

> **Course:** ICS1512 – Machine Learning Algorithms Laboratory  
> **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

## 📌 Overview

This project explores three ensemble learning strategies for binary classification. The models are trained on the **WDBC dataset**, optimized using **5-fold GridSearchCV**, and compared based on accuracy, precision, recall, F1-score, ROC-AUC, and generalization performance.

---

## 🎯 Objectives

- Implement Bagging and Boosting ensemble classifiers.
- Build a Stacked Ensemble using multiple base learners.
- Compare ensemble methods in terms of bias, variance, and stability.
- Perform hyperparameter tuning using 5-fold cross-validation.
- Evaluate which ensemble strategy generalizes best.

---

## 📂 Dataset

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

| Property | Value |
|----------|------|
| Samples | 569 |
| Features | 30 numerical features |
| Classes | Benign (0), Malignant (1) |
| Missing Values | 0 |
| Train/Test Split | 80% / 20% (455 / 114) |

The dataset contains measurements of breast cell nuclei extracted from digitized fine-needle aspirate images and is used to classify tumors as **Benign** or **Malignant**.

---

## 🛠️ Technologies Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 🤖 Ensemble Models

### Bagging
- Decision Tree base estimator
- Bootstrap aggregation
- Majority voting

### Boosting
- AdaBoost classifier
- Sequential error correction
- Weighted weak learners

### Stacked Ensemble
**Base Models**
- Support Vector Machine (SVM)
- Gaussian Naïve Bayes
- Decision Tree

**Meta Learner**
- Logistic Regression

---

## 📊 Exploratory Data Analysis

The notebook includes:

- Dataset overview
- Class distribution
- Missing value analysis
- Feature correlation heatmap
- Target distribution visualization

---

## 🏆 Final Performance

| Model | Accuracy | Precision | Recall | F1-score |
|------|:-------:|:---------:|:------:|:--------:|
| Bagging | 96.49% | 100.00% | 90.48% | 95.00% |
| **Boosting (AdaBoost)** | **98.25%** | **100.00%** | **95.24%** | **97.56%** |
| Stacked Ensemble | 92.98% | 97.22% | 83.33% | 89.74% |

**Best Performing Model:** AdaBoost Classifier

---

## ⚙️ Best Hyperparameters

### Bagging

| Parameter | Value |
|----------|------|
| Estimators | 100 |
| Max Samples | 1.0 |
| Max Features | 0.7 |

### Boosting

| Parameter | Value |
|----------|------|
| Estimators | 50 |
| Learning Rate | 1.0 |

### Stacked Ensemble

| Component | Model |
|-----------|------|
| Base Learners | SVM, Gaussian NB, Decision Tree |
| Meta Learner | Logistic Regression |

---

## 📈 Cross Validation

| Model | Average CV Accuracy |
|------|:-------------------:|
| Bagging | 96.92% |
| Boosting | 96.26% |
| Stacked Ensemble | 94.29% |

All models were evaluated using **5-fold Stratified Cross Validation**.

---

## 📊 ROC-AUC Comparison

| Model | ROC-AUC |
|------|:-------:|
| Bagging | 0.9907 |
| Boosting | 0.9841 |
| **Stacked Ensemble** | **0.9911** |

Although Stacking achieved the highest ROC-AUC, AdaBoost produced the best classification accuracy and F1-score on the test set.

---

## 📊 Visualizations Included

The notebook generates:

- Class Distribution
- Correlation Heatmap
- Bagging Confusion Matrix
- Boosting Confusion Matrix
- Stacked Ensemble Confusion Matrix
- ROC Curve Comparison
- Accuracy Comparison
- F1-score Comparison

---

## 🚀 Project Structure

```text
Ensemble-Learning-WDBC/
│
├── datasets/
│   └── wdbc.csv
│
├── notebooks/
│   └── Experiment_06.ipynb
│
├── outputs/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── comparison_plots/
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

Open **Experiment_06.ipynb** and run all cells.

---

## 📚 Learning Outcomes

- Implemented Bagging, Boosting, and Stacked Ensemble models.
- Understood variance reduction through bootstrap aggregation.
- Applied AdaBoost for sequential error-focused learning.
- Built a heterogeneous stacked ensemble using multiple classifiers.
- Compared ensemble methods using cross-validation and ROC-AUC.

---

## 👩‍💻 Author

**Sangeetha S K**

- GitHub: https://github.com/sangee-3010

---

## 📖 References

- UCI Machine Learning Repository – Wisconsin Diagnostic Breast Cancer Dataset
- Scikit-learn Ensemble Methods Documentation
- Scikit-learn BaggingClassifier Documentation
- NumPy Documentation
- Pandas Documentation
- Matplotlib Documentation
```