# 📩 Binary Classification using Logistic Regression & SVM

A Machine Learning laboratory project that implements and compares **Logistic Regression** and **Support Vector Machine (SVM)** models for spam email detection using the **Spambase Dataset**.

> **Course:** ICS1512 – Machine Learning Algorithms Laboratory  
> **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

## 📌 Overview

This project focuses on binary classification of emails into **Spam** and **Ham** using linear and kernel-based machine learning models. The experiment includes exploratory data analysis, feature standardization, hyperparameter tuning with **GridSearchCV**, kernel comparison, and model evaluation using cross-validation.

---

## 🎯 Objectives

- Implement Logistic Regression for spam email classification.
- Train Support Vector Machine models using multiple kernels.
- Compare Linear, Polynomial, RBF, and Sigmoid kernels.
- Optimize model parameters using GridSearchCV.
- Evaluate models using Accuracy, Precision, Recall, and F1-score.

---

## 📂 Dataset

**Dataset:** Spambase (Kaggle / UCI Repository)

| Property | Value |
|----------|------|
| Samples | 4,601 |
| Features | 57 |
| Classes | Spam (1), Ham (0) |
| Missing Values | None |
| Train/Test Split | 80% / 20% |

The dataset contains numerical features representing word frequencies, character frequencies, and capital letter statistics extracted from email messages.

---

## 🛠️ Technologies Used

- Python 3.12
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook (Google Colab)

---

## 🤖 Machine Learning Models

### Logistic Regression
- L1 & L2 Regularization
- Hyperparameter tuning using GridSearchCV
- Standardized input features

### Support Vector Machine
- Linear Kernel
- Polynomial Kernel
- RBF Kernel
- Sigmoid Kernel
- GridSearchCV optimization

---

## 📊 Exploratory Data Analysis

The notebook includes:

- Class distribution analysis
- Correlation heatmap
- Feature histograms
- Boxplots for feature distributions

These visualizations help understand class imbalance, feature relationships, and data sparsity before model training.

---

## 🏆 Final Results

### Tuned Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|------|:-------:|:---------:|:------:|:--------:|
| Logistic Regression | 91.75% | 91.98% | 88.21% | 90.05% |
| **SVM (RBF Kernel)** | **93.49%** | **95.08%** | **89.23%** | **92.06%** |

### SVM Kernel Comparison

| Kernel | Accuracy | F1-score |
|---------|:--------:|:--------:|
| Linear | 92.62% | 91.05% |
| Polynomial | 76.66% | 63.37% |
| **RBF** | **93.49%** | **92.06%** |
| Sigmoid | 87.62% | 85.00% |

---

## ⚙️ Best Hyperparameters

### Logistic Regression

| Parameter | Value |
|----------|------|
| Penalty | L1 |
| C | 10 |
| Solver | liblinear |

### Support Vector Machine

| Parameter | Value |
|----------|------|
| Kernel | RBF |
| C | 1 |
| Gamma | scale |
| Degree | 2 |

---

## 📈 Cross Validation

| Model | Average 5-Fold Accuracy |
|------|:------------------------:|
| Logistic Regression | 92.74% |
| **SVM (RBF)** | **93.07%** |

The RBF kernel consistently achieved the highest validation accuracy across folds.

---

## ⚡ Training Time Comparison

| Model | Training Time |
|------|--------------:|
| Logistic Regression | 528.18 s |
| SVM (Grid Search) | 796.10 s |

**Observation:** Logistic Regression is computationally cheaper, while SVM provides superior predictive performance.

---

## 📊 Visualizations Included

The notebook generates:

- Class Distribution
- Correlation Heatmap
- Feature Histograms
- Feature Boxplots
- Logistic Regression Confusion Matrix
- Tuned Logistic Regression Confusion Matrix
- Tuned SVM Confusion Matrix

---

## 🚀 Project Structure

```text
Binary-Classification-Logistic-SVM/
│
├── datasets/
│   └── spambase.csv
│
├── notebooks/
│   └── Experiment_04.ipynb
│
├── outputs/
│   ├── confusion_matrices/
│   ├── heatmaps/
│   └── plots/
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

Open **Experiment_04.ipynb** and run all cells.

---

## 📚 Learning Outcomes

- Applied Logistic Regression for binary classification.
- Implemented kernel-based Support Vector Machines.
- Compared multiple SVM kernels and optimized hyperparameters.
- Used feature standardization to improve model performance.
- Evaluated models using cross-validation and classification metrics.

---

## 👩‍💻 Author

**Sangeetha S K**

- GitHub: https://github.com/sangee-3010

---

## 📖 References

- UCI Machine Learning Repository – Spambase Dataset
- Scikit-learn Documentation
- NumPy Documentation
- Pandas Documentation
- Matplotlib Documentation