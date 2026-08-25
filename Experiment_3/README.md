# 📧 Email Spam/Ham Classification using Naïve Bayes & KNN

A Machine Learning laboratory project that builds and compares **Gaussian, Multinomial, and Bernoulli Naïve Bayes** classifiers with **K-Nearest Neighbours (KNN)** for email spam detection using the **Spambase Dataset**.

> **Course:** ICS1512 – Machine Learning Algorithms Laboratory  
> **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

## 📌 Overview

This project investigates how different supervised learning algorithms perform on the **Spambase** email dataset. Besides comparing Naïve Bayes variants, it explores KNN hyperparameter tuning using **GridSearchCV** and **RandomizedSearchCV**, neighbour search strategies (**KDTree vs BallTree**), and evaluates models using **5-fold Cross Validation**.

---

## 🎯 Objectives

- Build spam classifiers using Gaussian, Multinomial & Bernoulli Naïve Bayes.
- Implement K-Nearest Neighbours (KNN) for spam classification.
- Study the effect of different **k** values in KNN.
- Compare Grid Search and Randomized Search for hyperparameter tuning.
- Evaluate KDTree and BallTree neighbour search methods.
- Compare models using accuracy, precision, recall, F1-score and ROC-AUC.

---

## 📂 Dataset

**Dataset:** Spambase Dataset

| Property | Value |
|----------|------|
| Samples | 4,601 |
| Features | 57 |
| Classes | Spam (1), Ham (0) |
| Train/Test Split | 80% / 20% |
| Missing Values | None |

The dataset consists of numerical features representing word frequencies, character frequencies, and capital letter statistics extracted from emails.

---

## 🛠️ Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 🤖 Machine Learning Models

### Naïve Bayes
- Gaussian Naïve Bayes
- Multinomial Naïve Bayes
- Bernoulli Naïve Bayes

### K-Nearest Neighbours
- Manual K-value comparison
- GridSearchCV optimization
- RandomizedSearchCV optimization
- KDTree & BallTree comparison

---

## 📊 Final Results

### Naïve Bayes Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|:-------:|:---------:|:------:|:---:|:------:|
| Gaussian NB | 83.39% | 71.78% | **95.32%** | 81.89% | 0.9449 |
| Multinomial NB | 77.63% | 71.99% | 70.80% | 71.39% | 0.8248 |
| **Bernoulli NB** | **90.01%** | **85.94%** | 89.26% | **87.57%** | **0.9510** |

### KNN (Manual K Values)

| k | Accuracy |
|---|----------|
| 1 | **80.24%** |
| 3 | 78.94% |
| 5 | 79.91% |
| 7 | 79.26% |
| 9 | 78.94% |
| 11 | 77.96% |

### Best Hyperparameters

| Method | Best Parameters | CV Accuracy |
|---------|----------------|------------|
| GridSearchCV | k=9, Manhattan, Distance | **85.95%** |
| RandomizedSearchCV | k=17, Manhattan, BallTree | 85.63% |

---

## ⚡ Performance Comparison

| Algorithm | Training Time | Prediction Time |
|-----------|--------------:|---------------:|
| Gaussian NB | 0.008 s | 0.004 s |
| Multinomial NB | 0.035 s | 0.006 s |
| Bernoulli NB | 0.013 s | **0.004 s** |
| Best KNN | **0.003 s** | 0.064 s |

**Observation:** Naïve Bayes predicts significantly faster, making it more suitable for real-time spam filtering.

---

## 📈 Visualizations Included

The notebook generates:

- Confusion Matrix
- ROC Curve
- Precision–Recall Curve
- KNN Performance for different **k**
- GridSearchCV Heatmap
- RandomizedSearchCV Score Distribution
- 5-Fold Cross Validation Comparison
- KDTree vs BallTree Accuracy Comparison

---

## 🚀 Project Structure

```text
Email-Spam-Classification/
│
├── datasets/
│   └── spambase_csv.csv
│
├── notebooks/
│   └── Experiment_02.ipynb
│
├── outputs/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── precision_recall_curves/
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

### Launch Jupyter

```bash
jupyter notebook
```

Open **Experiment_02.ipynb** and run all cells.

---

## 📚 Learning Outcomes

- Implemented three variants of Naïve Bayes for binary classification.
- Understood the influence of neighbourhood size in KNN.
- Applied GridSearchCV and RandomizedSearchCV for model optimization.
- Compared exact neighbour search methods (KDTree vs BallTree).
- Evaluated classifiers using cross-validation and multiple performance metrics.

---

## 👩‍💻 Author

**Sangeetha S K**

- GitHub: [@sangee-3010](https://github.com/sangee-3010)

---

## 📖 References

- UCI Machine Learning Repository – Spambase Dataset
- Scikit-learn Documentation
- NumPy Documentation
- Pandas Documentation
- Matplotlib Documentation