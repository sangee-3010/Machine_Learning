# Experiment 7: Dimensionality Reduction and Model Evaluation with and without PCA

## 📌 Overview

This laboratory project studies the effect of **Principal Component Analysis (PCA)** on the performance of machine learning classification algorithms for breast cancer diagnosis.

The models are trained and evaluated in two settings:

1. **Without PCA** – using the original standardized feature space.
2. **With PCA** – using a reduced feature space while retaining approximately 95% of the variance.

Hyperparameter tuning is performed using **GridSearchCV with 5-fold stratified cross-validation**. The models are compared using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation stability.

> **Course:** ICS1512 – Machine Learning Algorithms Laboratory
> **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

## 🎯 Objectives

* Understand dimensionality reduction using PCA.
* Compare machine learning models with and without PCA.
* Implement and evaluate ensemble models such as AdaBoost and Stacking.
* Perform hyperparameter tuning using GridSearchCV.
* Evaluate models using 5-fold stratified cross-validation.
* Compare models using accuracy, precision, recall, F1-score, ROC-AUC, and stability.
* Analyze whether PCA improves model generalization and performance.

---

## 📊 Dataset

### WDBC – Wisconsin Diagnostic Breast Cancer Dataset

The dataset used is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset from the **UCI Machine Learning Repository**.

**Dataset file:**

```text
wdbc.data
```

### Dataset Characteristics

| Property          |                 Value |
| ----------------- | --------------------: |
| Samples           |                   569 |
| Original Features | 30 numerical features |
| Target Classes    |                     2 |
| Classes           |     Benign, Malignant |
| Missing Values    |                  None |
| Task              | Binary Classification |
| Train/Test Split  |             80% / 20% |
| Training Samples  |                   455 |
| Testing Samples   |                   114 |

The dataset contains measurements of breast cell nuclei extracted from digitized fine-needle aspirate images. The objective is to classify tumors as **Benign** or **Malignant**.

The diagnosis labels are encoded numerically as:

```text
Benign    → 0
Malignant → 1
```

---

## ⚙️ Preprocessing

The following preprocessing steps were performed:

1. The `wdbc.data` file was loaded into a Pandas DataFrame.
2. The ID column was removed because it does not contain predictive information.
3. The diagnosis column was converted into numerical target labels.
4. Missing values were checked.
5. The dataset was divided into training and testing sets using an **80:20 stratified split**.
6. Features were standardized using `StandardScaler`.
7. PCA was applied after standardization.
8. The same preprocessing logic was used consistently for model comparison.

### Train-Test Split

```text
Training Set : 80% (455 samples)
Testing Set  : 20% (114 samples)
```

Stratification was used to preserve the class distribution in both subsets.

---

## 🔬 Exploratory Data Analysis

The notebook includes the following exploratory analysis:

* Dataset overview
* Dataset shape and feature information
* Class distribution
* Missing-value analysis
* Target-label distribution
* Feature correlation heatmap
* Basic statistical summaries
* Visualization of class balance

---

# 🔬 Principal Component Analysis

PCA was applied to the standardized training data.

The number of principal components was selected using a **95% cumulative explained variance target**. The transformed feature space was then used to train and evaluate the models under the **With-PCA** setting.

### PCA Analysis Includes

* Individual explained variance ratio
* Cumulative explained variance
* PCA variance plot
* Number of components required to retain approximately 95% variance
* Comparison of original and reduced feature dimensions

PCA was evaluated separately from the original feature space to determine whether dimensionality reduction improves classification performance.

---

# 🤖 Machine Learning Models

The following 10 models were evaluated:

1. Support Vector Machine (SVM)
2. Gaussian Naive Bayes
3. K-Nearest Neighbors (KNN)
4. Logistic Regression
5. Decision Tree
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. XGBoost
10. Stacking Classifier

The model collection includes linear, probabilistic, distance-based, tree-based, boosting, and ensemble learning methods.

---

## 🤝 Ensemble Models

### Bagging

Bagging uses bootstrap aggregation to reduce variance by training multiple decision-tree estimators on different bootstrap samples.

Characteristics:

* Decision Tree base estimator
* Bootstrap aggregation
* Majority voting
* Variance reduction

### Boosting

AdaBoost trains weak learners sequentially. Each new learner focuses more strongly on samples that were incorrectly classified by previous learners.

Characteristics:

* Sequential error correction
* Weighted weak learners
* Bias reduction
* Improved classification boundaries

### Stacked Ensemble

The Stacking Classifier combines predictions from multiple heterogeneous base learners and passes them to a meta-learner.

#### Base Learners

* Support Vector Machine
* Gaussian Naive Bayes
* Decision Tree

#### Meta Learner

* Logistic Regression

The Stacking Classifier was evaluated both with and without PCA.

---

# 🔧 Hyperparameter Tuning

GridSearchCV was used to identify suitable hyperparameters for each model.

The models were tuned separately for:

```text
No-PCA
With-PCA
```

The optimization metric used for hyperparameter selection was:

```text
F1-score
```

### Example Hyperparameters

#### SVM

Parameters explored:

```text
C
kernel
gamma
```

#### KNN

Parameters explored:

```text
n_neighbors
weights
metric
```

#### Random Forest

Parameters explored:

```text
n_estimators
max_depth
min_samples_split
```

#### Bagging

Best configuration:

| Parameter            | Value |
| -------------------- | ----: |
| Number of Estimators |   100 |
| Max Samples          |   1.0 |
| Max Features         |   0.7 |

#### AdaBoost

Best configuration:

| Parameter            | Value |
| -------------------- | ----: |
| Number of Estimators |    50 |
| Learning Rate        |   1.0 |

#### Stacking

| Component     | Model                                    |
| ------------- | ---------------------------------------- |
| Base Learners | SVM, Gaussian Naive Bayes, Decision Tree |
| Meta Learner  | Logistic Regression                      |

Similar parameter grids were defined for the remaining models.

---

# 🔁 Cross-Validation

A **5-fold Stratified Cross-Validation** strategy was used for model evaluation and hyperparameter tuning.

```text
Number of folds = 5
Shuffle         = True
Random State    = 42
```

The following metrics were recorded:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Standard deviation of accuracy

The performance of each fold was recorded for both:

```text
No-PCA
With-PCA
```

### Cross-Validation Results for Ensemble Models

| Model            | Average CV Accuracy |
| ---------------- | ------------------: |
| Bagging          |              96.92% |
| Boosting         |              96.26% |
| Stacked Ensemble |              94.29% |

All models were evaluated using **5-fold stratified cross-validation**.

---

# 📈 Evaluation

The models were evaluated using multiple performance measures.

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Visual Evaluation

The notebook generates:

* PCA cumulative variance plot
* Confusion matrices
* ROC curves
* Precision-Recall curves
* Accuracy comparison graph
* F1-score comparison graph
* Class distribution plot
* Feature correlation heatmap

## Stability Analysis

The standard deviation of accuracy across the five cross-validation folds was used to compare model stability and generalization performance.

---

# 📊 Results and Observations

The experiment showed that **PCA did not improve accuracy for the majority of models**.

Similarly, **PCA did not improve F1-score for the majority of models**. However, PCA improved stability for more models than it reduced, indicating that dimensionality reduction can sometimes provide more consistent performance across cross-validation folds.

The effect of PCA was model-dependent:

* Some linear and distance-based models benefited from the reduced feature space.
* Tree-based models often performed well using the original features.
* Ensemble models responded differently depending on their base learners.
* Stacking achieved strong performance in some PCA configurations because it combines multiple learning strategies.

---

## 🏆 Best Results

| Setting     | Best Model |   Accuracy |
| ----------- | ---------- | ---------: |
| Without PCA | SVM        | **97.58%** |
| With PCA    | Stacking   | **98.02%** |

The best accuracy obtained with PCA was approximately:

```text
98.02%
```

compared with:

```text
97.58%
```

without PCA.

### PCA Effect

* **Largest Accuracy Improvement:** Decision Tree
* **Largest F1-score Improvement:** Decision Tree
* **Largest F1-score Decrease:** Gaussian Naive Bayes
* **Best Accuracy without PCA:** SVM – 97.58%
* **Best Accuracy with PCA:** Stacking – 98.02%

---

## 📊 Ensemble Model Test Performance

The ensemble models were also compared using test-set metrics.

| Model                   |   Accuracy |   Precision |     Recall |   F1-score |
| ----------------------- | ---------: | ----------: | ---------: | ---------: |
| Bagging                 |     96.49% |     100.00% |     90.48% |     95.00% |
| **Boosting (AdaBoost)** | **98.25%** | **100.00%** | **95.24%** | **97.56%** |
| Stacked Ensemble        |     92.98% |      97.22% |     83.33% |     89.74% |

### Best Performing Ensemble Model

**AdaBoost Classifier**

AdaBoost achieved the highest test accuracy and F1-score among the evaluated ensemble models.

---

## 📊 ROC-AUC Comparison

| Model                |    ROC-AUC |
| -------------------- | ---------: |
| Bagging              |     0.9907 |
| Boosting             |     0.9841 |
| **Stacked Ensemble** | **0.9911** |

Although the Stacked Ensemble achieved the highest ROC-AUC, AdaBoost produced the best classification accuracy and F1-score on the test set.

This shows that different evaluation metrics can identify different strengths:

* **AdaBoost:** Best test accuracy and F1-score
* **Stacking:** Best ROC-AUC
* **Bagging:** Strong cross-validation accuracy and ROC-AUC

---

# 💡 Interpretation

The results demonstrate that the effectiveness of PCA is **model-dependent**.

PCA can reduce the dimensionality of the feature space and remove redundant information. However, dimensionality reduction does not necessarily result in higher predictive performance for every classifier.

Linear and distance-based algorithms can sometimes benefit from a compact representation of the data. On the other hand, tree-based algorithms such as Random Forest, Gradient Boosting, XGBoost, and Bagging can work effectively with the original feature representation and may not always benefit from PCA.

AdaBoost performed best among the ensemble models because its sequential learning process focuses on correcting previous classification errors. This resulted in strong accuracy, precision, recall, and F1-score.

The Stacking Classifier achieved the highest ROC-AUC among the ensemble models, demonstrating that combining heterogeneous base learners can produce a strong ranking of positive and negative instances. However, its test accuracy and F1-score were lower than those of AdaBoost in this experiment.

Overall, the results show that no single model is optimal for every metric. Model selection should depend on the evaluation objective, class distribution, and desired balance between precision, recall, accuracy, and generalization.

---

# 📁 Project Structure

```text
Experiment-7/
│
├── wdbc.data
├── Experiment7.ipynb
├── README.md
│
└── Experiment7_Output/
    │
    ├── Images/
    │   ├── class_distribution.png
    │   ├── correlation_heatmap.png
    │   ├── pca_variance_plot.png
    │   ├── SVM_Confusion_No_PCA.png
    │   ├── SVM_Confusion_With_PCA.png
    │   ├── Logistic_Regression_Confusion_No_PCA.png
    │   ├── Logistic_Regression_Confusion_With_PCA.png
    │   ├── Random_Forest_Confusion_No_PCA.png
    │   ├── Random_Forest_Confusion_With_PCA.png
    │   ├── XGBoost_Confusion_No_PCA.png
    │   ├── XGBoost_Confusion_With_PCA.png
    │   ├── Bagging_Confusion_Matrix.png
    │   ├── Boosting_Confusion_Matrix.png
    │   ├── Stacking_Confusion_Matrix.png
    │   ├── ROC_Curves_PCA_vs_No_PCA.png
    │   ├── Precision_Recall_Curves_PCA_vs_No_PCA.png
    │   ├── ROC_Curve_Ensemble_Comparison.png
    │   ├── Accuracy_Comparison_All_10_Models.png
    │   ├── F1_Score_Comparison_All_10_Models.png
    │   └── Ensemble_Accuracy_Comparison.png
    │
    └── CSV_Files/
        ├── Table1_PCA_Variance.csv
        ├── Table5_Cross_Validation_Accuracy.csv
        ├── Table5_Fold_Wise_Results.csv
        ├── Model_Average_Comparison.csv
        ├── Hyperparameter_Tuning_Results.csv
        ├── Test_Set_Results.csv
        ├── F1_Cross_Validation_Results.csv
        ├── All_Models_5Fold_CV_Results.csv
        ├── Stacking_Test_Results.csv
        ├── Ensemble_Test_Results.csv
        ├── Ensemble_ROC_AUC_Results.csv
        ├── FINAL_10_MODEL_PCA_COMPARISON.csv
        └── Stability_Analysis.csv
```

---

# 🛠️ Technologies Used

* Python 3.x
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

---

# 📦 Installation

Install the required Python libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

# ▶️ How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/sangee-3010/Machine_Learning.git
```

## 2. Navigate to the Project Directory

```bash
cd Machine_Learning
```

## 3. Start Jupyter Notebook

```bash
jupyter notebook
```

## 4. Open the Notebook

```text
Experiment7.ipynb
```

## 5. Ensure the Dataset Is Available

Make sure the following file is available in the dataset path used by the notebook:

```text
wdbc.data
```

## 6. Run All Cells Sequentially

The notebook performs the following workflow:

```text
Data Loading
      ↓
Exploratory Data Analysis
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Feature Standardization
      ↓
PCA Analysis
      ↓
Hyperparameter Tuning
      ↓
5-Fold Stratified Cross-Validation
      ↓
Test Evaluation
      ↓
Confusion Matrices
      ↓
ROC Curves
      ↓
Precision-Recall Curves
      ↓
Model Comparison
      ↓
Stability Analysis
      ↓
Final Conclusion
```

---

# 📌 Key Conclusion

This experiment demonstrates that **PCA is not universally beneficial**.

Although PCA reduced the dimensionality of the WDBC dataset, it did not improve accuracy or F1-score for most of the evaluated models. However, it improved stability for more models than it reduced.

The best accuracy without PCA was achieved by **SVM with 97.58%**, while the best accuracy with PCA was achieved by **Stacking with 98.02%**.

Among the ensemble models, **AdaBoost achieved the best test performance**, with:

* Accuracy: **98.25%**
* Precision: **100.00%**
* Recall: **95.24%**
* F1-score: **97.56%**

The Stacked Ensemble achieved the highest ensemble ROC-AUC of **0.9911**, while Bagging achieved a ROC-AUC of **0.9907** and AdaBoost achieved **0.9841**.

Therefore, PCA and ensemble methods should be selected based on experimental evidence, the characteristics of the dataset, and the evaluation metric that is most important for the application. In medical classification tasks, recall and F1-score may be especially important because failing to identify a malignant tumor can have serious consequences.

---

## 📚 Learning Outcomes

* Implemented PCA for dimensionality reduction.
* Compared classification models with and without PCA.
* Implemented Bagging, Boosting, and Stacked Ensemble models.
* Understood variance reduction through bootstrap aggregation.
* Applied AdaBoost for sequential error-focused learning.
* Built a heterogeneous stacked ensemble using multiple classifiers.
* Performed hyperparameter tuning using GridSearchCV.
* Applied 5-fold stratified cross-validation.
* Compared models using accuracy, precision, recall, F1-score, ROC-AUC, and stability.
* Analyzed the effect of PCA on model performance and generalization.

---

## 👩‍💻 Author

**Sangeetha S K**

Integrated M.Tech – Computer Science and Engineering

* GitHub: https://github.com/sangee-3010

**Experiment 7 – Dimensionality Reduction, Ensemble Learning, and Model Evaluation**

---

## 📖 References

* UCI Machine Learning Repository – Wisconsin Diagnostic Breast Cancer Dataset
* Scikit-learn Ensemble Methods Documentation
* Scikit-learn `BaggingClassifier` Documentation
* Scikit-learn `AdaBoostClassifier` Documentation
* Scikit-learn `StackingClassifier` Documentation
* Scikit-learn PCA Documentation
* NumPy Documentation
* Pandas Documentation
* Matplotlib Documentation
* Seaborn Documentation
* XGBoost Documentation
