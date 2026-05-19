# ArXiv-Machine-Learning-Model
Built a multi-class classifier for 39 arXiv CS categories using TF-IDF and classical ML (NLP). Iterated from a scratch logistic regression to a soft-voting ensemble (ComplementNB, Logistic Regression, LinearSVC, SGD), reaching 68.7% validation accuracy, on private leaderboard on kaggle.

# arXiv CS Paper Classifier
Multi-class text classification of arXiv papers across 39 Computer Science subcategories using TF-IDF features and classical ML methods.

Built as part of **50.007 Machine Learning** at SUTD. 
https://www.kaggle.com/competitions/50-007-machine-learning-spring-2026

---

## Problem Statement
Given the abstract of an arXiv paper, predict its primary CS subcategory (e.g. cs.AI, cs.CV, cs.LG). The dataset contains up to 40 distinct categories, making this a challenging multi-class classification task.

---

## Project Structure

```
├── LogReg_Prediction.csv # Task 1 Kaggle submission
├── FinalModel_LinearSVM_Task3 #Task 3 Kaggle submission
├── Task1.ipynb       # Logistic regression from scratch (binary + OvR)
├── Task2.ipynb # PCA vs TF-IDF feature selection comparison
├── Task3.ipynb     # Models for task 3
└── README.md
```

---

## Tasks

### Task 1: Logistic Regression from Scratch
Implemented a binary logistic regression classifier from scratch using NumPy, with the following core components:
- `sigmoid(z)` — maps real-valued input to (0, 1)
- `loss(y, y_hat)` — log loss (binary cross-entropy)
- `gradients(X, y, y_hat)` — gradient descent updates for weights and bias
- `train(X, y, bs, epochs, lr)` — mini-batch training loop
- `predict(X)` — binary class prediction

Extended to multi-class via a One-vs-Rest (OvR) architecture, training one binary classifier per class and selecting the label with the highest predicted probability using `np.argmax()`.

### Task 2: Dimensionality Reduction
Compared two approaches for handling the 5000-feature TF-IDF space:
- **Feature Selection** — restricting TF-IDF vocabulary to top N features
- **Dimensionality Reduction** — applying TruncatedSVD (sparse PCA) to project down to N components

Evaluated at N = 2000, 1000, 500, and 100 using KNN (k=2) and reported Macro F1 scores on the Kaggle test set.

### Task 3: Ensemble Classifier
Explored multiple classical ML models: 
- Complement Naive Bayes
- Logistic Regression (sklearn)
- LinearSVC
- SGDClassifier
- Built soft ensemble combining all models experimented
  
Linear SVM model with n = 80000 features, with orchestrated TF-IDF, achieved approximately **68.7% validation accuracy** on the 39-class task, on the private dataset on kaggle.

---

## Setup

```bash
pip install numpy scikit-learn pandas
```

All experiments were run in Google Colab with a standard CPU runtime.

---

## Results Summary

| Model | Validation Accuracy |
|---|---|
| Logistic Regression (scratch, OvR) | ~55% | Kaggle: 0.506
| KNN (2000 TF-IDF features) | ~45% |
| Linear SVC | ~75% | Kaggle: 0.697

---

## Tech Stack
Python, NumPy, scikit-learn, pandas
