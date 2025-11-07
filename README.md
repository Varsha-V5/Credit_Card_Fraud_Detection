# Credit_Card_Fraud_Detection
# Credit Card Fraud Detection Project

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning models. The dataset used is the popular **Kaggle Credit Card Fraud Detection dataset**, which is highly imbalanced. To address this, oversampling techniques such as **SMOTE** were applied. Multiple ML models were trained and evaluated to determine the best-performing classifier.

---

## ✅ Objectives

* Load and analyze the credit card dataset
* Perform sanity checks and exploratory analysis
* Select top features using **SelectKBest (ANOVA F-test)**
* Handle class imbalance using **SMOTE**
* Train classification models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * XGBoost (tuned)
* Evaluate model performance using precision, recall, F1 score, and confusion matrix
* Save the final model as a `.pkl` file for deployment

---

## 📂 Dataset

**Dataset used:** Kaggle Credit Card Fraud Detection dataset

* Contains anonymized features V1–V28, `Time`, `Amount`, and target `Class`
* `Class = 1` indicates fraud
* Highly imbalanced (fraudulent transactions < 1%)

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Pickle

---

## 🔍 Steps Performed

### **1. Import Required Libraries**

All necessary ML, visualization, and preprocessing libraries are imported.

### **2. Load Dataset**

```python
crd_data = pd.read_csv("creditcard.csv")
```

Dataset is loaded into a pandas DataFrame.

### **3. Sanity Check**

* `describe()` for statistical summary
* `info()` for datatypes
* Check for null values and unique counts

### **4. Feature Selection**

Used **SelectKBest (ANOVA F-test)** to pick top 15 features.
Final selected features used for modeling:

```
V17, V14, V12, V10, V16, V3, V7, V11,
V4, V18, V1, V9, V5, V2
```

### **5. Train–Test Split**

Data is split using:

```python
test_size = 0.25, random_state = 42
```

### **6. Handle Class Imbalance**

Applied **SMOTE (Synthetic Minority Oversampling Technique)** on training data.

### **7. Model Training and Evaluation**

Multiple algorithms were trained:

#### ✅ Logistic Regression

Evaluated using accuracy, precision, recall, F1 score, and confusion matrix.

#### ✅ Decision Tree Classifier

Evaluated similarly to LR.

#### ✅ Random Forest (with class weights balanced)

Achieved improved performance.

#### ✅ XGBoost (Tuned)

Hyperparameters used:

```
max_depth=5
learning_rate=0.1
n_estimators=300
subsample=0.8
colsample_bytree=1.0
gamma=1
scale_pos_weight=50
```

Best performing model based on F1 and recall.

### **8. Classification Metrics Calculated**

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

### **9. Threshold Tuning**

Decision threshold manually adjusted (e.g., 0.20, 0.25, 0.30, 0.50) to improve recall.

### **10. Save Final Model**

Final XGBoost model exported as:

```python
xgb_fraud_model.pkl
```

---

## 📊 Results Summary

* Traditional models performed decently but struggled with recall.
* **XGBoost provided the best fraud detection performance**, especially after adjusting threshold.
* Threshold tuning further improved fraud catching capability.

---

## 🚀 How to Run This Project

### **1. Install Dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### **2. Place Dataset File**

Download dataset from Kaggle and save it as:

```
creditcard.csv
```

### **3. Run the Script**

```bash
python fraud_detection.py
```

### **4. Saved Model**

The trained model will be stored as:

```
xgb_fraud_model.pkl
```

---

## ✅ Conclusion

This project successfully demonstrates an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. Using advanced techniques like **feature selection**, **SMOTE**, **threshold tuning**, and **XGBoost**, the model achieves strong performance in identifying rare fraudulent cases.