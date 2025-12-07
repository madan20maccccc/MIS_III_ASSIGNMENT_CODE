# 🧮 MIS III Assignment – Diabetes Classification Using SVM & SVD  
### *Mathematics for Intelligent Systems – Model Building & Optimization*

This repository contains the Google Colab notebook **MIS_assignment.ipynb**, implementing multiple machine-learning models—including SVM, SVD+SVM, Decision Tree, and Random Forest—for diabetes prediction using the Pima Indians Diabetes Dataset.

---

## 📌 Objectives
1. Load and understand the dataset.
2. Identify and replace medically impossible zero values using median imputation.
3. Split data into training, validation, and test sets (60–20–20).
4. Train a **Baseline SVM** model.
5. Build a **Truncated SVD + SVM** dimensionality-reduction pipeline.
6. Optimize the SVD+SVM pipeline using **RandomizedSearchCV**.
7. Train **Decision Tree** and **Random Forest** models.
8. Compare all models using evaluation metrics and visualizations.

---

## 🗂️ Notebook Workflow

### 🔹 Step 1 — Load the Dataset
- Read CSV file  
- Display head(), describe(), and column names  
- Verify data structure and ranges  

### 🔹 Step 2 — Data Cleaning
Medical features containing invalid zero values:
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  

Cleaning steps:
- Replace zeros → NaN  
- Fill NaN with **median** of respective column  

### 🔹 Step 3 — Dataset Splitting
- 60% → Training  
- 20% → Validation  
- 20% → Testing  
- Stratified splitting to maintain class balance  

---

## 💻 Models Implemented

### 1️⃣ Baseline SVM (No SVD)
- StandardScaler → SVM (RBF kernel)  
- Evaluated using accuracy, F1, precision, recall  


### 2️⃣ SVD + SVM Pipeline
Pipeline:
- Dimensionality reduction  
- Improved recall and F1 over baseline  

### 3️⃣ Optimized SVD + SVM
Optimized parameters via RandomizedSearchCV:
- SVD components  
- Kernel  
- C  
- Gamma  

### 4️⃣ Decision Tree Classifier
- Simple and interpretable baseline model  

### 5️⃣ Random Forest Classifier
- Ensemble-based improvement over Decision Tree  

---

## 📊 Evaluation Metrics
Each model is evaluated using:
- Accuracy  
- F1-score  
- Precision  
- Recall  
- Confusion Matrix  
- ROC-AUC  

---

## 📈 Model Comparison Summary

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Baseline SVM | 0.7468 | 0.5806 | 0.6750 | 0.5094 |
| SVD + SVM (3 Components) | 0.7403 | **0.6154** | 0.6275 | **0.6038** |
| Optimized SVD + SVM | 0.7338 | 0.5591 | **0.6500** | 0.4906 |
| Random Forest | 0.7078 | 0.5714 | 0.5769 | 0.5660 |
| Decision Tree | 0.6883 | 0.5472 | 0.5472 | 0.5472 |

### 🔍 Key Insights
- **Best Recall:** SVD + SVM (3 components)  
- **Best Precision & AUC:** Optimized SVD + SVM  
- **Best overall balance:** SVD + SVM (3 components)  
- **Most stable tree model:** Random Forest  

---

## 🚀 Technologies Used
- Python 3  
- Pandas  
- NumPy  
- scikit-learn  
- Matplotlib  
- Seaborn  
- Google Colab  

---

## ▶️ How to Run
1. Open the `.ipynb` notebook in **Google Colab** or Jupyter Notebook.  
2. Upload the dataset or set correct dataset path.  
3. Run all cells sequentially.  
4. View outputs, evaluations, and visualizations.  

---

## 👤 Authors
**Madan M** (DL.AI.U4AID24021)  
**Anna Clara Mathew** (DL.AI.U4AID24005)  
**Lakxmi Chinmaya Aditya Katharu** (DL.AI.U4AID24018)  

B.Tech AI & DS (Medical Engineering)  
Amrita Vishwa Vidyapeetham – Faridabad Campus  

---

## 📄 License
This project is for academic and research purposes only.
