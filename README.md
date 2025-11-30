# 📌 Credit Card Fraud Detection (Machine Learning Project)

This project builds a machine learning system to detect fraudulent credit card transactions using the popular Kaggle Credit Card Fraud Dataset.
The data is highly imbalanced, and the goal is to create a model that maximizes fraud recall while keeping false alarms controlled.


## 🚀 Project Overview

Credit card fraud is a major challenge for banks and financial institutions.
Even a small number of fraudulent transactions leads to large financial losses.
This project aims to:
- Detect fraudulent transactions (Class 1)
- Minimize financial loss
- Reduce false alerts to maintain customer experience
- Build a complete ML pipeline from EDA → Modeling → Evaluation → Threshold Tuning → Deployment


## 📂 Dataset
Source: Kaggle Credit Card Fraud Dataset
- Total samples: 284,807
- Features: 30 PCA-transformed numerical features + Time + Amount
- Target:
   - 0 → Legitimate transaction
   - 1 → Fraudulent transaction

The dataset is highly imbalanced:
- Fraud cases ≈ 0.17%
- Legitimate ≈ 99.83%

## 🧠 Project Workflow
✔ 1. Data Cleaning
- Removed null values
- Checked duplicates
- Handled outliers in Amount
  
✔ 2. Exploratory Data Analysis (EDA)
Visualizations include:
  - Class imbalance
  - Amount distribution (log scale)
  - Time distribution
  - Scatter plots (Time vs Amount)
  - Correlation heatmap
  - Fraud patterns analysis
    
✔ 3. Handling Imbalance
- Implemented SMOTE (Synthetic Minority Oversampling Technique) only on training data after train_test_split to prevent data leakage.

✔ 4. Feature Scaling
- Applied StandardScaler to Amount and Time

✔ 5. Train–Test Split
- 80% training, 20% testing
- stratify=y used to preserve class proportions

## 🤖 Models Used
Four ML models were implemented and evaluated:
- Model	                Precision (Fraud)	   Recall (Fraud)	    F1 Score	  ROC-AUC
- Logistic Regression	      0.14	                  0.85	          0.24	       0.96
- SVM (RBF Kernel)	         0.37	                  0.76	          0.50	       0.95
- Random Forest	            0.87	                  0.76	          0.82	       0.97
- XGBoost	                  0.74	                  0.80	          0.76	       0.97

## 🏆 Best Model: XGBoost + Threshold Tuning
- While Random Forest had the highest precision, XGBoost achieved the best fraud recall, which is the most important metric for banks.
- Then threshold tuning was applied to improve recall/precision balance.
- The default threshold (0.50) was reduced to find the optimal trade-off.

## 📈 Evaluation Metrics
For each model, the following were computed:
  - Confusion Matrix
  - Classification Report
  - ROC Curve
  - ROC-AUC
  - Precision-Recall Curve
  - PR-AUC
These metrics are more meaningful than accuracy for imbalanced datasets.

##🔧 Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn
- imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib, Seaborn
- Google Colab

##📊 Key Insights
- Fraud transactions typically involve lower amounts
- Time is not a strong indicator of fraud
- PCA components contain hidden fraud patterns
- Imbalance handling (SMOTE) significantly improves performance

##🚀 Future Enhancements
- Deploy model using Flask / FastAPI
- Implement real-time fraud detection API
- Add model monitoring for data drift
- Try LightGBM & CatBoost

##🙌 Author
Sandeep Kumar Mahto
M.Tech Data Science Student
Passionate about ML, Deep Learning, and Real-world AI applications
