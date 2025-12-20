Credit Card Fraud Detection using Machine Learning

This project implements a machine learning–based system to detect fraudulent credit card transactions. Due to the highly imbalanced nature of fraud data, SMOTE is used to balance the training dataset. Multiple models, including Logistic Regression and Random Forest, are trained and evaluated using precision, recall, F1-score, and ROC-AUC metrics. The Random Forest model achieves high performance with an accuracy of 99% and a ROC-AUC score of 0.98, making it effective for real-world fraud detection scenarios. Feature importance analysis is also performed to understand key factors contributing to fraud detection.

Technologies Used: Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn
Dataset: Kaggle – Credit Card Fraud Detection
## Dataset

This project uses the Credit Card Fraud Detection dataset from Kaggle.

Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Note:
Due to file size and licensing, the dataset is not included in this repository.
Please download the dataset from Kaggle and place `creditcard.csv` inside the `dataset/` folder.
