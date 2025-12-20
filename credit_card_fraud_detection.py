import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
# 1. Load Dataset
data = pd.read_csv("/content/archive.zip")
print("Dataset Shape:", data.shape)
print(data.head())
# 2. Check Class Distribution
print("\nClass Distribution:")
print(data['is_fraud'].value_counts())
sns.countplot(x='is_fraud', data=data)
plt.title("Fraud vs Normal Transactions")
plt.show()
# 3. One-Hot Encode Categorical Features
categorical_cols = ['merchant_category']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
# 4. Feature Scaling
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data[['amount']])
data['Time_scaled'] = scaler.fit_transform(data[['transaction_hour']])
# 5. Split Features and Target
X = data.drop(['is_fraud', 'amount', 'transaction_hour'], axis=1)
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 6. Handle Imbalanced Data (SMOTE)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE:")
print(y_train_smote.value_counts())
# 7. Logistic Regression Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_smote, y_train_smote)
lr_pred = lr.predict(X_test)
print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_pred))
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print("Logistic Regression ROC-AUC:", lr_auc)
# 8. Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_smote, y_train_smote)
rf_pred = rf.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred))
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.show()
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print("Random Forest ROC-AUC:", rf_auc)
# 9. Feature Importance
importance = rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)
print("\nTop 10 Important Features:")
print(importance_df.head(10))
# 10. Save Model
joblib.dump(rf, "credit_card_fraud_model.pkl")
print("\nModel saved as credit_card_fraud_model.pkl")
