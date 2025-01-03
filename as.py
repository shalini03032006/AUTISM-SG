import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv("/content/AUTISM DSET.csv")
df.head()

print(df.info())
print(df.describe().T)
df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})

def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'
df['ageGroup'] = df['age'].apply(convertAge)

def add_feature(data):
    data['sum_score'] = data.filter(like='Score').sum(axis=1)
    if 'austim' in data.columns and 'used_app_before' in data.columns and 'jaundice' in data.columns:
        data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
    return data

df = add_feature(df)
df['age'] = df['age'].apply(lambda x: np.log(x) if x > 0 else x)

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

removal = ['ID', 'age_desc', 'used_app_before', 'austim']
features = df.drop(removal + ['Class/ASD'], axis=1)
target = df['Class/ASD']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_val_scaled = scaler.transform(X_val)

model = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_resampled, Y_resampled)

print(f"Best Parameters from GridSearch: {grid_search.best_params_}")


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC(kernel='rbf'),
    "Random Forest": RandomForestClassifier()
}

for model_name, model in models.items():
    model.fit(X_resampled, Y_resampled)
    y_train_pred = model.predict(X_resampled)
    y_val_pred = model.predict(X_val_scaled)

    print(f"### {model_name} Evaluation ###")
    print("Training ROC AUC:", roc_auc_score(Y_resampled, y_train_pred))
    print("Validation ROC AUC:", roc_auc_score(Y_val, y_val_pred))

  cv_scores = cross_val_score(model, X_resampled, Y_resampled, cv=5, scoring='roc_auc')
    print("Cross-Validation ROC AUC (mean):", cv_scores.mean())

  print("Confusion Matrix (Validation):")
  print(confusion_matrix(Y_val, y_val_pred))
  print("Classification Report (Validation):")
  print(classification_report(Y_val, y_val_pre

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_resampled, Y_resampled)
y_pred_dt = dt_classifier.predict(X_val_scaled)
print("### Decision Tree Classifier ###")
print("Accuracy:", accuracy_score(Y_val, y_pred_dt))
print("Classification Report:")
print(classification_report(Y_val, y_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred_dt))

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_resampled, Y_resampled)
y_pred_knn = knn_classifier.predict(X_val_scaled)

print("### KNN Classifier ###")
print("Accuracy:", accuracy_score(Y_val, y_pred_knn))
print("Classification Report:")
print(classification_report(Y_val, y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred_knn))


