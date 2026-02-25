# train_model.py
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

CSV_PATH = "heart.csv"  
df = pd.read_csv(CSV_PATH)

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = ["age", "trestbps", "chol", "oldpeak", "thalachh"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

others = [c for c in X.columns if c not in numeric_features + categorical_features]
numeric_features = numeric_features + others

print("[INFO] numeric_features:", numeric_features)
print("[INFO] categorical_features:", categorical_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. SVM (RBF) + GridSearch
svm_rbf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", SVC(kernel="rbf", probability=True, random_state=42))
])

param_grid_rbf = {
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", 0.1, 1],
    "clf__class_weight": [None, "balanced"],
}

grid_rbf = GridSearchCV(
    svm_rbf,
    param_grid_rbf,
    cv=5,
    scoring="roc_auc",
    n_jobs=1,
    verbose=1
)

print("[INFO] Start GridSearchCV...")
grid_rbf.fit(X_train, y_train)

print("[INFO] Best params:", grid_rbf.best_params_)
print("[INFO] Best CV ROC AUC:", grid_rbf.best_score_)

best_model = grid_rbf.best_estimator_
y_proba = best_model.predict_proba(X_test)[:, 1]
print("[INFO] Test ROC AUC:", roc_auc_score(y_test, y_proba))

MODEL_PATH = "heart_svm_model.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")

print("[INFO] feature_names_in_:", best_model.feature_names_in_)


X_test_copy = X_test.copy()
X_test_copy["proba"] = y_proba

low_example  = X_test_copy.sort_values("proba").head(5)
high_example = X_test_copy.sort_values("proba").tail(5)

print(low_example[["proba"] + list(X_test.columns)])
print(high_example[["proba"] + list(X_test.columns)])
probas = y_proba  # best_model.predict_proba(X_test)[:, 1]

low_thr = np.quantile(probas, 0.33)
high_thr = np.quantile(probas, 0.66)

print("low_thr:", low_thr)
print("high_thr:", high_thr)