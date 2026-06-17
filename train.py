import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# Configure terminal output encoding to prevent Unicode errors on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, "data", "processed", "sar_dataset.csv")

if not os.path.exists(file_path):
    file_path = "data/processed/sar_dataset.csv"
    if not os.path.exists(file_path):
        file_path = r"C:\Users\HP\Desktop\EDI's\EDI\data\processed\sar_dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(" Run preprocessing first to generate the dataset at data/processed/sar_dataset.csv")

df = pd.read_csv(file_path)

print("Dataset Loaded:", df.shape)


features = [
    'amount', 'time', 'amount_log', 'time_scaled',
    'amount_squared', 'high_amount_flag',
    'is_night', 'amount_bin', 'amount_ratio'
]

X = df[features]
y = df['fraud']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", sum(y_train), "fraud samples")


model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

print(" Model Trained")


y_probs = model.predict_proba(X_test)[:, 1]

threshold = 0.9
y_pred = (y_probs > threshold).astype(int)


print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(features, os.path.join(BASE_DIR, "features.pkl"))

print(" Model Saved")