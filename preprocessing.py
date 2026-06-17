import pandas as pd
import random
import os
import numpy as np
import sys

# Configure terminal output encoding to prevent Unicode errors on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)

file_path = os.path.join(BASE_DIR, "creditcard.csv")

if not os.path.exists(file_path):
    # Fallback to local cwd just in case
    file_path = "creditcard.csv"
    if not os.path.exists(file_path):
        file_path = r"C:\Users\HP\Desktop\EDI's\EDI\creditcard.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("creditcard.csv not found. Please place it in the root folder.")

df = pd.read_csv(file_path)

print("✅ Original Shape:", df.shape)

df = df.sample(100000, random_state=42)

df.rename(columns={
    'Amount': 'amount',
    'Time': 'time',
    'Class': 'fraud'
}, inplace=True)


locations = ['India', 'Dubai', 'USA', 'UK', 'Singapore']
types = ['transfer', 'withdrawal', 'payment']

df['location'] = [random.choice(locations) for _ in range(len(df))]
df['transaction_type'] = [random.choice(types) for _ in range(len(df))]


df['amount_log'] = np.log1p(df['amount'])
df['time_scaled'] = df['time'] / df['time'].max()
df['amount_squared'] = df['amount'] ** 2
df['high_amount_flag'] = (df['amount'] > 10000).astype(int)


df['is_night'] = (df['time'] % 86400 < 21600).astype(int)
df['amount_bin'] = pd.cut(df['amount'], bins=5, labels=False)
df['amount_ratio'] = df['amount'] / df['amount'].mean()


df.to_csv(os.path.join(BASE_DIR, "data", "processed", "sar_dataset.csv"), index=False)

print(" Final Dataset Saved:", df.shape)
print(df.head())