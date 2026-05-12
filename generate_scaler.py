import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load maternal data
maternal_df = pd.read_csv("database/additional_db.csv")

signal_dir = "database/signals"
maternal_data = []

for _, row in maternal_df.iterrows():
    patient_id = str(int(row["ID"]))
    maternal_features = row.drop(["ID", "Deliv. type"]).values

    signal_path = os.path.join(signal_dir, f"{patient_id}.csv")
    if os.path.exists(signal_path):
        df = pd.read_csv(signal_path)
        if "FHR" in df.columns and "UC" in df.columns:
            maternal_data.append(maternal_features)

X_maternal = np.array(maternal_data)

# Impute missing values if any
X_maternal = pd.DataFrame(X_maternal).interpolate(axis=0).fillna(method='bfill').fillna(method='ffill').values

# Scale maternal data
scaler = StandardScaler()
scaler.fit(X_maternal)

dump(scaler, 'scaler.pkl')
print("scaler.pkl generated successfully.")
