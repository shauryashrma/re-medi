import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQUENCE_LENGTH = 2400
MODEL_PATH = os.path.join(BASE_DIR, "cnn_bilstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

print("Loading model and scaler...")
model = load_model(MODEL_PATH)
scaler = load(SCALER_PATH)

print("Loading maternal data...")
maternal_file = os.path.join(BASE_DIR, "database", "additional_db.csv")
maternal_df = pd.read_csv(maternal_file)

# Just use the first row that has a corresponding signal file
for _, row in maternal_df.iterrows():
    patient_id_float = float(row["ID"])
    ctg_file = os.path.join(BASE_DIR, "database", "signals", f"{int(patient_id_float)}.csv")
    if os.path.exists(ctg_file):
        break

patient_row = maternal_df[maternal_df["ID"] == patient_id_float]

print("Loading CTG data...")
ctg_df = pd.read_csv(ctg_file)
ctg_df = ctg_df.interpolate(method="linear", limit_direction="both")

maternal_features = patient_row.drop(columns=["ID", "Deliv. type"], errors='ignore').values
print(f"Maternal features shape: {maternal_features.shape}")

try:
    maternal_scaled = scaler.transform(maternal_features)
    print("Scaler transform successful.")
except ValueError as e:
    print(f"Scaler Error: {e}")

signal = ctg_df[["FHR", "UC"]].values
if signal.shape[0] < SEQUENCE_LENGTH:
    pad = np.zeros((SEQUENCE_LENGTH - signal.shape[0], 2))
    signal = np.vstack([signal, pad])
else:
    signal = signal[:SEQUENCE_LENGTH]
signal = np.expand_dims(signal, axis=0)

print(f"Signal shape: {signal.shape}")

try:
    y_prob = model.predict([signal, maternal_scaled])[0][0]
    print(f"Prediction probability: {y_prob}")
except Exception as e:
    print(f"Prediction Error: {e}")
    import traceback
    traceback.print_exc()
