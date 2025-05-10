import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
import matplotlib.pyplot as plt

# Constants
SEQUENCE_LENGTH = 2400
MODEL_PATH = "cnn_bilstm_model.keras"
SCALER_PATH = "scaler.pkl"  # 🔥 Pre-saved scaler from training

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = load(SCALER_PATH)

# Streamlit UI
st.set_page_config(page_title="NeoGnosis", layout="centered")
st.title("NeoGnosis")
st.markdown("Let Technology Guide Your Delivery Decisions, with NeoGnosis.")

# Patient ID input
patient_id = st.text_input("👤 Enter Patient ID:")

# Upload maternal DB
maternal_file = st.file_uploader("📤 Upload Maternal Database (CSV)", type="csv")

# Upload CTG file
ctg_file = st.file_uploader("📤 Upload Fetal CTG File (.csv)", type="csv")

# Run only if all inputs are provided
# Run only if all inputs are provided
if patient_id and maternal_file and ctg_file:
    try:
        # Try converting patient ID
        patient_id_float = float(patient_id)

        # Load maternal data
        maternal_df = pd.read_csv(maternal_file)
        patient_row = maternal_df[maternal_df["ID"] == patient_id_float]

        if patient_row.empty:
            st.error("❌ Patient ID not found in the maternal data file.")
        else:
            st.subheader("Patient Data Preview:")
            st.write(patient_row)

            # Load and show fetal CTG data
            ctg_df = pd.read_csv(ctg_file)
            st.subheader("Fetal Data Preview:")
            st.dataframe(ctg_df)

            # Interpolate and preprocess CTG
            ctg_df = ctg_df.interpolate(method="linear", limit_direction="both")

            if "FHR" not in ctg_df.columns or "UC" not in ctg_df.columns:
                st.error("❌ CTG file must contain 'FHR' and 'UC' columns.")
            else:
                # Prepare maternal features
                maternal_features = patient_row.drop(columns=["ID", "Deliv. type"], errors='ignore').values
                maternal_scaled = scaler.transform(maternal_features)

                # Prepare CTG signal
                signal = ctg_df[["FHR", "UC"]].values
                if signal.shape[0] < SEQUENCE_LENGTH:
                    pad = np.zeros((SEQUENCE_LENGTH - signal.shape[0], 2))
                    signal = np.vstack([signal, pad])
                else:
                    signal = signal[:SEQUENCE_LENGTH]
                signal = np.expand_dims(signal, axis=0)

                # Prediction
                y_prob = model.predict([signal, maternal_scaled])[0][0]
                prediction = "Cesarean" if y_prob >= 0.5 else "Vaginal"
                confidence = y_prob if y_prob >= 0.5 else 1 - y_prob

                # Show result
                st.success(f"🧠 Prediction: {prediction}")
                st.markdown(f"**Confidence Score:** {confidence * 100:.2f}%")
                st.progress(int(confidence * 100))

                if confidence >= 0.80:
                    st.success("✅ Model is highly confident in its prediction.")
                elif confidence >= 0.60:
                    st.info("ℹ️ Model shows moderate confidence. Please verify with clinical evaluation.")
                elif confidence >= 0.45:
                    st.warning("⚠️ Model shows low confidence. Consider further analysis or clinical review.")
                else:
                    st.error("❗ Model is very uncertain. Strongly consider further clinical assessment.")

                # Plot FHR
                st.subheader("📈 Fetal Heart Rate (FHR)")
                fig1, ax1 = plt.subplots(figsize=(10, 3))
                ax1.plot(ctg_df["FHR"], color="blue")
                ax1.set_xlabel("Time (frames)")
                ax1.set_ylabel("FHR (bpm)")
                ax1.grid(True)
                st.pyplot(fig1)

                # Plot UC
                st.subheader("📈 Uterine Contractions (UC)")
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.plot(ctg_df["UC"], color="orange")
                ax2.set_xlabel("Time (frames)")
                ax2.set_ylabel("UC (mmHg)")
                ax2.grid(True)
                st.pyplot(fig2)

                st.markdown("---")
                st.caption("🩺 This tool is designed to assist healthcare professionals and does not replace clinical expertise.")

    except ValueError:
        st.error("⚠️ Please enter a valid numeric Patient ID.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
