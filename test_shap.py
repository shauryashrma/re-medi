import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import shap
from tensorflow.keras.models import load_model

model = load_model('cnn_bilstm_model.keras')

# Dummy data
signal = np.random.rand(1, 2400, 2)
maternal = np.random.rand(1, 34)

# Try GradientExplainer
try:
    print("Testing GradientExplainer...")
    explainer = shap.GradientExplainer(model, [np.random.rand(10, 2400, 2), np.random.rand(10, 34)])
    shap_values = explainer.shap_values([signal, maternal])
    print("GradientExplainer success. SHAP values length:", len(shap_values))
    print("SHAP maternal shape:", shap_values[1].shape)
except Exception as e:
    print("GradientExplainer failed:", e)

# Try KernelExplainer on the second input only, holding the first input constant
try:
    print("Testing KernelExplainer on maternal features...")
    def f(X):
        # X is (N, 34)
        N = X.shape[0]
        # duplicate signal N times
        sig = np.repeat(signal, N, axis=0)
        return model.predict([sig, X], verbose=0)
    
    explainer = shap.KernelExplainer(f, np.random.rand(10, 34))
    shap_values = explainer.shap_values(maternal)
    print("KernelExplainer success. SHAP values shape:", shap_values.shape)
except Exception as e:
    print("KernelExplainer failed:", e)
