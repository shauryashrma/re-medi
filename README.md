# Mode of Delivery Prediction using Deep Learning

This project predicts the **mode of delivery** (Normal or Cesarean) using deep learning models trained on **fetal CTG signals** (FHR and UC) and **maternal features**.

## Overview

Accurate prediction of delivery type can help improve maternal and fetal outcomes. This system combines clinical data and raw signals to build a predictive model using CNN and BiLSTM architectures.

## Models Used

- **CNN**: Captures spatial patterns from CTG signals.
- **LSTM / BiLSTM**: Learns temporal dependencies in FHR and UC.
- **CNN-BiLSTM**: Final hybrid model with superior performance.

## Features

- **Inputs**:
  - Maternal: age, BMI, BP, glucose levels, etc.
  - Fetal: FHR (Fetal Heart Rate), UC (Uterine Contraction)
- **Output**: Binary classification — Normal Delivery or Cesarean

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Divyax/NeoGnosis.git
   cd NeoGnosis

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
