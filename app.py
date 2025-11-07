import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------------
# Load Saved Model
# ----------------------
with open("xgb_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------
# Prediction Function
# ----------------------
def predict_fraud(data, threshold=0.50):
    proba = model.predict_proba(data)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba

# ----------------------
# Streamlit UI
# ----------------------
st.title("💳 Credit Card Fraud Detection (XGBoost)")

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.50, 0.01)

option = st.radio("Input Method", ["Upload CSV", "Manual Input"])


if option == "Upload CSV":
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)

        required_cols = ["V17","V14","V12","V10","V16","V3","V7","V11",
                         "V4","V18","V1","V9","V5","V2"]

        # Check if CSV contains all required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            # ✅ Select ONLY the required columns
            df = df[required_cols]

            st.write("✅ Filtered Columns (Used for Prediction):")
            st.dataframe(df.head())

            if st.button("Predict"):
                preds, probas = predict_fraud(df, threshold)
                df["Fraud_Probability"] = probas
                df["Prediction"] = preds

                st.write("✅ Prediction Results:")
                st.dataframe(df)
                
# ----------------------
# ✅ Manual Input Mode
# ----------------------
else:
    st.write("Enter the 14 selected feature values:")

    features = ["V17","V14","V12","V10","V16","V3","V7","V11","V4","V18","V1","V9","V5","V2"]
    user_data = {}

    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            user_data[feature] = st.number_input(feature, value=0.0)

    df_user = pd.DataFrame([user_data])

    if st.button("Predict"):
        pred, proba = predict_fraud(df_user, threshold)
        st.write("Fraud Probability:", float(proba[0]))

        if pred[0] == 1:
            st.error("🚨 Fraud Detected!")
        else:
            st.success("✅ Not Fraud")
